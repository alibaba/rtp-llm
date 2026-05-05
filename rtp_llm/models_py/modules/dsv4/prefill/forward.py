"""DSV4 prefill forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style prefill primitives as free functions so the Model
class stays thin:

* ``set_cp_info``                    — bind/clear Context-Parallel metadata on ``v4``
* ``forward_layers``                 — per-layer loop body (embed → layers → reduce → norm)
* ``DSv4WriteCacheStoreOp``          — per-layer PD-disagg cache_store writer (qwen3-style)
* ``create_dsv4_write_cache_store_impl`` — factory helper mirroring
  :func:`common.create_write_cache_store_impl`
* ``forward_prefill``                — full prefill arm (per-request loop over flat 1D input_ids)

Generic KV-cache lookup helpers (``gid_for``, ``build_block_tables_batched``) live
in :mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.

Nothing in here holds state. ``DeepSeekV4Model.forward`` feeds in
``self.v4`` / ``self.kv_cache`` / ``self.parallelism_config`` explicitly.

Paired with :mod:`rtp_llm.models_py.modules.dsv4.decode.forward`, which
does the same job for the decode path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    build_block_tables_batched,
    gid_for,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    KVCacheRegionName,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    # Kept behind TYPE_CHECKING to avoid an import cycle — ``transformer``
    # doesn't depend on ``prefill`` today but this guard makes that
    # non-load-bearing (module loads fine even if the cycle reappears).
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


# ---- DSV4 per-layer pool set ------------------------------------------------
# Each layer's compress_ratio selects the set of pools it participates in:
#   0   -> SWA-only: {SWA_KV}
#   4   -> CSA layer: {SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE, CSA_STATE}
#   128 -> HCA layer: {SWA_KV, HCA_KV, HCA_STATE}
_POOL_SET_BY_RATIO: Dict[int, frozenset] = {
    0: frozenset((SWA_KV,)),
    4: frozenset((SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE, CSA_STATE)),
    128: frozenset((SWA_KV, HCA_KV, HCA_STATE)),
}

# Attn types written as a LINEAR ring — their block_id rows are padded to
# the group's full max_blocks_num but only the first 2 entries are valid
# (DSV4 linear_fixed_cap=2). Tail is junk zero padding that would register
# wrong block_ids in cache_store.
_LINEAR_ATTN_TYPES: frozenset = frozenset((INDEXER_STATE, CSA_STATE, HCA_STATE, SWA_KV))


def _build_positions_from_lengths(
    input_lengths: torch.Tensor,  # [B] int
    prefix_lengths: torch.Tensor,  # [B] int
    device: torch.device,
) -> torch.Tensor:
    """Synthesize per-token global positions ``[T_total]`` int64 when the
    framework didn't populate ``attn.position_ids`` (warmup path).

    For each request ``b`` with prefix ``sp[b]`` and input length ``L[b]``,
    emit ``sp[b], sp[b]+1, ..., sp[b]+L[b]-1``; concatenated across the batch.
    """
    input_lengths = input_lengths.to(device=device, dtype=torch.int64)
    prefix_lengths = prefix_lengths.to(device=device, dtype=torch.int64)
    batch_size = int(input_lengths.numel())
    segments = []
    for b in range(batch_size):
        length = int(input_lengths[b].item())
        start = int(prefix_lengths[b].item())
        segments.append(
            torch.arange(start, start + length, dtype=torch.int64, device=device)
        )
    if not segments:
        return torch.zeros(0, dtype=torch.int64, device=device)
    return torch.cat(segments, dim=0)


def set_cp_info(
    v4: V4Transformer,
    parallelism_config: Optional[ParallelismConfig],
    attn: Optional[PyAttentionInputs],
    is_prefill: bool,
) -> None:
    """Stash per-forward Context-Parallel metadata on ``v4`` so
    :func:`forward_layers` can build + propagate the derived
    ``CPContext`` when it enters the per-layer loop.

    Clears with ``(None, 1, 0)`` when CP is off so no stale ctx leaks
    from a prior request (warmup, etc.).
    """
    cp_enabled = (
        parallelism_config is not None
        and getattr(parallelism_config, "prefill_cp_config", None) is not None
        and parallelism_config.prefill_cp_config.is_enabled()
        and is_prefill
        and attn is not None
        and getattr(attn, "context_parallel_info", None) is not None
    )
    if cp_enabled:
        v4.set_cp_info(
            cp_info=attn.context_parallel_info,
            cp_size=int(parallelism_config.tp_size),
            cp_rank=int(parallelism_config.tp_rank),
        )
    else:
        v4.set_cp_info(None, 1, 0)


def forward_layers(
    v4: V4Transformer,
    kv_cache: Optional[KVCache],
    input_ids: torch.Tensor,  # [T_total] flat 1D
    positions: torch.Tensor,  # [T_total] int64 — per-token global absolute pos
    cu_seqlens: torch.Tensor,  # [B+1] int64 — request boundaries
    block_tables_by_type: Optional[Dict[int, torch.Tensor]],
    attn_inputs: Optional[PyAttentionInputs] = None,
) -> torch.Tensor:
    """Flat per-layer loop — vLLM-aligned layout.

    Shapes:
      * ``input_ids``   ``[T_total]``    — flat tokens across the forward's requests
      * ``positions``   ``[T_total]``    — per-token global absolute position (RoPE)
      * ``cu_seqlens``  ``[B+1]``        — per-request cumulative-token prefix sum
      * ``hidden``      ``[T_total, hc, dim]`` — internal, flat in the token axis
      * returns         ``[T_total, dim]`` — pre-lm-head, engine applies lm_head

    The ``B`` axis is collapsed out of ``input_ids`` / ``hidden`` entirely,
    matching vLLM's ``DeepseekV4`` (``deepseek_v4.py:1310-1317``). Per-request
    bookkeeping that still needs request boundaries (block-table lookups,
    compressor/indexer per-row state) is carried by ``cu_seqlens``.

    **Stage-2 compat shim**: ``Block.forward`` is now flat-native (accepts
    ``[T, hc, dim]`` + 1D ``input_ids`` / ``positions`` / ``cu_seqlens``)
    so the layer call site has no unsqueeze/squeeze. ``attention.py`` /
    ``compressor.py`` / ``indexer.py`` still consume ``[B=1, T, hc, dim]``
    internally — ``Block.forward`` re-wraps them. ``_hc_head_reduce`` +
    ``norm`` also still assume a 4D input (``dim=2`` for the hc reduction),
    so the reduce + norm pair here is still wrapped until ``transformer.py``
    is flattened.

    When ``attn_inputs`` is provided AND cache_store is active, each layer's
    KV pool write is registered with the PD-disagg cache_store immediately
    after that layer's forward — matches the qwen3-style per-layer ownership
    model (see :class:`DSv4WriteCacheStoreOp`).
    """
    # Build + propagate CP context once per prefill step. Under CP the
    # caller hands us a per-rank chunk slice (T_local = chunk_length),
    # and each attn / compressor / indexer reads ``cp_ctx`` off the
    # module to compute its own per-token positions. Without CP we pass
    # None to clear any stale context from a prior forward (warmup).
    cp_info = getattr(v4, "_cp_info", None)
    cp_size = getattr(v4, "_cp_size", 1)
    cp_rank = getattr(v4, "_cp_rank", 0)
    cp_ctx = None
    if cp_info is not None and cp_size > 1:
        T_local = int(input_ids.size(0))
        prefix_length = 0
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if prefix_lengths is not None and prefix_lengths.numel() > 0:
            prefix_length = int(prefix_lengths.reshape(-1)[0].item())
        cp_ctx = build_cp_context(
            cp_info,
            cp_size,
            cp_rank,
            T_local,
            input_ids.device,
            position_offset=prefix_length,
        )
    v4._propagate_cp_ctx(cp_ctx)

    # MOEDBG hook (mirrors V4Transformer.forward standalone path so the
    # smoke / production prefill path produces the same per-layer dump
    # consumed by /tmp/moedbg_runs diff scripts).  Read once per forward.
    _rt_on = _rt.ENABLED
    if _rt_on:
        _rt.begin(seqlen=int(input_ids.size(0)))
        if _rt._get_buf() is None:
            _rt_on = False

    # Build the per-layer cache_store writer once per forward. Active
    # only on prefill calls with cache_store_inputs bound; otherwise
    # ``write_cache_store_impl`` is None and the per-layer call site is
    # a cheap None check.
    write_cache_store_impl = None
    if kv_cache is not None and attn_inputs is not None:
        write_cache_store_impl = create_dsv4_write_cache_store_impl(
            attn_inputs,
            [layer.attn.compress_ratio for layer in v4.layers],
        )

    h = v4.embed(input_ids)  # [T_total, dim]
    if _rt_on:
        _rt.record("prefill_embed_out", h)
    h = h.unsqueeze(-2).repeat(1, v4.hc_mult, 1)  # [T_total, hc, dim]
    if _rt_on:
        _rt.record("prefill_embed_hc_expanded", h)

    for layer_idx, layer in enumerate(v4.layers):
        h = layer(
            h,  # [T, hc, dim]
            input_ids,  # [T]
            positions,  # [T]
            cu_seqlens,  # [B+1]
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
            attn_inputs=attn_inputs,
        )  # [T, hc, dim]
        if _rt_on:
            _rt.record(f"prefill_layer{layer_idx:02d}_out", h)
        if write_cache_store_impl is not None:
            write_cache_store_impl(kv_cache, layer_idx)
        if _rt_on:
            _rt.record(f"layer{layer_idx:02d}_out", h)
            if cp_ctx is None:
                layer_last = h[-1:].contiguous()
            else:
                layer_last_pos = cp_ctx.seq_len_total - 1
                layer_last_mask = (
                    cp_ctx.global_positions == layer_last_pos
                ) & cp_ctx.local_is_real
                layer_last = h[layer_last_mask].contiguous()
                dbg_pos = getattr(_rt, "_DBG_GLOBAL_POS", -1)
                if dbg_pos >= 0:
                    layer_pos_mask = (
                        cp_ctx.global_positions == dbg_pos
                    ) & cp_ctx.local_is_real
                    _rt.record(
                        f"layer{layer_idx:02d}_pos{dbg_pos}",
                        h[layer_pos_mask].contiguous(),
                    )
                layer_tail_mask = (
                    (cp_ctx.global_positions >= max(cp_ctx.seq_len_total - 128, 0))
                    & (cp_ctx.global_positions < cp_ctx.seq_len_total)
                    & cp_ctx.local_is_real
                )
                _rt.record(
                    f"layer{layer_idx:02d}_tail128", h[layer_tail_mask].contiguous()
                )
            _rt.record(f"layer{layer_idx:02d}_last", layer_last)

    # _hc_head_reduce is flat-native: [T, hc, dim] -> [T, dim].
    # Framework ``RMSNorm`` expects 2D, which matches the [T, dim] shape here.
    v4._hc_head_positions = positions
    h = v4._hc_head_reduce(h)  # [T, dim]
    v4._hc_head_positions = None
    if _rt_on:
        _rt.record("prefill_hc_reduced", h)
    h = v4.norm(h)  # [T, dim]
    if _rt_on:
        _rt.record("prefill_final_norm", h)
        if cp_ctx is None:
            last_h = h[-1:].contiguous()
        else:
            last_pos = cp_ctx.seq_len_total - 1
            last_mask = (cp_ctx.global_positions == last_pos) & cp_ctx.local_is_real
            last_h = h[last_mask].contiguous()
        _rt.record("lm_last_hidden", last_h)
        lm_logits = torch.mm(last_h.float(), v4.head_weight.t()).float()
        _rt.record("lm_logits_last", lm_logits)
        top_k = min(16, lm_logits.size(-1))
        lm_top_values, lm_top_indices = torch.topk(lm_logits, k=top_k, dim=-1)
        _rt.record("lm_top_values", lm_top_values)
        _rt.record("lm_top_indices", lm_top_indices)

    if _rt_on:
        extra: dict = {
            "input_ids_shape": tuple(input_ids.shape),
            "input_ids": input_ids.detach().cpu(),
            "path": "prefill",
            "positions": positions.detach().cpu(),
            "cu_seqlens": cu_seqlens.detach().cpu(),
        }
        if cp_ctx is not None:
            extra.update(
                {
                    "cp_size": cp_ctx.cp_size,
                    "cp_rank": cp_ctx.cp_rank,
                    "chunk_length": cp_ctx.chunk_length,
                    "padded_seq_len": cp_ctx.padded_seq_len,
                    "seq_len_full": cp_ctx.seq_len_full,
                    "prefix_length": cp_ctx.prefix_length,
                    "seq_len_total": cp_ctx.seq_len_total,
                    "relative_positions": cp_ctx.relative_positions.detach().cpu(),
                    "global_positions": cp_ctx.global_positions.detach().cpu(),
                    "unpad_restore": cp_ctx.unpad_restore.detach().cpu(),
                    "local_is_real": cp_ctx.local_is_real.detach().cpu(),
                }
            )
        else:
            extra.update(
                {
                    "cp_size": 1,
                    "cp_rank": 0,
                    "seq_len_full": int(input_ids.size(0)),
                    "prefix_length": 0,
                    "seq_len_total": int(input_ids.size(0)),
                }
            )
        if attn_inputs is not None:
            for name in ("input_lengths", "prefix_lengths", "sequence_lengths"):
                value = getattr(attn_inputs, name, None)
                if value is not None and value.numel() > 0:
                    extra[name] = value.detach().cpu()
        step = getattr(v4, "_dbg_step", 0)
        _rt.dump(step=step, extra=extra)
        v4._dbg_step = step + 1
    return h  # [T, dim]


class DSv4WriteCacheStoreOp(nn.Module):
    """Per-layer PD-disagg cache_store writer for DSV4.

    Mirrors the :class:`rtp_llm.models_py.modules.base.common.kvcache_store
    .WriteCacheStoreOp` pattern used by qwen3/xqa, but adapted to DSV4's
    per-layer 5-pool layout:

    * SWA-only layer: ``{SWA_KV}``
    * CSA layer (ratio=4): ``{SWA_KV, CSA_KV, INDEXER_KV, INDEXER_STATE, CSA_STATE}``
    * HCA layer (ratio=128): ``{SWA_KV, HCA_KV, HCA_STATE}``

    Each pool has its own ``(head_dim × entries_per_block × dtype)``
    layout and its own block_id table. The shared
    ``compute_ops.write_cache_store`` path assumes one pool per layer, so
    this Op calls it once per (layer × pool), passing the pool-specific
    block_id table, raw ``[num_blocks, stride_bytes]`` tensor,
    ``group_id`` (used in the ``"_g{gid}"`` cache key suffix that decode
    mirrors in ``DecodeRpcServer``) and per-pool stride.

    Invariants bound at construct time:
      * ``attn_inputs``      — ``input_lengths``, ``prefix_lengths``,
        ``cache_store_inputs``, ``kv_cache_kernel_block_id_host_by_group``
      * ``compress_ratios``  — list of per-layer ratios so ``forward``
        can pick the right pool set without reaching back into ``v4``

    Call convention mirrors qwen3/xqa: ``impl(kv_cache, layer_idx)``.
    """

    # int id → pybind ``KVCacheRegionName`` enum. Built at class scope so
    # every instance shares the same table; ``attn_type.py`` stays free
    # of a ``compute_ops`` (C++ .so) import dependency.
    _ATTN_TYPE_ENUM_BY_INT: Dict[int, KVCacheRegionName] = {
        CSA_KV: KVCacheRegionName.CSA_KV,
        HCA_KV: KVCacheRegionName.HCA_KV,
        INDEXER_KV: KVCacheRegionName.INDEXER_KV,
        INDEXER_STATE: KVCacheRegionName.INDEXER_STATE,
        CSA_STATE: KVCacheRegionName.CSA_STATE,
        HCA_STATE: KVCacheRegionName.HCA_STATE,
        SWA_KV: KVCacheRegionName.SWA_KV,
    }

    def __init__(
        self,
        attn_inputs: PyAttentionInputs,
        compress_ratios: List[int],
    ):
        super().__init__()
        self.attn_inputs = attn_inputs
        self.compress_ratios = compress_ratios
        # Invariant references — safe to capture once because they don't
        # change across the forward.
        self.input_lengths: torch.Tensor = attn_inputs.input_lengths
        cp_info = getattr(attn_inputs, "context_parallel_info", None)
        if cp_info is not None:
            actual_lengths = getattr(cp_info, "prefill_actual_input_lengths_cpu", None)
            if actual_lengths is not None and actual_lengths.numel() > 0:
                self.input_lengths = actual_lengths
        self.prefix_lengths: torch.Tensor = attn_inputs.prefix_lengths
        self.cache_store_inputs = attn_inputs.cache_store_inputs
        self.by_group_host: List[torch.Tensor] = (
            attn_inputs.kv_cache_kernel_block_id_host_by_group
        )

    def forward(self, kv_cache: KVCache, layer_idx: int) -> None:
        import rtp_llm.ops.compute_ops as compute_ops

        ratio = self.compress_ratios[layer_idx]
        pool_set = _POOL_SET_BY_RATIO.get(ratio, _POOL_SET_BY_RATIO[0])
        for attn_type in pool_set:
            # Model is group-oblivious: it only knows which attn_types this
            # layer participates in. Resolve (layer, attn_type) → gid via
            # the DSV4 dense gid list populated by NormalModelInputGatherer.
            gid = gid_for(kv_cache, self.attn_inputs, layer_idx, attn_type)
            if gid < 0 or gid >= len(self.by_group_host):
                continue
            attn_type_enum = self._ATTN_TYPE_ENUM_BY_INT.get(attn_type)
            if attn_type_enum is None:
                continue
            block_ids_2d: torch.Tensor = self.by_group_host[gid]
            if block_ids_2d is None or block_ids_2d.numel() == 0:
                continue
            layer_kv: LayerKVCache = kv_cache.get_layer_cache(layer_idx, attn_type_enum)
            compute_ops.write_cache_store(
                self.input_lengths,
                self.prefix_lengths,
                block_ids_2d,
                self.cache_store_inputs,
                layer_kv,
            )


def create_dsv4_write_cache_store_impl(
    attn_inputs: PyAttentionInputs,
    compress_ratios: List[int],
) -> Optional[DSv4WriteCacheStoreOp]:
    """Factory mirroring :func:`common.create_write_cache_store_impl`:
    return an active :class:`DSv4WriteCacheStoreOp` only when this is a
    prefill step with cache_store bound AND a populated per-group block-id
    host tensor; else ``None`` so the caller can no-op cheaply.
    """
    if not getattr(attn_inputs, "is_prefill", False):
        return None
    if getattr(attn_inputs, "cache_store_inputs", None) is None:
        return None
    by_group_host = getattr(attn_inputs, "kv_cache_kernel_block_id_host_by_group", None)
    if not by_group_host:
        return None
    return DSv4WriteCacheStoreOp(attn_inputs, compress_ratios)


def forward_prefill(
    v4: V4Transformer,
    kv_cache: Optional[KVCache],
    parallelism_config: Optional[ParallelismConfig],
    inputs: PyModelInputs,
) -> PyModelOutputs:
    """Prefill dispatcher — single :func:`forward_layers` call on the full
    flat ``[T_total]`` batch (vLLM-aligned).

    Pulls flat metadata off :attr:`PyModelInputs.attention_inputs` directly:

    * ``positions``  = ``attn.position_ids``     — ``[T_total]`` int32 global pos
    * ``cu_seqlens`` = ``attn.cu_seqlens``       — ``[B+1]`` int32 prefix sum
    * ``block_tables_by_type`` = Dict[attn_type, [B, max_blocks]] — full-batch
      block tables (B axis = request axis), built via
      :func:`build_block_tables_batched`.

    Downstream (``block.py`` / ``attention.py`` / ``compressor.py`` /
    ``indexer.py``) still has to be made cu_seqlens-aware to honour per-request
    boundaries correctly when ``B > 1``; until then the single-call path will
    compute as if the flat ``T_total`` were one giant sequence. ``B==1`` is
    bit-equal to the prior per-request loop.

    Returns ``PyModelOutputs`` with ``[T_total, dim]`` pre-lm-head hidden.
    """
    attn = inputs.attention_inputs

    # Context-Parallel setup must precede the per-layer loop because
    # forward_layers reads v4._cp_info to build the CP context.
    set_cp_info(v4, parallelism_config, attn, is_prefill=True)

    input_ids: torch.Tensor = inputs.input_ids  # [T_total] flat 1D

    # Framework already populates these — don't recompute.
    #  * ``attn.cu_seqlens``   : [B+1]     per-request cumulative prefix sum
    #  * ``attn.position_ids`` : [T_total] per-token global absolute position
    cu_seqlens = attn.cu_seqlens
    positions = attn.position_ids
    # warmup path doesn't populate position_ids — synthesize from
    # (prefix_lengths, input_lengths).
    if positions is None:
        positions = _build_positions_from_lengths(
            attn.input_lengths, attn.prefix_lengths, input_ids.device
        )

    block_tables_by_type = build_block_tables_batched(kv_cache, attn)

    hidden = forward_layers(
        v4,
        kv_cache,
        input_ids,
        positions,
        cu_seqlens,
        block_tables_by_type,
        attn_inputs=attn,
    )  # [T_total, dim]
    return PyModelOutputs(hidden)
