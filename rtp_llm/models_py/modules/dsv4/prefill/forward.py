"""DSV4 prefill forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style prefill primitives as free functions so the Model
class stays thin:

* ``set_cp_info``                    — bind/clear Context-Parallel metadata on ``v4``
* ``forward_layers``                 — per-layer loop body (embed → layers → reduce → norm)
* unified cache-store registration via
  :func:`rtp_llm.models_py.modules.factory.attention.common.create_write_cache_store_impl`
* ``forward_prefill``                — full prefill arm (per-request loop over flat 1D input_ids)

Generic KV-cache lookup helpers (``build_block_tables_batched``) live
in :mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`.

Nothing in here holds state. ``DeepSeekV4Model.forward`` feeds in
``self.v4`` / ``self.kv_cache`` / ``self.parallelism_config`` explicitly.

Paired with :mod:`rtp_llm.models_py.modules.dsv4.decode.forward`, which
does the same job for the decode path.

----------------------------------------------------------------------
Context-Parallel (CP) prefill data flow
----------------------------------------------------------------------

CP repurposes the TP process group as the CP group (see
``ParallelismConfig::get_attn_tp_size`` — returns 1 when CP enabled).
The C++ ``ZigZagProcessor`` splits each request's padded prefill tokens
across the CP group with a zigzag layout. ``forward_layers`` consumes
the resulting per-rank metadata and builds a ``CPContext`` (in
``cp.py``) bound onto every Attention / Compressor / Indexer module
via ``v4._propagate_cp_ctx`` before the per-layer loop runs.

Per-rank inputs (rank-local, shaped for ``T_local = chunk_length``):
  * ``input_ids``                — token slice owned by this rank
  * ``attn.position_ids``        — framework-provided positions; under CP,
                                   ``forward_layers`` replaces these with
                                   CPContext's per-token request-absolute
                                   positions
  * ``attn.cu_seqlens``          — rank-local request boundaries
  * ``attn.input_lengths``       — rank-local per-req token count

Rank-invariant inputs:
  * ``attn.prefix_lengths``      — global per-req KV prefix length

Global view (held on ``CPContext``, derived once in
``build_cp_context``):
  * ``cp_ctx.input_lengths_global`` — full per-req length, =
    ``cp_info.prefill_actual_input_lengths_cpu``
  * ``cp_ctx.cu_seqlens_global``    — cumsum, used as ``query_start_loc``
    for SWA-pool write meta
  * ``cp_ctx.global_positions``     — GLOBAL absolute pos per rank-local
    token (zigzag-derived, per-request for B>=1)
  * ``cp_ctx.seq_len_full``         — total real prefill length

Per-layer pipeline under CP (compress_ratio == 0, SWA-only):
  1. ``_prefill_compute_qkv``: rank-local Q + KV → KV all-gathered to
     ``kv_full[seq_len_full, D]`` in GLOBAL request order
  2. ``_prefill_write_swa_fp8_paged``: every rank writes the GATHER'd
     KV to its own paged pool. ``slot_mapping`` is built from the
     global write trio (cu_seqlens_global / combined_seq_lens_global /
     seq_len_full) so all ranks' pools end up bit-identical.
  3. ``_attn_fp8_swa_via_kv_full`` (fresh, sp==0): rank-local Q over
     gathered KV. The varlen topk builder uses CP global positions plus
     global per-request cu_seqlens, so topk indices address rows in
     ``kv_full`` for B>=1.
  4. ``_attn_fp8_swa_via_concat`` (cont, sp>0): workspace ``[B, M, D]``
     with per-request prefix tails and new-K slots. ``combined_indices`` /
     ``combined_lens`` are built in Python because the Triton helper's
     ``pos = start_pos + token_idx_in_query`` formula assumes contiguous Q,
     which zigzag CP breaks.

CSA / HCA layers (compress_ratio == 4 / 128) add:
  * ``CompressorFP8.forward`` all-gathers KV/score then drops the
    rank-local ``meta`` and rebuilds ``state_slots`` / ``kv_slots`` from
    CPContext's full per-request positions.
  * ``IndexerFP8.prepare`` swaps ``input_lengths`` →
    ``cp_ctx.input_lengths_global`` for ``T_per_req`` so ks / ke /
    cu_kv_seqlens index into the per-rank pool's GLOBAL compressed-K
    extent. Nested compressor_meta is nulled for the same rebuild path.

Output: each layer's hidden state ``h`` is rank-local
``[T_local, hc, dim]`` — the framework's exit all-gather + strip-pad
gather (driven by ``cp_info.prefill_qkv_restore_indice`` /
``prefill_qkv_padding_mask``) reassembles the full sequence for the
next-layer / lm-head step.

Decode does NOT all-gather. Each rank's pool already holds the full
sequence's compressed entries (each rank wrote the gather'd new K
during prefill), so per-rank decode reads remain self-contained.

Padding-token slots are nulled via ``cp_info.prefill_qkv_padding_mask``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.cp import (
    build_cp_context,
    cp_all_gather_full_varlen,
)
from rtp_llm.models_py.modules.dsv4.fp8.prefill_meta import (
    build_and_propagate_prefill_meta_fp8,
    clear_prefill_meta_shared_fp8,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import build_block_tables_batched
from rtp_llm.models_py.modules.factory.attention.common import (
    create_write_cache_store_impl,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    # Kept behind TYPE_CHECKING to avoid an import cycle — ``transformer``
    # doesn't depend on ``prefill`` today but this guard makes that
    # non-load-bearing (module loads fine even if the cycle reappears).
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _build_positions_from_lengths(
    input_lengths: torch.Tensor,  # [B] int
    prefix_lengths: torch.Tensor,  # [B] int
    device: torch.device,
    total_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Synthesize per-token global positions ``[T_total]`` int64 when the
    framework didn't populate ``attn.position_ids`` (warmup / cudagraph
    capture path).

    For each request ``b`` with prefix ``sp[b]`` and input length ``L[b]``,
    emit ``sp[b], sp[b]+1, ..., sp[b]+L[b]-1``; concatenated across the batch.

    Must be CUDA-graph-capture-safe: callers pass GPU-resident tensors
    (``input_lengths`` / ``prefix_lengths``) during capture. Keep the
    body tensor-only so capture does not synchronize on scalar reads.
    """
    input_lengths = input_lengths.to(device=device, dtype=torch.int64)
    prefix_lengths = prefix_lengths.to(device=device, dtype=torch.int64)
    batch_size = int(input_lengths.numel())
    if total_tokens is None:
        total_tokens = int(input_lengths.sum().item())
    if batch_size == 0 or total_tokens == 0:
        return torch.zeros(0, dtype=torch.int64, device=device)

    token_offsets = torch.arange(total_tokens, dtype=torch.int64, device=device)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=device),
            input_lengths.cumsum(0),
        ],
        dim=0,
    )
    req_ids = torch.searchsorted(cu_seqlens[1:], token_offsets, right=True)
    req_ids = req_ids.clamp(max=batch_size - 1)
    local_offsets = token_offsets - cu_seqlens.gather(0, req_ids)
    return prefix_lengths.gather(0, req_ids) + local_offsets


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
    prepare_hidden_fn: Optional[Any] = None,
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
    owned KV regions are registered with the PD-disagg cache_store immediately
    after that layer's forward.
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
        prefix_offsets = 0
        prefix_lengths = getattr(attn_inputs, "prefix_lengths", None)
        if prefix_lengths is not None and prefix_lengths.numel() > 0:
            prefix_offsets = prefix_lengths.to(
                device=input_ids.device, dtype=torch.long
            )
        cp_ctx = build_cp_context(
            cp_info,
            cp_size,
            cp_rank,
            T_local,
            input_ids.device,
            position_offset=prefix_offsets,
        )
    v4._propagate_cp_ctx(cp_ctx)
    if cp_ctx is not None:
        # The framework's fallback position_ids are rank-local contiguous
        # after ZigZagProcessor rewrites input_lengths to CP chunk lengths.
        # DSV4 attention/indexer/compressor need the per-token absolute
        # request positions carried by CPContext.
        positions = cp_ctx.global_positions.to(
            device=positions.device, dtype=torch.long
        )
    positions = positions.reshape(-1).contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.reshape(-1).contiguous()

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
        write_cache_store_impl = create_write_cache_store_impl(attn_inputs, kv_cache)

    if prepare_hidden_fn is None:
        h = v4.embed(input_ids)  # [T_total, dim]
        if _rt_on:
            _rt.record("prefill_embed_out", h)
        h = h.unsqueeze(-2).repeat(1, v4.hc_mult, 1)  # [T_total, hc, dim]
    else:
        h = prepare_hidden_fn(input_ids=input_ids, positions=positions)
    if _rt_on:
        _rt.record("prefill_embed_hc_expanded", h)

    # FP8 KV-cache: hoist host-side prefill metadata once per ratio bucket
    # and broadcast to every layer's ``AttentionFP8._prefill_meta_shared``.
    # BF16 path doesn't need this; ``Attention`` rebuilds meta inside its
    # own forward.
    if v4.fp8_kv_cache:
        sp_int_for_meta = int(positions[0].item())
        sp_per_req: Optional[torch.Tensor] = None
        req_id_per_token: Optional[torch.Tensor] = None
        if cp_ctx is not None:
            # Under CP, rank-local token order is zigzagged. The first token of
            # each rank-local request chunk is therefore not necessarily the
            # request's absolute start position. Use CP metadata instead of
            # deriving request ids from rank-local cu_seqlens.
            sp_per_req = cp_ctx.prefix_lengths.to(
                device=positions.device, dtype=torch.int64
            ).contiguous()
            req_id_per_token = cp_ctx.req_id_per_token.to(
                device=positions.device, dtype=torch.int32
            ).contiguous()
        elif cu_seqlens is not None and cu_seqlens.numel() >= 2:
            starts = cu_seqlens[:-1].to(device=positions.device, dtype=torch.int64)
            sp_per_req = positions.index_select(0, starts).to(torch.int64).contiguous()
            req_id_per_token = (
                torch.searchsorted(
                    cu_seqlens.to(device=positions.device, dtype=torch.int64),
                    torch.arange(
                        int(cu_seqlens[-1].item()),
                        device=positions.device,
                        dtype=torch.int64,
                    ),
                    right=True,
                )
                .sub_(1)
                .to(torch.int32)
                .contiguous()
            )
        batch_size = 1
        if cu_seqlens is not None and cu_seqlens.numel() >= 2:
            batch_size = int(cu_seqlens.numel() - 1)
        input_lengths: Optional[torch.Tensor] = None
        prefix_lengths: Optional[torch.Tensor] = None
        max_seqlen_q = 0
        if attn_inputs is not None:
            il = getattr(attn_inputs, "input_lengths", None)
            if il is not None and il.numel() > 0:
                input_lengths = il.to(
                    device=positions.device, dtype=torch.int32
                ).contiguous()
                max_seqlen_q = int(input_lengths.max().item())
            pl = getattr(attn_inputs, "prefix_lengths", None)
            if pl is not None and pl.numel() > 0:
                prefix_lengths = pl.to(
                    device=positions.device, dtype=torch.int32
                ).contiguous()
        build_and_propagate_prefill_meta_fp8(
            v4,
            h,
            sp_int_for_meta,
            kv_cache,
            block_tables_by_type,
            sp_per_req=sp_per_req,
            cu_seqlens=cu_seqlens,
            batch_size=batch_size,
            input_lengths=input_lengths,
            prefix_lengths=prefix_lengths,
            position_ids=positions,
            req_id_per_token=req_id_per_token,
            max_seqlen_q=max_seqlen_q,
        )

    for layer_idx, layer in enumerate(v4.layers):
        h = layer(
            h,  # [T, hc, dim]
            input_ids,  # [T]
            positions,  # [T]
            cu_seqlens,  # [B+1]
            kv_cache=kv_cache,
            block_tables_by_type=block_tables_by_type,
        )  # [T, hc, dim]
        if _rt_on:
            _rt.record(f"prefill_layer{layer_idx:02d}_out", h)
        if write_cache_store_impl is not None:
            write_cache_store_impl(kv_cache.get_layer_caches(layer_idx))
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

    if v4.fp8_kv_cache:
        clear_prefill_meta_shared_fp8(v4)

    if getattr(v4, "_mtp_hidden_buffer", None) is not None:
        _pre_hc_flat = h.flatten(-2)
        if cp_ctx is not None and cp_ctx.cp_size > 1:
            _pre_hc_flat = cp_all_gather_full_varlen(_pre_hc_flat, cp_ctx)
        v4._write_mtp_hidden_buffer(_pre_hc_flat, is_cuda_graph=False)

    # _hc_head_reduce is flat-native: [T, hc, dim] -> [T, dim].
    # Framework ``RMSNorm`` expects 2D, which matches the [T, dim] shape here.
    h = v4._hc_head_reduce(h)  # [T, dim]
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
        lm_logits = torch.mm(
            last_h.to(v4.head_weight.dtype), v4.head_weight.t()
        ).float()
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


def forward_prefill(
    v4: V4Transformer,
    kv_cache: Optional[KVCache],
    parallelism_config: Optional[ParallelismConfig],
    inputs: PyModelInputs,
    prepare_hidden_fn: Optional[Any] = None,
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
    ``indexer.py``) consumes the cu_seqlens-aware metadata directly; under CP
    the per-layer setup swaps in CPContext's request-absolute positions and
    full-length write-side view.

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
    # warmup / cudagraph capture path doesn't populate position_ids —
    # synthesize from (prefix_lengths, input_lengths). Prefer ``_d`` (GPU)
    # variants when available: during cudagraph capture, the host-side
    # ``input_lengths`` / ``prefix_lengths`` are pinned int32 CPU tensors,
    # but a dtype-converting ``.to(device=..., dtype=int64)`` on a pinned
    # tensor produces an unpinned intermediate which capture rejects.
    if positions is None:
        il_d = attn.input_lengths
        pl_d = attn.prefix_lengths
        input_lens = il_d if il_d.numel() > 0 else attn.input_lengths
        prefix_lens = pl_d if pl_d.numel() > 0 else attn.prefix_lengths
        positions = _build_positions_from_lengths(
            input_lens,
            prefix_lens,
            input_ids.device,
            total_tokens=int(input_ids.numel()),
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
        prepare_hidden_fn=prepare_hidden_fn,
    )  # [T_total, dim]
    return PyModelOutputs(hidden)
