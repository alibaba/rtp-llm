"""DSV4 decode forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style decode primitives as free functions so the Model
class stays thin:

* ``build_paged_pool_specs`` — per-attn_type paged pool geometry
* ``build_metadata_eager``   — DSv4DecodeAttnMetadata from raw attn_inputs
* ``forward_layers``         — per-layer loop body (embed → layers → reduce + norm)
* ``forward_decode``         — full decode arm (metadata dispatch + per-layer + packing)

Paired with :mod:`rtp_llm.models_py.modules.dsv4.prefill.forward`, which
does the same job for the prefill path.

Nothing here holds state. The CUDA-graph-captured metadata (kept alive
inside ``DSv4DecodeFmhaImpl``) is looked up by the caller; this module
only builds the *eager* metadata when ``fmha_impl`` isn't a persistent
decode impl.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4 import _forward_tensor_debug as _fwd_dbg
from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    DSV4_KERNEL_ROW_TAGS,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
    build_block_tables_for_tags,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    group_tags as _kv_cache_group_tags,
)
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import primary_attention_inputs

# DSV4 paged (FULL) + fixed/state pool tags, in the probe order the metadata
# allocator expects. SWA_KV lives on every layer; CSA/HCA/INDEXER only on the
# compressor layers.
_DSV4_DECODE_POOL_TAGS: Tuple[str, ...] = (
    SWA_KV,
    HCA_KV,
    INDEXER_KV,
    CSA_KV,
    CSA_STATE,
    HCA_STATE,
    INDEXER_STATE,
)


def _dsv4_kernel_tokens_per_block(kv_cache: Any) -> int:
    """No-fallback accessor for the FULL pools' kernel tokens-per-block.

    The tag-driven KVCache exposes this per cache group
    (``get_kernel_seq_size_per_block(tag)``); every DSV4 paged pool shares the
    one value ``DSV4CacheConfigHelper`` writes, so the first kernel-row tag the
    topology carries is authoritative. Surfaces the C++ propagation bug instead
    of silently sizing block tables with the wrong stride.
    """
    if kv_cache is None:
        raise RuntimeError(
            "DSV4 decode: kv_cache is None when sizing paged pool specs."
        )
    tags = _kv_cache_group_tags(kv_cache)
    getter = getattr(kv_cache, "get_kernel_seq_size_per_block", None)
    if getter is not None:
        for tag in DSV4_KERNEL_ROW_TAGS:
            if tag not in tags:
                continue
            try:
                ksb = int(getter(tag))
            except RuntimeError:
                continue
            if ksb > 0:
                return ksb
    raise RuntimeError(
        "DSV4 KVCache kernel_seq_size_per_block is unavailable for every paged "
        "tag (expected >0 for one of %r). group_tags=%r."
        % (list(DSV4_KERNEL_ROW_TAGS), tags)
    )


def _dsv4_pool_tokens_per_block(kv_cache: Any, tag: str) -> int:
    return require_pool_tokens_per_block(kv_cache, tag=str(tag))


def build_paged_pool_specs(
    kv_cache: Optional[Any],
    v4: Any,
    max_seq_len: Optional[int] = None,
) -> Dict[str, Tuple[int, int, int]]:
    """Per-cache-tag paged pool specs for metadata pre-allocation.

    Returns ``(entries_per_block, tokens_per_block, max_blocks_per_req)``.

    ``entries_per_block`` is derived from the framework pool tensor's
    stride on layer 0 (all layers share the same allocator geometry per
    cache tag).

    ``max_blocks_per_req`` MUST cover each pool's own block-table geometry
    plus the decode-step slack required by ``gen_num_per_cycle``. Under-sizing
    here truncates the framework block_table on copy in
    ``update_decode_metadata_in_place``, leaving zero block-ids in the
    unfilled tail; the captured graph then reads block_id=0 for real decode
    positions, computes a slot in pool block 0, and overruns ``pool_view`` →
    ``index_copy_`` OOB.
    """
    if kv_cache is None or not v4.layers:
        return {}
    if max_seq_len is None:
        max_seq_len = int(getattr(v4, "max_seq_len", 0)) or int(
            getattr(getattr(v4, "args", None), "max_seq_len", 0)
        )
        if max_seq_len <= 0:
            raise ValueError(
                "build_paged_pool_specs: max_seq_len required to size paged "
                "block tables to match the framework allocator."
            )
    # The framework exposes a different raw-token coverage for each pool.
    # In particular INDEXER_KV uses 256 raw tokens per kernel row while
    # the common/SWA table can use 1024.  Sizing every persistent metadata
    # table with the common geometry silently truncated the tagged indexer
    # table during CUDA-graph preparation and eventually caused an illegal
    # GPU access.  Compute the width per pool below from its own geometry.
    # ``_pool_entries_per_block`` reads ``self._kv_cache`` which is only
    # bound during ``Attention.forward_decode``'s try/finally. Caller
    # (decode/forward.forward_decode) invokes us BEFORE the layer forward
    # so every attention's ``self._kv_cache`` is None and every lookup
    # returns 0. Temporarily stash the framework handle on each layer's
    # attn while probing, then restore.
    #
    # SWA_KV lives on every layer, but CSA/HCA/INDEXER only live on the
    # compressor layers (layer 0/1 are SWA-only on DSV4). Probe the first
    # layer that has the pool — per-tag geometry is uniform across
    # the layers that own it.
    specs: Dict[str, Tuple[int, int, int]] = {}
    graph_table_slack = max(
        1, int(getattr(getattr(v4, "args", None), "gen_num_per_cycle", 1))
    )
    saved_kv: Dict[int, Any] = {}
    try:
        # #50: STATE pool block tables must also flow through metadata so
        # compressor/indexer can gather their fp32 state on each decode
        # step.  Include CSA_STATE / HCA_STATE / INDEXER_STATE alongside
        # the KV pools.
        for tag in _DSV4_DECODE_POOL_TAGS:
            for layer in v4.layers:
                attn = layer.attn
                if id(attn) not in saved_kv:
                    saved_kv[id(attn)] = (attn, attn._kv_cache)
                    attn._kv_cache = kv_cache
                entries_per_block = attn._pool_entries_per_block(tag)
                if entries_per_block > 0:
                    tokens_per_block = _dsv4_pool_tokens_per_block(
                        kv_cache,
                        tag,
                    )
                    # INDEXER_KV stores one entry per four raw tokens.  Its
                    # shared physical owner may report the wider owner-row
                    # coverage here, while the indexer block table advances
                    # after exactly ``entries_per_block * 4`` raw tokens.
                    # Metadata must follow the table's kernel geometry.
                    table_tokens_per_block = (
                        entries_per_block * 4
                        if tag == INDEXER_KV
                        else tokens_per_block
                    )
                    max_blocks_per_req = (
                        max_seq_len + table_tokens_per_block - 1
                    ) // table_tokens_per_block + graph_table_slack
                    specs[tag] = (
                        entries_per_block,
                        tokens_per_block,
                        max_blocks_per_req,
                    )
                    break
        return specs
    finally:
        for attn, prev_kv in saved_kv.values():
            attn._kv_cache = prev_kv


def build_metadata_eager(
    v4_args: Any,
    attn_inputs: Any,
    device: torch.device,
    paged_pool_specs: Dict[str, Tuple[int, int, int]],
    kv_cache: Optional[Any] = None,
    fp8_kv_cache: bool = False,
) -> Optional[Any]:  # DSv4DecodeAttnMetadata | None
    """Build ``DSv4DecodeAttnMetadata`` inline from framework attn inputs.

    ``attn_inputs`` is ``PyModelInputs.attention_inputs``: either a single
    ``PyAttentionInputs`` or the ``{tag: PyAttentionInputs}`` mapping DSV4 gets
    from the multi-group cache. Group-invariant fields are read off the primary
    entry; block tables are collected per tag.

    Only used on the eager path (``fmha_impl`` is None or not a
    ``DSv4DecodeFmhaImpl``). CUDA-graph capture has its own persistent
    metadata owned by ``DSv4DecodeFmhaImpl.metadata`` — the caller
    checks the fmha_impl type and picks.

    Returns ``None`` when the incoming batch is empty (bs == 0) so the
    caller can short-circuit to an empty ``PyModelOutputs``.

    Per-request first-token position is read from ``attn.sequence_lengths``
    (normal decode) or ``attn.prefix_lengths`` (target verify). The FP8
    branch passes ``attn`` straight through and lets
    :func:`build_decode_metadata_fp8` derive + clamp ``start_pos``
    internally via :func:`_build_start_pos_from_attention_inputs`; the
    BF16 branch builds + clamps ``start_pos`` here before calling the
    legacy :func:`build_decode_metadata`. Both clamp to
    ``[0, max_seq_len - q_len]`` so the whole ``[start_pos, start_pos + q_len)``
    window stays within KV/compressed pool capacity.
    """
    # Target verify is a multi-token decode. C++ MtpExecutor (see
    # MtpExecutor.cc:879-958) clears ``sequence_lengths`` and stashes the
    # prior decode positions into ``prefix_lengths``; ``input_lengths``
    # carries the uniform verify width = ``gen_num_per_cycle + 1``.
    attn = primary_attention_inputs(attn_inputs, kv_cache)
    if attn is None:
        return None
    is_target_verify = bool(getattr(attn, "is_target_verify", False))
    if is_target_verify:
        input_lengths_d = attn.input_lengths
        q_len = int(input_lengths_d[0]) if input_lengths_d.numel() > 0 else 1
        bs = int(attn.prefix_lengths.shape[0])
    else:
        q_len = 1
        bs = int(attn.sequence_lengths.shape[0])
    if bs == 0:
        return None

    max_s = int(v4_args.max_seq_len)

    # Pull per-tag block_tables + entries_per_block for the paged read/write
    # path. Eager allocates fresh per step (no graph capture, no
    # forbid_realloc).
    paged_block_tables: Dict[str, Any] = {}
    paged_entries_per_block: Dict[str, int] = {}
    paged_tokens_per_block: Dict[str, int] = {}
    if paged_pool_specs:
        # tokens_per_block is a pure geometry fact — publish it for every pool
        # the decode impl asked for, whether or not a block table arrived.
        for tag, spec in paged_pool_specs.items():
            paged_tokens_per_block[tag] = int(spec[1])
        tagged_block_tables = build_block_tables_for_tags(
            kv_cache, attn_inputs, paged_pool_specs.keys()
        )
        for tag, block_table in (tagged_block_tables or {}).items():
            paged_block_tables[tag] = block_table
            paged_entries_per_block[tag] = int(paged_pool_specs[tag][0])

    if fp8_kv_cache:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            build_decode_metadata_fp8,
        )

        return build_decode_metadata_fp8(
            attention_inputs=attn,
            q_len=q_len,
            window_size=int(v4_args.window_size),
            head_dim=int(v4_args.head_dim),
            max_seq_len=max_s,
            compress_ratios=list(v4_args.compress_ratios)[: v4_args.n_layers],
            index_topk=int(v4_args.index_topk),
            device=device,
            paged_block_tables=paged_block_tables or None,
            paged_pool_entries_per_block=paged_entries_per_block or None,
            paged_pool_tokens_per_block=paged_tokens_per_block or None,
        )

    # BF16 path: legacy ``build_decode_metadata`` takes a raw ``start_pos``
    # tensor. Derive + clamp here so it shares semantics with the FP8 branch
    # (whole [start_pos, start_pos+q_len) window inside KV capacity).
    from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
        build_decode_metadata,
    )

    if is_target_verify:
        start_pos = attn.prefix_lengths
    else:
        start_pos = attn.sequence_lengths
    start_pos = start_pos.to(device=device, dtype=torch.int32)
    max_start = max(0, max_s - q_len)
    start_pos = torch.clamp(start_pos, min=0, max=max_start)

    return build_decode_metadata(
        start_pos=start_pos,
        q_len=q_len,
        window_size=int(v4_args.window_size),
        head_dim=int(v4_args.head_dim),
        max_seq_len=max_s,
        compress_ratios=list(v4_args.compress_ratios)[: v4_args.n_layers],
        index_topk=int(v4_args.index_topk),
        device=device,
        paged_block_tables=paged_block_tables or None,
        paged_pool_entries_per_block=paged_entries_per_block or None,
    )


def forward_layers(
    v4: Any,
    kv_cache: Optional[Any],
    input_ids: torch.Tensor,  # [T_total]
    attn_metadata: Any,  # DSv4DecodeAttnMetadata
    prepare_hidden_fn: Optional[Any] = None,
) -> torch.Tensor:
    """qwen3-style decode per-layer loop. Same body shape as the prefill
    helper (:func:`rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers`)
    but dispatches to ``layer.forward_decode`` (FlashMLA / FP8 path)
    and threads the pre-built decode metadata.

    ``prepare_hidden_fn``, when given, replaces the default
    embed-and-expand step. Signature: ``fn(input_ids, meta) -> Tensor`` of
    shape ``[B, q_len, hc, dim]``. Used by MTP to splice the
    e_proj/h_proj fusion stage in front of the layer loop while sharing
    the rest of the decode body with the main model.
    """
    B = attn_metadata.batch_size
    q_len = attn_metadata.q_len_per_req
    input_ids = input_ids.reshape(-1)

    _rt_on = _rt.ENABLED
    if _rt_on:
        _rt.begin(seqlen=int(input_ids.numel()))
        if _rt._get_buf() is None:
            _rt_on = False

    if prepare_hidden_fn is None:
        h = v4.embed(input_ids).view(B, q_len, -1)  # [B, q_len, dim]
        if _rt_on:
            _rt.record("decode_embed_out", h)
        h = h.unsqueeze(2).repeat(1, 1, v4.hc_mult, 1)  # [B, q_len, hc, dim]
    else:
        h = prepare_hidden_fn(input_ids=input_ids, meta=attn_metadata)
    if _rt_on:
        _rt.record("decode_embed_hc_expanded", h)
    capture_ids = frozenset(v4.capture_aux_hidden_layer_ids)
    for layer_idx, layer in enumerate(v4.layers):
        h = layer.forward_decode(h, attn_metadata, input_ids, kv_cache=kv_cache)
        if layer_idx in capture_ids:
            v4.capture_aux_hidden(layer_idx, h)
        if _rt_on:
            _rt.record(f"decode_layer{layer.layer_id:02d}_out", h)
    if v4._mtp_hidden_buffer is not None:
        if capture_ids:
            # DSpARK mode: the buffer already holds this forward's aux rows
            # (written per selected layer above); only account for them.
            v4._note_aux_hidden_rows(
                h.size(0) * h.size(1), is_cuda_graph=attn_metadata.is_cuda_graph
            )
        else:
            _pre_hc_flat = h.flatten(-2).reshape(-1, h.size(-2) * h.size(-1))
            v4._write_mtp_hidden_buffer(
                _pre_hc_flat, is_cuda_graph=attn_metadata.is_cuda_graph
            )
    h = v4._hc_head_reduce(h)
    if _rt_on:
        _rt.record("decode_hc_reduced", h)
    # Framework RMSNorm wants 2D — collapse [B, q_len, dim] then view back.
    bsz, q_len, dim_ = h.shape
    h = v4.norm(h.reshape(bsz * q_len, dim_)).view(bsz, q_len, dim_)
    if _rt_on:
        _rt.record("decode_final_norm", h)
        step = getattr(v4, "_dbg_step", 0)
        _rt.dump(
            step=step,
            extra={
                "path": "decode",
                "input_ids_shape": tuple(input_ids.shape),
                "input_ids": input_ids.detach().cpu(),
                "start_pos": attn_metadata.start_pos[: attn_metadata.batch_size]
                .detach()
                .cpu(),
                "batch_size": int(attn_metadata.batch_size),
                "q_len": int(attn_metadata.q_len_per_req),
            },
        )
        v4._dbg_step = step + 1
    return h


def forward_decode(
    v4: Any,
    kv_cache: Optional[Any],
    v4_args: Any,
    inputs: Any,  # PyModelInputs
    fmha_impl: Any = None,  # Optional[DSv4DecodeFmhaImpl]
    prepare_hidden_fn: Optional[Any] = None,
) -> Any:  # PyModelOutputs
    """Batched decode arm — full orchestration used by
    ``DeepSeekV4Model.forward`` dispatcher.

    Two metadata paths:

    * **CUDA-graph path** (``fmha_impl`` is a ``DSv4DecodeFmhaImpl``): read
      ``fmha_impl.metadata`` as-is. It was populated either in
      ``DSv4DecodeFmhaImpl.__init__`` (initial dtype-check forward) or by
      C++ ``prepare_cuda_graph`` before each replay. Reading
      ``attn.sequence_lengths`` here during stream capture would trigger a
      CPU→CUDA copy that's illegal inside a graph.
    * **Eager path**: build :class:`DSv4DecodeAttnMetadata` inline via
      :func:`build_metadata_eager`.

    Input ``inputs.input_ids`` arrives flat ``[T_total]``; we view as
    ``[B, q_len]``, dispatch to :func:`forward_layers`, then re-pack to
    ``[T_total, dim]`` for the sampler.
    """
    from rtp_llm.models_py.modules.dsv4.decode.decode_fmha_impl import (
        DSv4DecodeFmhaImpl,
    )
    from rtp_llm.ops.compute_ops import PyModelOutputs

    # FP8 decode has a separate, non-inheriting FmhaImpl class. Include both
    # in the graph-path dispatch so the CUDA-graph capture (which passes an
    # ``fmha_impl`` via ``prepare_fmha_impl``) reads the impl's persistent
    # metadata instead of falling through to ``build_metadata_eager`` — the
    # eager path does CPU→GPU copies on ``attn.sequence_lengths`` which are
    # rejected inside a CUDA stream capture.
    _graph_impl_types: Tuple[type, ...] = (DSv4DecodeFmhaImpl,)
    try:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_fmha_impl import (
            DSv4DecodeFmhaImplFP8,
        )

        _graph_impl_types = (DSv4DecodeFmhaImpl, DSv4DecodeFmhaImplFP8)
    except ImportError:
        pass

    # ``attention_inputs`` is a ``{tag: PyAttentionInputs}`` mapping for the
    # multi-group DSV4 cache; keep the raw value for per-tag block-table
    # lookups and derive the group-invariant view for scalar fields.
    attn_inputs = inputs.attention_inputs
    attn = primary_attention_inputs(attn_inputs, kv_cache)
    # No nn.Parameter on V4Transformer anymore — pull device from a known-bound tensor.
    param_dev = v4.embed.weight.device

    input_ids = inputs.input_ids
    if input_ids.dim() == 0:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.device != param_dev:
        input_ids = input_ids.to(param_dev)
    input_ids = input_ids.reshape(-1)

    if isinstance(fmha_impl, _graph_impl_types):
        meta = fmha_impl.metadata
    else:
        paged_specs = build_paged_pool_specs(
            kv_cache, v4, max_seq_len=int(v4_args.max_seq_len)
        )
        meta = build_metadata_eager(
            v4_args,
            attn_inputs,
            param_dev,
            paged_specs,
            kv_cache=kv_cache,
            fp8_kv_cache=bool(getattr(v4, "fp8_kv_cache", False)),
        )
        if meta is None:
            # Empty batch (B == 0) — short-circuit with zero-row hidden.
            return PyModelOutputs(
                torch.zeros((0, v4_args.dim), dtype=torch.bfloat16, device=param_dev)
            )

    B = meta.batch_size
    q_len = meta.q_len_per_req
    _rt_on = _rt.ENABLED
    if _rt_on:
        _rt.begin(seqlen=int(input_ids.numel()))
        if _rt._get_buf() is None:
            _rt_on = False
        else:
            _rt.record("decode_input_ids", input_ids)

    h = forward_layers(
        v4,
        kv_cache,
        input_ids,
        meta,
        prepare_hidden_fn=prepare_hidden_fn,
    )  # [B, q_len, dim]
    hidden = h.reshape(B * q_len, v4_args.dim)  # packed [T_total, dim]
    if _fwd_dbg.enabled():
        _fwd_dbg.print_decode(
            hidden=hidden,
            input_ids_2d=input_ids.view(B, q_len),
            attn_inputs=attn,
            meta=meta,
            head_weight=getattr(v4, "head_weight", None),
            step=int(getattr(v4, "_dbg_step", 0)),
        )
    if _rt_on:
        _rt.record("decode_hidden", hidden)
        lm_logits = torch.mm(
            hidden.to(v4.head_weight.dtype), v4.head_weight.t()
        ).float()
        _rt.record("decode_lm_logits", lm_logits)
        top_k = min(16, lm_logits.size(-1))
        lm_top_values, lm_top_indices = torch.topk(lm_logits, k=top_k, dim=-1)
        _rt.record("decode_lm_top_values", lm_top_values)
        _rt.record("decode_lm_top_indices", lm_top_indices)
        extra = {
            "is_decode": True,
            "input_ids_shape": tuple(input_ids.shape),
            "input_ids": input_ids.detach().cpu(),
            "batch_size": int(B),
            "q_len": int(q_len),
        }
        seq_lens = getattr(attn, "sequence_lengths", None)
        if seq_lens is not None:
            extra["sequence_lengths"] = seq_lens.detach().cpu()
        start_pos = getattr(meta, "start_pos", None)
        if start_pos is not None:
            extra["start_pos"] = start_pos.detach().cpu()
        _rt.dump(step=v4._dbg_step, extra=extra)
        v4._dbg_step += 1
    return PyModelOutputs(hidden)
