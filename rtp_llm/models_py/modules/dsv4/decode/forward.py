"""DSV4 decode forward helpers — extracted from ``DeepSeekV4Model``.

Exposes qwen3-style decode primitives as free functions so the Model
class stays thin:

* ``build_paged_pool_specs`` — per-attn_type (entries_per_block, max_blocks_per_req)
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

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)


def build_paged_pool_specs(
    kv_cache: Optional[Any],
    v4: Any,
) -> Dict[int, Tuple[int, int]]:
    """Per-attn_type ``(entries_per_block, max_blocks_per_req)`` for the
    decode-FMHA impl's metadata pre-allocation.

    ``entries_per_block`` is derived from the framework pool tensor's
    stride on layer 0 (all layers share the same allocator geometry per
    attn_type). Pools 3-6 get fixed 2 blocks/req; SWA gets 2 blocks (256
    entries/block × 2 = 512-slot ring, plenty for win).
    """
    if kv_cache is None or not v4.layers:
        return {}
    # ``_pool_entries_per_block`` reads ``self._kv_cache`` which is only
    # bound during ``Attention.forward_decode``'s try/finally. Caller
    # (decode/forward.forward_decode) invokes us BEFORE the layer forward
    # so every attention's ``self._kv_cache`` is None and every lookup
    # returns 0. Temporarily stash the framework handle on each layer's
    # attn while probing, then restore.
    #
    # SWA_KV lives on every layer, but CSA/HCA/INDEXER only live on the
    # compressor layers (layer 0/1 are SWA-only on DSV4). Probe the first
    # layer that has the pool — per-attn_type geometry is uniform across
    # the layers that own it.
    specs: Dict[int, Tuple[int, int]] = {}
    saved_kv: Dict[int, Any] = {}
    try:
        # #50: STATE pool block tables must also flow through metadata so
        # compressor/indexer can gather their fp32 state on each decode
        # step.  Include CSA_STATE / HCA_STATE / INDEXER_STATE alongside
        # the KV pools.
        for attn_type in (
            SWA_KV,
            HCA_KV,
            INDEXER_KV,
            CSA_KV,
            CSA_STATE,
            HCA_STATE,
            INDEXER_STATE,
        ):
            for layer in v4.layers:
                attn = layer.attn
                if id(attn) not in saved_kv:
                    saved_kv[id(attn)] = (attn, attn._kv_cache)
                    attn._kv_cache = kv_cache
                entries_per_block = attn._pool_entries_per_block(attn_type)
                if entries_per_block > 0:
                    specs[attn_type] = (entries_per_block, 2)
                    break
        return specs
    finally:
        for attn, prev_kv in saved_kv.values():
            attn._kv_cache = prev_kv


def build_metadata_eager(
    v4_args: Any,
    attn: Any,
    device: torch.device,
    paged_pool_specs: Dict[int, Tuple[int, int]],
    kv_cache: Optional[Any] = None,
) -> Optional[Any]:  # DSv4DecodeAttnMetadata | None
    """Build ``DSv4DecodeAttnMetadata`` inline from framework attn inputs.

    Only used on the eager path (``fmha_impl`` is None or not a
    ``DSv4DecodeFmhaImpl``). CUDA-graph capture has its own persistent
    metadata owned by ``DSv4DecodeFmhaImpl.metadata`` — the caller
    checks the fmha_impl type and picks.

    Returns ``None`` when the incoming batch is empty (B == 0) so the
    caller can short-circuit to an empty ``PyModelOutputs``.

    ``start_pos[r]`` is the absolute position of the new token's
    predecessor (per ``NormalModelInputGatherer.cc:255``). Clamped to
    ``max_seq_len - 1`` for warmup safety (probe at max_seq_len then
    decode).
    """
    from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
        build_decode_metadata,
    )

    seq_lens_d = attn.sequence_lengths
    if seq_lens_d.device.type == "cpu":
        seq_lens_d = seq_lens_d.to(device)
    start_pos = seq_lens_d.to(torch.int32)
    B = int(start_pos.shape[0])
    if B == 0:
        return None

    max_s = int(v4_args.max_seq_len)
    start_pos = torch.clamp(start_pos, min=0, max=max(0, max_s - 1))

    # Pull per-attn_type block_tables + entries_per_block for the paged
    # read/write path. Eager allocates fresh per step (no graph capture,
    # no forbid_realloc).
    paged_block_tables: Dict[int, Any] = {}
    paged_entries_per_block: Dict[int, int] = {}
    if paged_pool_specs:
        by_group = getattr(attn, "kv_cache_kernel_block_id_device_by_group", None)
        group_region_names = (
            getattr(kv_cache, "group_region_names", None)
            if kv_cache is not None
            else None
        )
        if by_group is not None and len(by_group) > 0 and group_region_names:
            # Walk the framework's group list: position IS the group id,
            # entry IS the attn_type. Keep the group only if the decode
            # impl asked for it via paged_pool_specs.
            for group_id, attn_type_enum in enumerate(group_region_names):
                if group_id >= len(by_group):
                    continue
                attn_type = int(attn_type_enum)
                spec = paged_pool_specs.get(attn_type)
                if spec is None:
                    continue
                entries_per_block, _ = spec
                group_block_table = by_group[group_id]
                if group_block_table is None or group_block_table.numel() == 0:
                    continue
                paged_block_tables[attn_type] = group_block_table
                paged_entries_per_block[attn_type] = entries_per_block

    return build_decode_metadata(
        start_pos=start_pos,
        q_len=1,
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
    input_ids_2d: torch.Tensor,  # [B, q_len]
    attn_metadata: Any,  # DSv4DecodeAttnMetadata
) -> torch.Tensor:
    """qwen3-style decode per-layer loop. Same body shape as the prefill
    helper (:func:`rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers`)
    but dispatches to ``layer.forward_decode`` (FlashMLA / FP8 path)
    and threads the pre-built decode metadata."""
    h = v4.embed(input_ids_2d)  # [B, q_len, dim]
    h = h.unsqueeze(2).repeat(1, 1, v4.hc_mult, 1)  # [B, q_len, hc, dim]
    for layer in v4.layers:
        h = layer.forward_decode(h, attn_metadata, input_ids_2d, kv_cache=kv_cache)
    h = v4._hc_head_reduce(h)
    h = v4.norm(h)
    return h


def forward_decode(
    v4: Any,
    kv_cache: Optional[Any],
    v4_args: Any,
    inputs: Any,  # PyModelInputs
    fmha_impl: Any = None,  # Optional[DSv4DecodeFmhaImpl]
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

    attn = inputs.attention_inputs
    param_dev = next(v4.parameters()).device

    input_ids = inputs.input_ids
    if input_ids.dim() == 0:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.device != param_dev:
        input_ids = input_ids.to(param_dev)

    if isinstance(fmha_impl, DSv4DecodeFmhaImpl):
        meta = fmha_impl.metadata
    else:
        paged_specs = build_paged_pool_specs(kv_cache, v4)
        meta = build_metadata_eager(
            v4_args,
            attn,
            param_dev,
            paged_specs,
            kv_cache=kv_cache,
        )
        if meta is None:
            # Empty batch (B == 0) — short-circuit with zero-row hidden.
            return PyModelOutputs(
                torch.zeros(
                    (0, v4_args.dim),
                    dtype=torch.bfloat16,
                    device=param_dev,
                )
            )

    B = meta.batch_size
    q_len = meta.q_len_per_req
    input_ids_2d = input_ids.view(B, q_len) if input_ids.dim() == 1 else input_ids
    h = forward_layers(v4, kv_cache, input_ids_2d, meta)  # [B, q_len, dim]
    hidden = h.reshape(B * q_len, v4_args.dim)  # packed [T_total, dim]
    return PyModelOutputs(hidden)
