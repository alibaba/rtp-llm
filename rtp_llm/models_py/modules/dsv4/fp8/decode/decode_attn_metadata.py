"""Per-step decode attention metadata for DeepSeek-V4.

For batched decode (B requests, q_len=1 each, may be 1+spec later), we
compute once per forward step:

  * ``slot_mapping_swa[T_total]`` int32 — for each generated token,
    the absolute SWA ring-buffer slot to write its K into. Slot is
    flat across batch (i.e. per-request offset baked in: see below).
  * ``slot_mapping_compressed[T_total]`` int32 — same idea for the
    compressed-K buffer (only used by layers with ``compress_ratio>0``;
    SWA-only layers ignore it).
  * ``topk_window_idxs[B, S, win]`` int32 — sliding-window topk
    (the "trivial" topk: read the last ``win`` tokens). Per-request
    cyclic ring offsets baked in.
  * ``topk_buffer_compressed[B, S, K_compressed]`` int32 — pre-allocated
    output buffer for the indexer. Filled in by IndexerDecodeV4Op
    layer-by-layer (each compress=4 layer overwrites; layer compute is
    sequential in Python, so a single buffer is fine).
  * ``compressed_lens[B]`` int32 — number of compressed-K entries each
    request has accumulated (used to mask invalid topk slots in CSA
    layers).

This builder is stateless and graph-friendly: every output tensor lives
on cuda, computed via vectorized torch ops, no Python-per-token loops.

KV layout (during Phase 1-3):
  Each request ``r`` owns a slot range in the per-layer register_buffer:
    SWA          : ``buffer[r, 0 : win]``         — ring of last ``win`` K
    Compressed-K : ``buffer[r, win : win + max_seq_len // ratio]``
  Per-step we write 1 token to:
    SWA          : ``buffer[r, start_pos[r] % win]``
    Compressed-K : iff ``(start_pos[r]+1) % ratio == 0``,
                   ``buffer[r, win + (start_pos[r]+1) // ratio - 1]``

The 2-D buffer is flattened to a 1-D slot-major view at op call time —
the slot indices we produce here are ``r * stride + offset_in_request``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class DSv4DecodeAttnMetadataFP8:
    """Metadata produced once per decode step, consumed by every layer.

    Fields are device-resident tensors unless noted. Built by
    :func:`build_decode_metadata_fp8` at the top of
    ``DeepSeekV4Model.forward`` (decode arm).

    Layer-type-aware fields are dicts keyed by ``compress_ratio`` so a
    layer just looks up its own slice — keeps the per-layer call site
    free of branching.
    """

    # Geometry (Python-scalar, used at op call time for shapes)
    batch_size: int
    q_len_per_req: int  # 1 in pure decode, 1+spec for MTP
    total_tokens: int  # batch_size * q_len_per_req
    window_size: int  # SWA size (V4: 128)
    head_dim: int  # V4: 512
    max_seq_len: int
    swa_buffer_t_dim: int  # = window_size
    compressed_buffer_t_dim_per_ratio: Dict[int, int]  # ratio -> max_seq_len // ratio

    # Per-request first-token absolute positions (device int32 tensor; B-shaped).
    # Normal decode has q_len=1. Target verify has q_len>1 and derives the full
    # request-major position stream as ``start_pos[:, None] + arange(q_len)``.
    start_pos: torch.Tensor  # [B] int32
    # Per-token absolute positions, flattened token-major over [B, q_len].
    # Generated internally and used as the single source of truth for RoPE,
    # slot mapping, topk windows, and cache lengths.
    position_ids: torch.Tensor  # [T_total] int32

    # Slot mappings — flat over [T_total] with per-request offset baked in.
    # SWA: applies to all layers.
    slot_mapping_swa: torch.Tensor  # [T_total] int32

    # Compressed-K slot mappings — keyed by ratio (4 for CSA, 128 for HCA).
    # ``slot_mapping_compressed[ratio][t] == -1`` if this token does NOT
    # land on a compression boundary for that ratio (so the write op
    # should skip it).
    slot_mapping_compressed: Dict[int, torch.Tensor]  # ratio -> [T_total] int32

    # Window topk indices (the "trivial" topk = last ``win`` tokens).
    # Already on the per-layer KV buffer's flat slot-space.
    # Shape [B, q_len, win]; -1 entries mark out-of-range (early decode).
    topk_window_idxs: torch.Tensor  # [B, S, win] int32

    # Pre-allocated indexer output buffer (only CSA / ratio=4 layers use it).
    # IndexerDecodeV4Op writes into this per-step. Shape [B, S, K_compressed].
    topk_buffer_compressed: torch.Tensor  # [B, S, K] int32

    # Number of valid compressed-K entries each request currently has
    # in the compressed pool, per ratio. Used to mask topk after gather.
    # ``compressed_lens[ratio][r] = (start_pos[r] + 1) // ratio``  (after
    # this step's writes are applied).
    compressed_lens: Dict[int, torch.Tensor]  # ratio -> [B] int32

    # Concatenated topk indices per layer-type. For SWA-only layers,
    # equals ``topk_window_idxs``. For compressed layers (CSA/HCA),
    # equals ``cat([topk_window_idxs, topk_compressed_for_ratio], dim=-1)``.
    # IndexerDecodeV4Op fills the compressed half in-place.
    topk_total_by_ratio: Dict[int, torch.Tensor]  # ratio -> [B, S, win+K] int32

    # Per-request running compressed-K offset (= win, since the
    # compressed pool starts right after the SWA ring buffer). Same for
    # every request in current scheme.
    compressed_offset: int = 128  # = window_size by default

    # ---- Cuda graph: reserved for Phase 3 (forbid_realloc, fixed addr) ----
    is_cuda_graph: bool = False

    # ------------------------------------------------------------------
    # Paged-decode metadata (paged BlockPool read/write).
    # All optional — when empty, decode falls back to the legacy
    # register_buffer path. Populated by the model + impl once the
    # framework block tables are wired.
    # ------------------------------------------------------------------

    # Per-attn_type framework block_table: [max_B, max_blocks_per_req] int32.
    # Source: ``attn_inputs.kv_cache_kernel_block_id_device_by_group[gid]``.
    # Keys are attn_type ids (1=CSA_KV..7=SWA_KV) from
    # :mod:`rtp_llm.models_py.modules.dsv4.attn_type`; only pools that the
    # model actually uses are present.
    pool_block_tables: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-attn_type new-token write slot mapping: [max_T_total] int64.
    # ``slot[t] = block_table[req(t), abs_pos(t)//E] * E + abs_pos(t)%E``
    # where ``E`` is the pool's entries_per_block; ``-1`` = skip
    # (boundary-only writers like CSA-K / HCA-K / INDEXER-K).
    pool_write_slot_mappings: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-step absolute positions for the SWA window — left-aligned,
    # ``-1`` padded for pre-sequence-start entries. Shape ``[B, q_len, win]``
    # int32. UNLIKE ``topk_window_idxs`` (which is request-local ring
    # slots in [0, win)), this carries the actual absolute token
    # positions in [0, max_seq_len) so it can be turned into global pool
    # slot ids via ``translate_local_to_global_slots``. Optional —
    # populated only when paged-decode read is enabled.
    swa_abs_idx: torch.Tensor = field(default=None)  # type: ignore[assignment]

    # Phase F: ``layer_pool_descs`` deleted — Attention resolves pool
    # views via ``self._kv_cache.get_layer_cache(layer_id, attn_type)``
    # at call time, no per-layer descriptor cache needed.

    # FlashMLA's ``cache_seqlens`` argument = start_pos + 1 (number of
    # valid kv tokens for this step). The decode FP8 attention op is
    # called 43× per step (once per layer) with the same value, so we
    # compute the int32 tensor once here and reuse across layers instead
    # of re-running ``(start_pos[:bsz] + 1).to(torch.int32)`` per layer.
    # Pre-allocated at max_batch by ``allocate_decode_metadata_fp8`` and
    # refilled per step by ``update_decode_metadata_in_place_fp8``.
    cache_seqlens_i32: torch.Tensor = field(default=None)  # type: ignore[assignment]

    # Iter3.3: shared-across-layers request-id mapping ``[T]`` int32 = the
    # ``arange(bsz)`` (for q_len=1) passed to every ``translate_local_to_global_slots``
    # call. 43 layers × same arange = one cached tensor.
    req_id_per_token: Optional[torch.Tensor] = None

    # Iter3.3: cached ``swa_global_slots`` = translate_local_to_global_slots(
    # req_id, swa_pool_bt, swa_abs_idx, swa_eb). Shape [T, win] int32.
    # Shared across all 43 layers (SWA block table + swa_abs_idx are
    # identical). When non-None the per-layer body skips the triton
    # translator call and reads this directly.
    swa_global_slots: Optional[torch.Tensor] = None

    # Iter3.3: cached ``hca_cmp_global_slots`` = translate_local_to_global_slots(
    # req_id, hca_pool_bt, hca_cmp_local, hca_eb). Shape [T, K_h] int32.
    # Shared across all HCA layers in a step (HCA block table + dense cmp
    # idx are identical across HCA layers). Only populated when an HCA
    # pool is bound.
    hca_cmp_global_slots: Optional[torch.Tensor] = None

    # FlashMLA ``sched_meta`` cache — per-(batch_size, extra_attn_type).
    # Mirrors vLLM's ``swa_metadata.tile_sched_{swaonly,c4a,c128a}`` pattern:
    # the planner is called once per (mode, B) combination and the returned
    # ``sched_meta`` is reused across all layers of the same type in a decode
    # step (43× in DSv4). Lives on the metadata so:
    #   * Eager path: rebuilt each step with the metadata instance, freed
    #     when metadata GCs — no cross-step accumulation.
    #   * CUDA graph path: populated lazily during capture (tensors alloc'd
    #     via the graph-aware allocator), stays alive for the impl lifetime
    #     so replay at the same captured B reuses the same Python tensors.
    # CSA and HCA layers MUST NOT share a sched_meta: the first
    # ``flash_mla_with_kvcache`` call bakes ``extra_k_cache.shape[1]``
    # (page_block_size of the compressed pool) into the sched_meta, and the
    # wheel enforces ``sched_meta.config.extra_page_block_size ==
    # extra_k_cache.shape[1]`` on every subsequent call. CSA_KV and HCA_KV
    # pools have different page_block_size → separate cache keys.
    # ``extra_at`` is the ``attn_type`` int for the compressed pool
    # (``CSA_KV`` / ``HCA_KV``) or ``None`` for single-pool SWA-only.
    sched_meta_cache: Dict[Tuple[int, Optional[int]], Any] = field(default_factory=dict)


def get_or_build_sched_meta(
    metadata: "DSv4DecodeAttnMetadataFP8",
    *,
    batch_size: int,
    q_len: int,
    num_heads: int,
    topk: int,
    extra_attn_type: Optional[int] = None,
) -> Any:
    """Lazy-build + cache FlashMLA ``sched_meta`` on the metadata object.

    Replaces the ``SparseAttnV4DecodeFp8Op._sched_meta_cache`` anti-pattern
    (per-op instance cache accumulating across steps). Matches vLLM's
    ``DeepseekSparseSWAMetadataBuilder.build_tile_scheduler`` design —
    sched_meta lives on the per-step metadata object.

    Cache key is ``(batch_size, extra_attn_type)``:
      * The FlashMLA wheel bakes ``config.b`` (batch size) and
        ``config.extra_page_block_size`` (extra_k_cache page size) into the
        sched_meta on the first ``flash_mla_with_kvcache`` call and asserts
        equality on every subsequent call.
      * SWA-only (single pool) → ``extra_attn_type=None``.
      * CSA layers (dual pool, CSA_KV as extra) → ``extra_attn_type=CSA_KV``.
      * HCA layers (dual pool, HCA_KV as extra) → ``extra_attn_type=HCA_KV``.
      * CSA_KV and HCA_KV pools have different ``page_block_size``, so
        they MUST use separate sched_meta instances.

    Within one (B, extra_attn_type) bucket ``q_len / num_heads / topk`` are
    process-constants.
    """
    from flash_mla import get_mla_metadata  # type: ignore[import-not-found]

    key = (batch_size, extra_attn_type)
    sched_meta = metadata.sched_meta_cache.get(key)
    if sched_meta is None:
        sched_meta, _ = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=batch_size * q_len * num_heads,
            topk=topk,
            num_heads_q=num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        metadata.sched_meta_cache[key] = sched_meta
    return sched_meta


def _build_swa_slot_mapping(
    start_pos: torch.Tensor, q_len: int, window_size: int, swa_buffer_stride: int
) -> torch.Tensor:
    """For each (req r, q-index s) compute the flat slot in the SWA
    register_buffer. SWA buffer is per-request ``[window_size, head_dim]``
    flattened: slot = r * swa_buffer_stride + ((start_pos[r] + s) % window_size).

    Args:
        start_pos: [B] int32, host or device.
        q_len: scalar (1 for pure decode, 1+spec for MTP).
        window_size: SWA window (128).
        swa_buffer_stride: row stride of flattened buffer = window_size.

    Returns:
        [B * q_len] int32 device tensor.
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    # offsets [B, q_len] = start_pos[:, None] + arange(q_len)[None, :]
    s_offsets = start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)
    # ring offset within window
    in_ring = s_offsets % window_size
    # flat slot
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1) * swa_buffer_stride
    )
    return (req_base + in_ring).reshape(-1)


def _build_compressed_slot_mapping(
    start_pos: torch.Tensor, q_len: int, ratio: int, compressed_buffer_stride: int
) -> torch.Tensor:
    """For each (req r, q-index s) where the absolute position
    ``start_pos[r] + s`` is on a compression boundary
    (i.e. ``(start_pos[r] + s + 1) % ratio == 0``), compute the slot
    in the compressed-K buffer; else -1.

    Compressed slot index within request: ``(start_pos[r] + s + 1) // ratio - 1``.

    Args:
        start_pos: [B] int32.
        q_len: scalar.
        ratio: 4 (CSA) or 128 (HCA).
        compressed_buffer_stride: row stride of flattened buffer
            = ``max_seq_len // ratio``.

    Returns:
        [B * q_len] int32 device tensor (-1 means skip).
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    abs_pos_plus_1 = (
        start_pos.view(B, 1)
        + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
        + 1
    )  # [B, q_len]
    on_boundary = (abs_pos_plus_1 % ratio) == 0  # [B, q_len] bool
    in_req = abs_pos_plus_1 // ratio - 1  # [B, q_len] int32
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        * compressed_buffer_stride
    )
    flat = (req_base + in_req).reshape(-1)
    mask = on_boundary.reshape(-1)
    return torch.where(mask, flat, torch.full_like(flat, -1))


def _build_window_topk_idxs(
    start_pos: torch.Tensor, q_len: int, window_size: int
) -> torch.Tensor:
    """For each (req, q-index) build the int32 [win] topk pointing at
    the request-local ring positions to read.

    For abs_pos = start_pos[r] + s:
      * If abs_pos == 0: read [0, -1, -1, ...] (only token 0 valid).
      * If abs_pos < window_size: ascending from 0 to abs_pos, padding
        the rest with -1.
      * Else: cyclic — start at (abs_pos+1) % win, wrap once around.

    Returns the indices in the SAME flat slot-space the kernel reads,
    so we use *request-local* ring positions [0, win) — the kernel
    indexes into the per-request kv slice.

    Args:
        start_pos: [B] int32.
        q_len: scalar.
        window_size: SWA window.

    Returns:
        [B, q_len, window_size] int32.
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    # abs_pos[r, s] = start_pos[r] + s
    abs_pos = start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(
        1, q_len
    )  # [B, q_len]
    # LEFT-aligned: valid entries first, -1 padding at the tail.
    # Full ring (abs_pos >= win-1): cyclic read oldest→newest starting at (sp+1)%win.
    # Partial ring (abs_pos < win-1): ascending [0..abs_pos], then -1 padding.
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    abs_pos_b = abs_pos.unsqueeze(-1)  # [B, q_len, 1]
    sp = (abs_pos % window_size).unsqueeze(-1)  # [B, q_len, 1] ring write position
    # Full ring: k-th entry = (sp + 1 + k) % win  (oldest slot first)
    ring_full_idx = (sp + 1 + k_range) % window_size
    # Partial ring: k-th entry = k if k <= abs_pos, else -1
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    is_full = abs_pos_b >= (window_size - 1)
    return torch.where(is_full, ring_full_idx, partial_idx)


def _build_position_ids_2d(
    start_pos: torch.Tensor,
    q_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return absolute decode positions as ``[B, q_len]`` int32.

    The public metadata contract stores positions flat ``[T]``. Internally
    the slot/topk math needs a temporary request-major view. The framework
    position_ids field is intentionally not consumed here; DSv4 decode builds
    its own contiguous per-request position stream from the first token
    position and q_len.
    """
    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    B = int(start_pos.shape[0])
    return start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)


def _build_start_pos_from_attention_inputs(
    attention_inputs: Any,
    device: torch.device,
    max_seq_len: int,
) -> torch.Tensor:
    """Derive DSv4 decode first-token positions from framework attention inputs."""
    if isinstance(attention_inputs, torch.Tensor):
        start_pos = attention_inputs
    else:
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        if is_target_verify:
            start_pos = attention_inputs.prefix_lengths
        else:
            start_pos = attention_inputs.sequence_lengths

    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    return torch.clamp(start_pos, min=0, max=max(0, int(max_seq_len) - 1))


def _build_swa_slot_mapping_from_positions(
    position_ids_2d: torch.Tensor, window_size: int, swa_buffer_stride: int
) -> torch.Tensor:
    B = int(position_ids_2d.shape[0])
    device = position_ids_2d.device
    in_ring = position_ids_2d % window_size
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        * swa_buffer_stride
    )
    return (req_base + in_ring).reshape(-1)


def _build_compressed_slot_mapping_from_positions(
    position_ids_2d: torch.Tensor, ratio: int, compressed_buffer_stride: int
) -> torch.Tensor:
    B = int(position_ids_2d.shape[0])
    device = position_ids_2d.device
    abs_pos_plus_1 = position_ids_2d + 1
    on_boundary = (abs_pos_plus_1 % ratio) == 0
    in_req = abs_pos_plus_1 // ratio - 1
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        * compressed_buffer_stride
    )
    flat = (req_base + in_req).reshape(-1)
    mask = on_boundary.reshape(-1)
    return torch.where(mask, flat, torch.full_like(flat, -1))


def _build_window_topk_idxs_from_positions(
    position_ids_2d: torch.Tensor, window_size: int
) -> torch.Tensor:
    device = position_ids_2d.device
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    abs_pos_b = position_ids_2d.unsqueeze(-1)
    sp = (position_ids_2d % window_size).unsqueeze(-1)
    ring_full_idx = (sp + 1 + k_range) % window_size
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    is_full = abs_pos_b >= (window_size - 1)
    return torch.where(is_full, ring_full_idx, partial_idx)


def allocate_decode_metadata_fp8(
    max_batch_size: int,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    paged_pool_specs: Optional[Dict[int, Tuple[int, int]]] = None,
) -> "DSv4DecodeAttnMetadataFP8":
    """Pre-allocate a ``DSv4DecodeAttnMetadataFP8`` sized for ``max_batch_size``.

    Used by Phase 3 CUDA-graph capture: the impl owns a single metadata
    instance whose tensors live at fixed addresses; ``update_in_place``
    rewrites the contents per step. The captured graph then reads from
    these stable addresses on every replay.

    Tensor sizes are MAX, so ``update_in_place(start_pos)`` with
    ``start_pos.shape[0] == bs`` only writes the ``[:bs, ...]`` prefix.
    The captured graph at fixed BS slices these tensors at construction
    time (via ``DSv4DecodeFmhaImplFP8``); since each captured graph is
    per-BS, the slice offset is constant and known at capture.
    """
    B = max_batch_size
    T_total = B * q_len
    swa_buffer_stride = window_size

    start_pos = torch.zeros(B, dtype=torch.int32, device=device)
    position_ids = torch.zeros(T_total, dtype=torch.int32, device=device)
    slot_swa = torch.full((T_total,), -1, dtype=torch.int32, device=device)
    topk_window = torch.full(
        (B, q_len, window_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_buffer_compressed = torch.full(
        (B, q_len, index_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )

    unique_ratios: List[int] = sorted({r for r in compress_ratios if r > 1})
    slot_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_buffer_stride_per_ratio: Dict[int, int] = {}
    topk_total_by_ratio: Dict[int, torch.Tensor] = {}
    for r in unique_ratios:
        stride = max_seq_len // r
        compressed_buffer_stride_per_ratio[r] = stride
        slot_compressed[r] = torch.full(
            (T_total,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        compressed_lens[r] = torch.zeros(B, dtype=torch.int32, device=device)
        topk_total_by_ratio[r] = torch.full(
            (B, q_len, window_size + index_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )

    # Phase 2: paged pool block_table + write slot mapping buffers.
    # paged_pool_specs maps attn_type → (entries_per_block, max_blocks_per_req).
    # We never reallocate after construction, so the captured graph reads
    # from these stable addresses. When a pool is absent on a layer, the
    # corresponding entries are simply unused.
    pool_block_tables: Dict[int, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[int, torch.Tensor] = {}
    if paged_pool_specs:
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        for attn_type, (_, max_blocks) in paged_pool_specs.items():
            pool_block_tables[attn_type] = torch.zeros(
                B, max_blocks, dtype=torch.int32, device=device
            )
            # SWA_KV is the always-write path (mask_negative=False downstream),
            # so it must hold a *valid* slot at all times — including during
            # framework warmup where update_decode_metadata_in_place_fp8's paged
            # branch is skipped (no block_tables yet). Slot 0 is always a
            # valid pool index; warmup writes overlap there harmlessly.
            # Compressed pools (CSA/HCA/INDEXER) use mask_negative=True and
            # require -1 sentinel for non-boundary tokens.
            sentinel = 0 if attn_type == SWA_KV else -1
            pool_write_slot_mappings[attn_type] = torch.full(
                (T_total,), sentinel, dtype=torch.int64, device=device
            )

    # Pre-allocate swa_abs_idx[B, q_len, win] int32 (always — Phase 2B-2a
    # paged read may consume it; cost is trivial compared to the rest).
    swa_abs_idx = torch.full(
        (B, q_len, window_size),
        -1,
        dtype=torch.int32,
        device=device,
    )

    # Iter3.2: precomputed cache_seqlens = start_pos + 1 int32, reused
    # across all 43 decode layers. See field doc on DSv4DecodeAttnMetadataFP8.
    cache_seqlens_i32_alloc = torch.zeros(B, dtype=torch.int32, device=device)

    # Iter3.3: pre-allocated buffers for the shared-across-layers translate
    # outputs. Under CUDA graph capture the graph bakes in the buffer
    # address, so refill MUST .copy_() into the existing storage instead of
    # reassigning ``meta.swa_global_slots = translate(...)`` — that would
    # leave captured reads pointing at stale memory. ``req_id_per_token``
    # is deterministic (``arange(bs)`` for q_len=1) so we fill it once here.
    req_id_per_token_alloc = torch.arange(B, dtype=torch.int32, device=device)
    if q_len != 1:
        req_id_per_token_alloc = req_id_per_token_alloc.repeat_interleave(
            q_len
        ).contiguous()
    swa_global_slots_alloc = torch.full(
        (B * q_len, window_size), -1, dtype=torch.int32, device=device
    )
    # HCA dense-idx width = max_seq_len / 128 = index_topk by construction
    # (see allocate_decode_metadata_fp8: topk_total shape[-1] = win + index_topk).
    hca_cmp_global_slots_alloc = torch.full(
        (B * q_len, index_topk), -1, dtype=torch.int32, device=device
    )

    return DSv4DecodeAttnMetadataFP8(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
        position_ids=position_ids,
        slot_mapping_swa=slot_swa,
        slot_mapping_compressed=slot_compressed,
        topk_window_idxs=topk_window,
        topk_buffer_compressed=topk_buffer_compressed,
        compressed_lens=compressed_lens,
        topk_total_by_ratio=topk_total_by_ratio,
        compressed_offset=window_size,
        is_cuda_graph=True,
        pool_block_tables=pool_block_tables,
        pool_write_slot_mappings=pool_write_slot_mappings,
        swa_abs_idx=swa_abs_idx,
        cache_seqlens_i32=cache_seqlens_i32_alloc,
        req_id_per_token=req_id_per_token_alloc,
        swa_global_slots=swa_global_slots_alloc,
        hca_cmp_global_slots=hca_cmp_global_slots_alloc,
    )


def update_decode_metadata_in_place_fp8(
    meta: "DSv4DecodeAttnMetadataFP8",
    attention_inputs: Any,
    forbid_realloc: bool = False,
    paged_block_tables: Optional[Dict[int, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[int, int]] = None,
) -> None:
    """Recompute every metadata buffer IN PLACE for new attention inputs.

    Contract:
      * Every output tensor reuses its prior storage (``data_ptr()``
        unchanged); ``forbid_realloc=True`` makes any accidental realloc
        an immediate error rather than silent-correctness-bug.
      * Writes the prefix ``[:bs]`` (or ``[:bs * q_len]`` for slot
        mappings); leaves the tail at sentinel (-1) from
        ``allocate_decode_metadata_fp8``. The captured CUDA graph at that
        BS only reads the prefix, so the tail never matters.

    Mirrors :func:`build_decode_metadata_fp8` arithmetic. Kept as a separate
    function (not method) so it's callable from ``DSv4DecodeFmhaImplFP8.prepare``
    and from unit tests without instantiating the impl.

    Args:
        meta: Pre-allocated metadata (from ``allocate_decode_metadata_fp8``).
        attention_inputs: Framework attention inputs. Normal decode derives
            the first-token position from ``sequence_lengths``; target verify
            derives it from ``prefix_lengths`` because C++ clears
            ``sequence_lengths`` for that path. A tensor is accepted for
            focused metadata tests.
        forbid_realloc: If True, asserts every write reuses the existing
            tensor storage (sanity check for the captured-graph path).
    """
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = meta.start_pos.device

    start_pos = _build_start_pos_from_attention_inputs(
        attention_inputs,
        device,
        meta.max_seq_len,
    )
    bs = int(start_pos.shape[0])
    position_ids_2d = _build_position_ids_2d(start_pos, q_len, device)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()

    # snapshot pointers for the realloc-forbidden mode
    if forbid_realloc:
        ptr_snap = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        for r, t in meta.slot_mapping_compressed.items():
            ptr_snap[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            ptr_snap[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            ptr_snap[f"topk_total_by_ratio[{r}]"] = t.data_ptr()

    # start_pos
    meta.start_pos[:bs].copy_(start_pos)
    meta.position_ids[: bs * q_len].copy_(position_ids_flat)
    # Iter3.2: refresh cache_seqlens = last decoded position + 1 (shared across all 43
    # decode-layer calls this step). Cheap scalar add + copy, done once
    # here instead of 43× in ``_forward_decode_body``.
    if meta.cache_seqlens_i32 is not None:
        meta.cache_seqlens_i32[:bs].copy_(position_ids_2d[:, -1] + 1)

    # SWA slot mapping prefix [:bs * q_len]
    swa_buffer_stride = window_size
    s_offsets = position_ids_2d
    swa_slots = _build_swa_slot_mapping_from_positions(
        position_ids_2d, window_size, swa_buffer_stride
    )
    meta.slot_mapping_swa[: bs * q_len].copy_(swa_slots)

    # Window topk indices [:bs, :, :] — left-aligned (mirrors _build_window_topk_idxs)
    abs_pos = s_offsets  # [bs, q_len]
    window_idxs = _build_window_topk_idxs_from_positions(position_ids_2d, window_size)
    meta.topk_window_idxs[:bs].copy_(window_idxs)

    # Indexer output buffer reset to -1 prefix [:bs]
    meta.topk_buffer_compressed[:bs].fill_(-1)

    # Per-ratio compressed slot mappings + lens + topk_total
    for r, slot_t in meta.slot_mapping_compressed.items():
        stride = meta.compressed_buffer_t_dim_per_ratio[r]
        abs_pos_plus_1 = position_ids_2d + 1
        on_boundary = (abs_pos_plus_1 % r) == 0
        in_req = abs_pos_plus_1 // r - 1
        cmp_req_base = (
            torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1) * stride
        )
        flat = (cmp_req_base + in_req).reshape(-1)
        mask = on_boundary.reshape(-1)
        cmp_slots = torch.where(mask, flat, torch.full_like(flat, -1))
        slot_t[: bs * q_len].copy_(cmp_slots)

        # compressed_lens
        meta.compressed_lens[r][:bs].copy_(
            ((position_ids_2d[:, -1] + 1) // r).to(torch.int32)
        )

        # topk_total_by_ratio: refill window half, refill HCA dense half
        total = meta.topk_total_by_ratio[r]
        total[:bs, :, :window_size].copy_(window_idxs)
        if r != 4:
            K_dense = total.shape[-1] - window_size
            dense_idxs = (
                torch.arange(
                    K_dense,
                    device=device,
                    dtype=torch.int32,
                )
                .view(1, 1, K_dense)
                .expand(bs, q_len, K_dense)
            )
            cmp_lens_per_token = ((position_ids_2d + 1) // r).view(bs, q_len, 1)
            valid_h = dense_idxs < cmp_lens_per_token
            total[:bs, :, window_size:].copy_(
                torch.where(valid_h, dense_idxs, torch.full_like(dense_idxs, -1))
            )
        else:
            # CSA: indexer fills the compressed half per-call. Reset to -1.
            total[:bs, :, window_size:].fill_(-1)

    # ------------------------------------------------------------------
    # Phase 2: paged write slot mappings (per attn_type).
    # SWA-K writes EVERY decode token; CSA-K / HCA-K / INDEXER-K write
    # ONLY on their respective compression boundaries (sentinel ``-1``
    # otherwise — the write op honors that via ``mask_negative=True``).
    # ------------------------------------------------------------------
    if paged_block_tables is not None and paged_pool_entries_per_block is not None:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
            compute_kv_pool_slot_mapping,
        )

        # Snapshot block_table content into the metadata's stable buffer
        # (forbid_realloc-friendly: just `.copy_` the prefix). Skip pools
        # without a metadata buffer (e.g. SWA-only layers don't carry CSA).
        for at, src_bt in paged_block_tables.items():
            dst_bt = meta.pool_block_tables.get(at)
            if dst_bt is None:
                continue
            n_rows = min(src_bt.shape[0], dst_bt.shape[0])
            n_cols = min(src_bt.shape[1], dst_bt.shape[1])
            # Zero stale rows beyond current bs to avoid carrying old block
            # ids (defensive — graph reads only [:bs] anyway).
            dst_bt[bs:].zero_()
            dst_bt[:n_rows, :n_cols].copy_(src_bt[:n_rows, :n_cols])
            if n_cols < dst_bt.shape[1]:
                dst_bt[:n_rows, n_cols:].zero_()

        # SWA: every token writes; abs_pos = start_pos + s.
        if SWA_KV in meta.pool_block_tables:
            slot = meta.pool_write_slot_mappings[SWA_KV]
            E = paged_pool_entries_per_block.get(SWA_KV, window_size)
            abs_pos_swa = position_ids_flat
            mapped = compute_kv_pool_slot_mapping(
                meta.pool_block_tables[SWA_KV][:bs],
                abs_pos_swa,
                E,
            )
            slot[: bs * q_len].copy_(mapped)

        # Compressed pools (CSA / HCA / INDEXER): write only on boundary;
        # abs_pos here is the COMPRESSED entry index for this token, with
        # ``-1`` sentinel for non-boundary tokens. Indexer shares the
        # ratio=4 boundary with CSA.
        for ratio_key, attn_type_writers in (
            (4, [CSA_KV, INDEXER_KV]),
            (128, [HCA_KV]),
        ):
            if ratio_key not in meta.slot_mapping_compressed:
                continue
            # Per-request compressed entry index for THIS step's tokens,
            # with ``-1`` sentinel for non-boundary tokens. The legacy
            # ``meta.slot_mapping_compressed[ratio]`` is in register_buffer
            # coordinates (req_idx*stride+offset); for paged we need the
            # plain per-request entry index.
            abs_pos_plus_1 = position_ids_2d + 1
            on_boundary = (abs_pos_plus_1 % ratio_key) == 0
            cmp_idx = abs_pos_plus_1 // ratio_key - 1
            cmp_idx_with_skip = torch.where(
                on_boundary,
                cmp_idx,
                torch.full_like(cmp_idx, -1),
            ).reshape(-1)
            for at in attn_type_writers:
                if at not in meta.pool_block_tables:
                    continue
                E = paged_pool_entries_per_block.get(at, 1)
                mapped = compute_kv_pool_slot_mapping(
                    meta.pool_block_tables[at][:bs],
                    cmp_idx_with_skip,
                    E,
                )
                meta.pool_write_slot_mappings[at][: bs * q_len].copy_(mapped)

    # Phase 2B-2a: SWA absolute-position window (paged read). Left-aligned,
    # ``-1`` padded for entries before sequence start. Same shape as
    # ``topk_window_idxs`` but holds abs positions, not ring slots.
    if meta.swa_abs_idx is not None:
        win_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
            1,
            1,
            window_size,
        )
        win_start = (abs_pos.unsqueeze(-1) - window_size + 1).clamp(
            min=0
        )  # [bs,q_len,1]
        candidate = win_start + win_range  # [bs, q_len, win]
        valid_pos = candidate <= abs_pos.unsqueeze(-1)
        meta.swa_abs_idx[:bs].copy_(
            torch.where(valid_pos, candidate, torch.full_like(candidate, -1))
        )

    # Iter3.3: precompute translate_swa / translate_hca once per step.
    # Shared across all 43 attention layers. Graph-safe: ``.copy_()`` into
    # pre-allocated buffers on meta (allocated at max_bs); never reassigns
    # the attribute, so graph replays read from fixed addresses.
    # ``req_id_per_token`` is deterministic (``arange(B)``) so it was filled
    # at allocate time and stays stable.
    if paged_block_tables is not None and paged_pool_entries_per_block is not None:
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )

        T = bs * q_len
        req_id_bs = (
            meta.req_id_per_token[:T] if meta.req_id_per_token is not None else None
        )

        if (
            req_id_bs is not None
            and meta.swa_global_slots is not None
            and SWA_KV in meta.pool_block_tables
            and meta.swa_abs_idx is not None
        ):
            swa_eb = paged_pool_entries_per_block.get(SWA_KV, window_size)
            swa_local = meta.swa_abs_idx[:bs].reshape(T, window_size)
            swa_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[SWA_KV][:bs],
                swa_local,
                swa_eb,
            )
            meta.swa_global_slots[:T].copy_(swa_global_new)

        # HCA layers all share the dense-idx-masked cmp_local_raw (already
        # materialised as topk_total[r=128][:, :, win:]); translate it once.
        if (
            req_id_bs is not None
            and meta.hca_cmp_global_slots is not None
            and HCA_KV in meta.pool_block_tables
            and 128 in meta.topk_total_by_ratio
        ):
            hca_eb = paged_pool_entries_per_block.get(HCA_KV, 1)
            hca_tt = meta.topk_total_by_ratio[128]
            K_h = hca_tt.shape[-1] - window_size
            hca_cmp_local = hca_tt[:bs, :, window_size:].reshape(T, K_h).contiguous()
            hca_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[HCA_KV][:bs],
                hca_cmp_local,
                hca_eb,
            )
            meta.hca_cmp_global_slots[:T].copy_(hca_global_new)

    # Update Python-scalar geometry (cheap — these are not captured into the graph)
    meta.batch_size = bs
    meta.total_tokens = bs * q_len

    if forbid_realloc:
        # Verify every storage pointer is unchanged.
        cur = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        for r, t in meta.slot_mapping_compressed.items():
            cur[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            cur[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            cur[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
        for k, p_before in ptr_snap.items():
            assert cur[k] == p_before, (
                f"update_decode_metadata_in_place_fp8(forbid_realloc=True) "
                f"reallocated buffer '{k}': before={hex(p_before)} after={hex(cur[k])}"
            )


def build_decode_metadata_fp8(
    start_pos: torch.Tensor,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,  # noqa: ARG001 — reserved for future
    paged_block_tables: Optional[Dict[int, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[int, int]] = None,
) -> DSv4DecodeAttnMetadataFP8:
    """Top-level builder. Call ONCE per decode forward step.

    Args:
        start_pos: [B] int32 — current absolute position per request
            (i.e. number of tokens already in cache before this step).
        q_len: tokens per request this step (1 for pure decode).
        window_size: SWA size (V4: 128).
        head_dim: 512 for V4 attn KV; passed through for downstream
            shape-checking.
        max_seq_len: configured max — sets the compressed-K buffer
            stride for slot computation.
        compress_ratios: per-layer ratios; we deduplicate to figure out
            which compressed pools we need slot mappings for.
        index_topk: K (V4: 512) — for pre-allocating the indexer
            output buffer.
        device: cuda device.

    Returns:
        Fully-populated DSv4DecodeAttnMetadataFP8.
    """
    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)

    B = start_pos.shape[0]
    T_total = B * q_len
    position_ids_2d = _build_position_ids_2d(start_pos, q_len, device)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()

    # SWA slot mapping (always needed).
    swa_buffer_stride = window_size
    slot_swa = _build_swa_slot_mapping_from_positions(
        position_ids_2d, window_size, swa_buffer_stride
    )

    # Window topk — request-local ring positions.
    topk_window = _build_window_topk_idxs_from_positions(
        position_ids_2d, window_size
    )

    # Per-ratio compressed slot mappings.
    unique_ratios: List[int] = sorted({r for r in compress_ratios if r > 1})
    slot_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_buffer_stride_per_ratio: Dict[int, int] = {}
    for r in unique_ratios:
        stride = max_seq_len // r
        compressed_buffer_stride_per_ratio[r] = stride
        slot_compressed[r] = _build_compressed_slot_mapping_from_positions(
            position_ids_2d, r, stride
        )
        # After-this-step length: floor((start_pos + q_len) / r)
        # (each request now has this many compressed entries).
        compressed_lens[r] = ((position_ids_2d[:, -1] + 1) // r).to(torch.int32)

    # Indexer output buffer — pre-allocated; IndexerDecodeV4Op fills it.
    # Indexer is only present in compress_ratio==4 layers, so K=index_topk.
    topk_buffer_compressed = torch.full(
        (B, q_len, index_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )

    # Concatenated topk per ratio.
    # The compressed-half indices live in the compressed pool, which the
    # caller stitches behind the SWA pool: compressed_offset = window_size.
    # Per-layer call site uses these indices directly against the per-request
    # KV view ``buffer[r, : ]``.  IndexerDecodeV4Op writes into the slice
    # ``topk_total_by_ratio[ratio][:, :, window_size:]`` for ratio==4.
    topk_total_by_ratio: Dict[int, torch.Tensor] = {}
    for r in unique_ratios:
        total = torch.full(
            (B, q_len, window_size + index_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Window half: pre-fill from window topk (request-local ring slots).
        total[:, :, :window_size] = topk_window
        # The compressed half is filled per-call by indexer (CSA r=4) or by
        # _build_dense_compressed_idxs (HCA r=128). For HCA r=128 we fill
        # here directly since it's deterministic (read all valid compressed entries).
        if r != 4:
            # Dense compressed read: indices [0..compressed_lens[r]) per request,
            # right-padded with -1. Compressed half range [win, win+K_dense) where
            # K_dense=index_topk. K_dense should be >= max compressed_lens for HCA;
            # if not, this is data-loss (no-op for V4 since HCA layers only carry
            # max_seq_len/128 entries, which is way less than index_topk=512 at
            # max_seq_len=64K).
            K_dense = index_topk
            dense_idxs = (
                torch.arange(K_dense, device=device, dtype=torch.int32)
                .view(
                    1,
                    1,
                    K_dense,
                )
                .expand(B, q_len, K_dense)
            )
            cmp_lens_per_token = ((position_ids_2d + 1) // r).view(B, q_len, 1)
            valid = dense_idxs < cmp_lens_per_token
            # Indices are *request-local*: position within the compressed pool
            # (i.e. 0..max_seq_len/ratio).
            total[:, :, window_size:] = torch.where(
                valid,
                dense_idxs,
                torch.full_like(dense_idxs, -1),
            )
        # else (r==4): compressed half stays -1 sentinel until IndexerDecodeV4Op fills it.
        topk_total_by_ratio[r] = total

    # Phase 2 paged eager-path metadata (allocated fresh per step — eager
    # path doesn't share buffers across steps, unlike DSv4DecodeFmhaImplFP8).
    pool_block_tables: Dict[int, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[int, torch.Tensor] = {}
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
            compute_kv_pool_slot_mapping,
        )

        # Snapshot block tables (clone so downstream writes can't surprise
        # the framework's tensor).
        for at, bt in paged_block_tables.items():
            pool_block_tables[at] = bt[:B].contiguous().clone()

        if SWA_KV in pool_block_tables:
            E = paged_pool_entries_per_block.get(SWA_KV, window_size)
            pool_write_slot_mappings[SWA_KV] = compute_kv_pool_slot_mapping(
                pool_block_tables[SWA_KV],
                position_ids_flat,
                E,
            )

        for ratio_key, attn_type_writers in (
            (4, [CSA_KV, INDEXER_KV]),
            (128, [HCA_KV]),
        ):
            if ratio_key not in compressed_lens:
                continue
            abs_pos_plus_1 = position_ids_2d + 1
            on_boundary = (abs_pos_plus_1 % ratio_key) == 0
            cmp_idx = abs_pos_plus_1 // ratio_key - 1
            cmp_idx_with_skip = torch.where(
                on_boundary,
                cmp_idx,
                torch.full_like(cmp_idx, -1),
            ).reshape(-1)
            for at in attn_type_writers:
                if at not in pool_block_tables:
                    continue
                E = paged_pool_entries_per_block.get(at, 1)
                pool_write_slot_mappings[at] = compute_kv_pool_slot_mapping(
                    pool_block_tables[at],
                    cmp_idx_with_skip,
                    E,
                )

    # Phase 2B-2a: SWA absolute-position window (paged read).
    abs_pos_eager = position_ids_2d
    win_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1,
        1,
        window_size,
    )
    win_start = (abs_pos_eager.unsqueeze(-1) - window_size + 1).clamp(min=0)
    candidate = win_start + win_range
    swa_abs_idx = torch.where(
        candidate <= abs_pos_eager.unsqueeze(-1),
        candidate,
        torch.full_like(candidate, -1),
    )

    cache_seqlens_i32 = (position_ids_2d[:, -1] + 1).to(torch.int32)

    return DSv4DecodeAttnMetadataFP8(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
        position_ids=position_ids_flat,
        slot_mapping_swa=slot_swa,
        slot_mapping_compressed=slot_compressed,
        topk_window_idxs=topk_window,
        topk_buffer_compressed=topk_buffer_compressed,
        compressed_lens=compressed_lens,
        topk_total_by_ratio=topk_total_by_ratio,
        compressed_offset=window_size,
        is_cuda_graph=False,
        pool_block_tables=pool_block_tables,
        pool_write_slot_mappings=pool_write_slot_mappings,
        swa_abs_idx=swa_abs_idx,
        cache_seqlens_i32=cache_seqlens_i32,
    )
