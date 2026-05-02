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
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class DSv4DecodeAttnMetadata:
    """Metadata produced once per decode step, consumed by every layer.

    Fields are device-resident tensors unless noted. Built by
    :func:`build_decode_metadata` at the top of
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
    swa_buffer_t_dim: int  # = window_size
    compressed_buffer_t_dim_per_ratio: Dict[int, int]  # ratio -> max_seq_len // ratio

    # Per-request scalars (host int32 tensor; B-shaped)
    start_pos: torch.Tensor  # [B] int32 — start index of this step's tokens

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
    # Keys are PoolDescriptor attn_type ids (1=CSA_KV..7=SWA_KV); only
    # pools that the model actually uses are present.
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

    # Per-layer ``{attn_type: PoolDescriptor}``. Static reference (NOT a
    # tensor) — the pool tensor inside each descriptor is the framework
    # BlockPool handle, lifetime-managed by the C++ allocator. Attention
    # layers grab their pool views via this map at call time. ``None``
    # entries (or empty dict) mean "fall back to register_buffer".
    layer_pool_descs: Optional[List[Dict[int, "PoolDescriptor"]]] = None


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


def allocate_decode_metadata(
    max_batch_size: int,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    paged_pool_specs: Optional[Dict[int, Tuple[int, int]]] = None,
) -> "DSv4DecodeAttnMetadata":
    """Pre-allocate a ``DSv4DecodeAttnMetadata`` sized for ``max_batch_size``.

    Used by Phase 3 CUDA-graph capture: the impl owns a single metadata
    instance whose tensors live at fixed addresses; ``update_in_place``
    rewrites the contents per step. The captured graph then reads from
    these stable addresses on every replay.

    Tensor sizes are MAX, so ``update_in_place(start_pos)`` with
    ``start_pos.shape[0] == bs`` only writes the ``[:bs, ...]`` prefix.
    The captured graph at fixed BS slices these tensors at construction
    time (via ``DSv4DecodeFmhaImpl``); since each captured graph is
    per-BS, the slice offset is constant and known at capture.
    """
    B = max_batch_size
    T_total = B * q_len
    swa_buffer_stride = window_size

    start_pos = torch.zeros(B, dtype=torch.int32, device=device)
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
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import SWA_KV

        for attn_type, (_, max_blocks) in paged_pool_specs.items():
            pool_block_tables[attn_type] = torch.zeros(
                B, max_blocks, dtype=torch.int32, device=device
            )
            # SWA_KV is the always-write path (mask_negative=False downstream),
            # so it must hold a *valid* slot at all times — including during
            # framework warmup where update_decode_metadata_in_place's paged
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

    return DSv4DecodeAttnMetadata(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
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
    )


def update_decode_metadata_in_place(
    meta: "DSv4DecodeAttnMetadata",
    start_pos: torch.Tensor,
    forbid_realloc: bool = False,
    paged_block_tables: Optional[Dict[int, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[int, int]] = None,
) -> None:
    """Recompute every metadata buffer IN PLACE for a new ``start_pos``.

    Contract:
      * Every output tensor reuses its prior storage (``data_ptr()``
        unchanged); ``forbid_realloc=True`` makes any accidental realloc
        an immediate error rather than silent-correctness-bug.
      * Writes the prefix ``[:bs]`` (or ``[:bs * q_len]`` for slot
        mappings); leaves the tail at sentinel (-1) from
        ``allocate_decode_metadata``. The captured CUDA graph at that
        BS only reads the prefix, so the tail never matters.

    Mirrors :func:`build_decode_metadata` arithmetic. Kept as a separate
    function (not method) so it's callable from ``DSv4DecodeFmhaImpl.prepare``
    and from unit tests without instantiating the impl.

    Args:
        meta: Pre-allocated metadata (from ``allocate_decode_metadata``).
        start_pos: ``[bs]`` int — current absolute position per request.
            ``bs`` may be smaller than ``meta.batch_size`` (the alloc
            size); we write the ``[:bs]`` prefix only. For Phase 3
            CUDA-graph each captured graph is per-BS, so ``bs`` will
            equal ``meta.batch_size`` at runtime — but the prefix-only
            semantics keep this builder reusable for the eager path.
        forbid_realloc: If True, asserts every write reuses the existing
            tensor storage (sanity check for the captured-graph path).
    """
    bs = int(start_pos.shape[0])
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = meta.start_pos.device

    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)

    # snapshot pointers for the realloc-forbidden mode
    if forbid_realloc:
        ptr_snap = {
            "start_pos": meta.start_pos.data_ptr(),
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

    # SWA slot mapping prefix [:bs * q_len]
    swa_buffer_stride = window_size
    s_offsets = start_pos.view(bs, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)
    in_ring = s_offsets % window_size
    req_base = (
        torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1)
        * swa_buffer_stride
    )
    swa_slots = (req_base + in_ring).reshape(-1)
    meta.slot_mapping_swa[: bs * q_len].copy_(swa_slots)

    # Window topk indices [:bs, :, :] — left-aligned (mirrors _build_window_topk_idxs)
    abs_pos = s_offsets  # [bs, q_len]
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1,
        1,
        window_size,
    )
    abs_pos_b = abs_pos.unsqueeze(-1)  # [bs, q_len, 1]
    sp = (abs_pos % window_size).unsqueeze(-1)  # [bs, q_len, 1]
    ring_full_idx = (sp + 1 + k_range) % window_size
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    is_full = abs_pos_b >= (window_size - 1)
    window_idxs = torch.where(is_full, ring_full_idx, partial_idx)
    meta.topk_window_idxs[:bs].copy_(window_idxs)

    # Indexer output buffer reset to -1 prefix [:bs]
    meta.topk_buffer_compressed[:bs].fill_(-1)

    # Per-ratio compressed slot mappings + lens + topk_total
    for r, slot_t in meta.slot_mapping_compressed.items():
        stride = meta.compressed_buffer_t_dim_per_ratio[r]
        abs_pos_plus_1 = (
            start_pos.view(bs, 1)
            + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
            + 1
        )
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
        meta.compressed_lens[r][:bs].copy_(((start_pos + q_len) // r).to(torch.int32))

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
            cmp_lens = meta.compressed_lens[r][:bs].view(bs, 1, 1)
            valid_h = dense_idxs < cmp_lens
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
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.decode.pool_slot_mapping import (
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
            abs_pos_swa = (
                start_pos.view(bs, 1)
                + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
            ).reshape(-1)
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
            abs_pos_plus_1 = (
                start_pos.view(bs, 1)
                + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
                + 1
            )
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

    # Update Python-scalar geometry (cheap — these are not captured into the graph)
    meta.batch_size = bs
    meta.total_tokens = bs * q_len

    if forbid_realloc:
        # Verify every storage pointer is unchanged.
        cur = {
            "start_pos": meta.start_pos.data_ptr(),
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
                f"update_decode_metadata_in_place(forbid_realloc=True) "
                f"reallocated buffer '{k}': before={hex(p_before)} after={hex(cur[k])}"
            )


def build_decode_metadata(
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
) -> DSv4DecodeAttnMetadata:
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
        Fully-populated DSv4DecodeAttnMetadata.
    """
    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)

    B = start_pos.shape[0]
    T_total = B * q_len

    # SWA slot mapping (always needed).
    swa_buffer_stride = window_size
    slot_swa = _build_swa_slot_mapping(start_pos, q_len, window_size, swa_buffer_stride)

    # Window topk — request-local ring positions.
    topk_window = _build_window_topk_idxs(start_pos, q_len, window_size)

    # Per-ratio compressed slot mappings.
    unique_ratios: List[int] = sorted({r for r in compress_ratios if r > 1})
    slot_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_buffer_stride_per_ratio: Dict[int, int] = {}
    for r in unique_ratios:
        stride = max_seq_len // r
        compressed_buffer_stride_per_ratio[r] = stride
        slot_compressed[r] = _build_compressed_slot_mapping(start_pos, q_len, r, stride)
        # After-this-step length: floor((start_pos + q_len) / r)
        # (each request now has this many compressed entries).
        compressed_lens[r] = ((start_pos + q_len) // r).to(torch.int32)

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
            cmp_lens = compressed_lens[r].view(B, 1, 1)  # [B, 1, 1]
            valid = dense_idxs < cmp_lens
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
    # path doesn't share buffers across steps, unlike DSv4DecodeFmhaImpl).
    pool_block_tables: Dict[int, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[int, torch.Tensor] = {}
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.decode.pool_slot_mapping import (
            compute_kv_pool_slot_mapping,
        )

        # Snapshot block tables (clone so downstream writes can't surprise
        # the framework's tensor).
        for at, bt in paged_block_tables.items():
            pool_block_tables[at] = bt[:B].contiguous().clone()

        if SWA_KV in pool_block_tables:
            E = paged_pool_entries_per_block.get(SWA_KV, window_size)
            abs_pos_swa = (
                start_pos.view(B, 1)
                + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
            ).reshape(-1)
            pool_write_slot_mappings[SWA_KV] = compute_kv_pool_slot_mapping(
                pool_block_tables[SWA_KV],
                abs_pos_swa,
                E,
            )

        for ratio_key, attn_type_writers in (
            (4, [CSA_KV, INDEXER_KV]),
            (128, [HCA_KV]),
        ):
            if ratio_key not in compressed_lens:
                continue
            abs_pos_plus_1 = (
                start_pos.view(B, 1)
                + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
                + 1
            )
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
    abs_pos_eager = start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)
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

    return DSv4DecodeAttnMetadata(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
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
    )
