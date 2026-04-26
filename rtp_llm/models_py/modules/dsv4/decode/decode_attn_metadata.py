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
    # base ring offset for read: starts after the new write
    # When the ring is full (abs_pos >= win-1), the OLDEST slot is at (abs_pos+1) % win,
    # and we read win consecutive cyclic positions ending at abs_pos % win.
    # For abs_pos < win-1, valid tokens are [0..abs_pos]; we put them in the LAST slots
    # of the [win] indices (so the kernel sees a "right-aligned" decode window).
    # Implementation: for each col k in [0..win), read pos = (abs_pos - (win-1-k)) when >=0, else -1.
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    abs_pos_b = abs_pos.unsqueeze(-1)  # [B, q_len, 1]
    desired_pos = abs_pos_b - (window_size - 1) + k_range  # [B, q_len, win]
    valid = desired_pos >= 0
    # Map to ring slot, mask invalid to -1.
    ring_slot = desired_pos % window_size
    return torch.where(valid, ring_slot, torch.full_like(ring_slot, -1))


def allocate_decode_metadata(
    max_batch_size: int,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
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
    )


def update_decode_metadata_in_place(
    meta: "DSv4DecodeAttnMetadata",
    start_pos: torch.Tensor,
    forbid_realloc: bool = False,
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

    # Window topk indices [:bs, :, :]
    abs_pos = s_offsets  # [bs, q_len]
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1,
        1,
        window_size,
    )
    desired_pos = (
        abs_pos.unsqueeze(-1) - (window_size - 1) + k_range
    )  # [bs, q_len, win]
    valid = desired_pos >= 0
    ring_slot = desired_pos % window_size
    window_idxs = torch.where(valid, ring_slot, torch.full_like(ring_slot, -1))
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
    )
