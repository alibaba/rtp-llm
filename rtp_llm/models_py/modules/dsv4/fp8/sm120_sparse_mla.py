from __future__ import annotations

from typing import Optional, Sequence

import torch

_WORKSPACES: dict[torch.device, torch.Tensor] = {}
_PACKED_CACHE: dict[tuple, torch.Tensor] = {}
_GATHERED_CACHE: dict[tuple, torch.Tensor] = {}
_SLOT_CACHE: dict[tuple, torch.Tensor] = {}
_REMAP_CACHE: dict[tuple, torch.Tensor] = {}
_LOOKUP_CACHE: dict[tuple, torch.Tensor] = {}
_NEGATIVE_MASK_CACHE: dict[tuple, torch.Tensor] = {}

# FlashInfer's DSV4 sparse-decode dispatcher specializes on the static index
# width.  HCA compresses one entry per 128 input tokens, so a 1M context needs
# an 8192-wide extra sparse table.  Keep the complete production width set in
# one place so eager and CUDA-graph preparation cannot diverge.
SM120_SWA_TOPK_WIDTHS = (128, 512, 1024)
SM120_EXTRA_TOPK_WIDTHS = (2, 128, 512, 1024, 2048, 4096, 8192)


def _cached_buffer(
    cache: dict[tuple, torch.Tensor],
    key: tuple,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a grow-only buffer, reusing the largest shape seen for a key.

    Decode graph capture uses a small set of static top-k widths but may see
    different batch sizes.  Keying by capacity would retain one large tensor
    for every batch/sequence combination and eventually exhaust HBM.  A
    grow-only buffer per device/layout instead bounds the cache while keeping
    the eager-to-graph reuse that avoids allocations in the hot path.
    """
    needed = 1
    for extent in shape:
        needed *= int(extent)
    result = cache.get(key)
    if result is None or result.numel() < needed:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM120 sparse MLA packed buffers must be materialized before "
                "CUDA graph capture"
            )
        result = torch.empty(shape, dtype=dtype, device=device)
        cache[key] = result
    # Reuse the prefix of a larger grow-only allocation.  ``view(shape)`` on
    # the full tensor would fail as soon as a later request has fewer slots
    # than the largest request seen so far.
    return result.reshape(-1)[:needed].view(shape)


def workspace(device: torch.device) -> torch.Tensor:
    result = _WORKSPACES.get(device)
    if result is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM120 sparse MLA workspace must be materialized before "
                "CUDA graph capture"
            )
        result = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = result
    return result


def warmup(device: torch.device) -> torch.Tensor:
    """Materialize the fixed FlashInfer workspace before graph capture."""
    return workspace(device)


def token_lens(
    lengths: Optional[torch.Tensor],
    rows: int,
    width: int,
    device: torch.device,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize per-row sparse-index lengths for the FlashInfer ABI.

    ``lengths`` is a *prefix* length: the kernel scans the first ``length``
    entries of each row.  A few metadata producers can leave ``-1`` holes in
    that prefix (for example an empty page during the first decode step), so
    callers may provide ``valid_mask`` to derive the effective count instead.
    Keeping this conversion on the input device also avoids a host sync in
    CUDA-graph replay.
    """
    if rows < 0 or width < 0:
        raise ValueError(f"invalid sparse-index shape rows={rows}, width={width}")

    if lengths is None:
        if valid_mask is None:
            return torch.full((rows,), width, dtype=torch.int32, device=device)
        mask = valid_mask.to(device=device, dtype=torch.bool).reshape(rows, width)
        return mask.sum(-1, dtype=torch.int32).contiguous()

    result = lengths.to(device=device, dtype=torch.int32).reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"top-k lengths are empty; expected {rows} entries")
    if rows % result.numel() == 0 and result.numel() != rows:
        result = result.repeat_interleave(rows // result.numel())
    if result.numel() != rows:
        raise ValueError(
            f"top-k lengths have {result.numel()} entries; expected {rows}"
        )
    # FlashInfer has no useful interpretation for a count outside the static
    # index width.  Clamp rather than allowing an out-of-bounds scan; this is
    # especially important for graph warmup buffers whose values are filled in
    # place on a later replay.
    result = result.clamp_(min=0, max=width)
    if valid_mask is not None:
        mask = valid_mask.to(device=device, dtype=torch.bool).reshape(rows, width)
        prefix = torch.arange(width, device=device).unsqueeze(0) < result.unsqueeze(1)
        result = (mask & prefix).sum(-1, dtype=torch.int32)
    return result.contiguous()


def canonical_topk(
    indices: torch.Tensor,
    lengths: Optional[torch.Tensor],
    supported_widths: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and canonicalize sparse indices for the fixed-width kernel.

    FlashInfer consumes a prefix of each row, as described by its length
    tensor.  Therefore invalid ``-1`` entries are stably moved to the right
    before the length is calculated.  This preserves the order of selected
    slots while preventing an interspersed padding entry from shortening the
    effective prefix and silently dropping a later selected slot.
    """
    indices = indices.to(torch.int32).contiguous()
    if indices.dim() != 2:
        raise ValueError(
            f"sparse top-k indices must be 2D [rows, width], got {indices.shape}"
        )
    rows, width = indices.shape
    if not supported_widths or any(int(value) <= 0 for value in supported_widths):
        raise ValueError(
            f"supported top-k widths must be positive, got {supported_widths}"
        )
    if width not in supported_widths:
        padded_width = next(
            (value for value in supported_widths if value >= width), None
        )
        if padded_width is None:
            raise RuntimeError(
                f"SM120 sparse MLA Top-K width {width} exceeds the largest "
                f"FlashInfer instantiation ({supported_widths[-1]})"
            )
        padded = torch.full(
            (rows, padded_width), -1, dtype=torch.int32, device=indices.device
        )
        padded[:, :width] = indices
        indices = padded

    width = indices.shape[-1]
    valid = indices >= 0
    # An explicit length denotes a prefix in the original layout.  Ignore
    # entries beyond it before compacting; otherwise stale values in the tail
    # could become live merely because an earlier slot was -1.
    normalized_lengths = (
        token_lens(lengths, rows, width, indices.device)
        if lengths is not None
        else None
    )
    if normalized_lengths is not None:
        prefix = torch.arange(width, device=indices.device).unsqueeze(0) < (
            normalized_lengths.unsqueeze(1)
        )
        valid &= prefix

    # Stable sort by validity (valid first) keeps selected-slot order while
    # producing a fixed-shape tensor suitable for CUDA graph replay.  Avoid
    # ``masked_select`` here: its dynamically sized allocation is not graph
    # safe when the number of valid slots changes between replays.
    order = torch.argsort((~valid).to(torch.int8), dim=-1, stable=True)
    canonical = indices.gather(1, order)
    canonical_valid = valid.gather(1, order)
    canonical.masked_fill_(~canonical_valid, -1)
    effective_lengths = canonical_valid.sum(-1, dtype=torch.int32)
    return canonical, effective_lengths.contiguous()


def pack_logical_workspace(
    pool: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
    *,
    namespace: str = "default",
) -> tuple[torch.Tensor, torch.Tensor]:
    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
        gather_k_cache_slots_packed,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
        insert_packed_k_cache_flat,
    )

    flat_indices = indices.reshape(-1)
    # Padding slots use -1 by contract.  Gather a harmless row for them, then
    # keep -1 in the remapped indices so FlashInfer masks the slot instead of
    # accidentally attending to physical slot zero.
    slot_count = int(flat_indices.numel())
    # The SWA and extra-sparse pools can have identical layouts (notably both
    # use 64-token pages).  Keep their packed pages and remap workspaces
    # independent: the two calls happen back-to-back, but both tensors are
    # consumed by FlashInfer after the second call returns.
    cache_key = (
        namespace,
        pool.device,
        int(page_size),
        int(pool.shape[-1]),
        pool.dtype,
    )
    slot_key = cache_key
    lookup_indices = _cached_buffer(
        _LOOKUP_CACHE,
        slot_key,
        (slot_count,),
        dtype=torch.int64,
        device=pool.device,
    )
    lookup_indices.copy_(flat_indices)
    lookup_indices.clamp_min_(0)
    packed_rows = _cached_buffer(
        _GATHERED_CACHE,
        cache_key,
        (slot_count, pool.shape[-1]),
        dtype=pool.dtype,
        device=pool.device,
    )
    gather_k_cache_slots_packed(pool, lookup_indices, out=packed_rows)
    packed_pages = max((slot_count + page_size - 1) // page_size, 1)
    packed = _cached_buffer(
        _PACKED_CACHE,
        cache_key,
        (packed_pages, page_size, pool.shape[-1]),
        dtype=pool.dtype,
        device=pool.device,
    )
    packed.zero_()
    local_slots = _cached_buffer(
        _SLOT_CACHE,
        slot_key,
        (slot_count,),
        dtype=torch.int64,
        device=pool.device,
    )
    # The grow-only buffer may have been allocated for a larger request; only
    # populate the range used by this invocation.
    local_slots.copy_(torch.arange(slot_count, dtype=torch.int64, device=pool.device))
    insert_packed_k_cache_flat(packed_rows, packed, local_slots)
    remap = _cached_buffer(
        _REMAP_CACHE,
        slot_key,
        (slot_count,),
        dtype=torch.int32,
        device=pool.device,
    )
    remap.copy_(local_slots)
    negative_mask = _cached_buffer(
        _NEGATIVE_MASK_CACHE,
        slot_key,
        (slot_count,),
        dtype=torch.bool,
        device=pool.device,
    )
    torch.lt(flat_indices, 0, out=negative_mask)
    remap.masked_fill_(negative_mask, -1)
    return packed, remap.view_as(indices)


def run(
    *,
    query: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    sinks: torch.Tensor,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lens: Optional[torch.Tensor] = None,
) -> None:
    from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
        _sm120_sparse_mla,
    )

    if _sm120_sparse_mla is None:
        raise RuntimeError(
            "SM120 sparse MLA kernel was unavailable during model initialization"
        )

    kernel_query = query.contiguous()
    kernel_sinks = sinks.float()
    kernel_out = out
    original_heads = int(query.shape[-2])
    if original_heads == 8:
        kernel_query = torch.cat((kernel_query, torch.zeros_like(kernel_query)), dim=-2)
        kernel_sinks = torch.cat((kernel_sinks, torch.zeros_like(kernel_sinks)), dim=-1)
        kernel_out = torch.empty_like(kernel_query)
    _sm120_sparse_mla(
        query=kernel_query,
        swa_kv_cache=swa_cache.unsqueeze(-2),
        workspace_buffer=workspace(query.device),
        sparse_indices=swa_indices,
        compressed_kv_cache=(
            extra_cache.unsqueeze(-2) if extra_cache is not None else None
        ),
        out=kernel_out,
        bmm1_scale=scale,
        sinks=kernel_sinks,
        kv_layout="NHD",
        swa_topk_lens=swa_lens,
        extra_sparse_indices=extra_indices,
        extra_sparse_topk_lens=extra_lens,
    )
    if kernel_out is not out:
        out.copy_(kernel_out[..., :original_heads, :])
