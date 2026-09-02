from __future__ import annotations

from typing import Optional, Sequence

import torch

_WORKSPACES: dict[torch.device, torch.Tensor] = {}

# FlashInfer's DSV4 sparse-decode dispatcher specializes on the static index
# width.  HCA compresses one entry per 128 input tokens, so a 1M context needs
# an 8192-wide extra sparse table.  Keep the complete production width set in
# one place so eager and CUDA-graph preparation cannot diverge.
SM120_SWA_TOPK_WIDTHS = (128, 512, 1024)
SM120_EXTRA_TOPK_WIDTHS = (2, 128, 512, 1024, 2048, 4096, 8192)


def validate_sm120_swa_topk_width(
    requested_width: int,
    *,
    context: str,
) -> int:
    """Return the decode-kernel width used for an SWA request.

    FlashInfer precompiles the SM120 DSV4 decode launcher only for the widths
    in ``SM120_SWA_TOPK_WIDTHS``.  Canonicalization may pad up to the next
    instance, but it must never let an unsupported large window reach the
    first live request.
    """
    requested_width = int(requested_width)
    if requested_width <= 0:
        raise ValueError(
            f"{context} requires a positive SWA Top-K width, got {requested_width}"
        )
    kernel_width = next(
        (width for width in SM120_SWA_TOPK_WIDTHS if width >= requested_width),
        None,
    )
    if kernel_width is None:
        raise RuntimeError(
            f"{context} requires SWA Top-K width {requested_width}, which exceeds "
            "the largest SM120 FlashInfer DSV4 decode instantiation "
            f"({SM120_SWA_TOPK_WIDTHS[-1]}). Reduce sliding_window/DSpark "
            "proposal width or use a backend with a matching kernel instance."
        )
    return kernel_width


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
        _load_sm120_sparse_mla,
    )

    sm120_sparse_mla = _load_sm120_sparse_mla()
    if sm120_sparse_mla is None:
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
    sm120_sparse_mla(
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
