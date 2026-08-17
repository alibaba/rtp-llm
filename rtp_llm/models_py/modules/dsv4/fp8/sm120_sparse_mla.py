"""Shared FlashInfer adapter for DeepSeek-V4 sparse MLA on SM120."""

from __future__ import annotations

from typing import Optional, Sequence

import torch


_WORKSPACES: dict[torch.device, torch.Tensor] = {}


def workspace(device: torch.device) -> torch.Tensor:
    result = _WORKSPACES.get(device)
    if result is None:
        result = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        _WORKSPACES[device] = result
    return result


def token_lens(
    lengths: Optional[torch.Tensor], rows: int, width: int, device: torch.device
) -> torch.Tensor:
    if lengths is None:
        return torch.full((rows,), width, dtype=torch.int32, device=device)
    result = lengths.to(device=device, dtype=torch.int32).reshape(-1)
    if rows % result.numel() == 0 and result.numel() != rows:
        result = result.repeat_interleave(rows // result.numel())
    if result.numel() != rows:
        raise ValueError(f"top-k lengths have {result.numel()} entries; expected {rows}")
    return result.contiguous()


def canonical_topk(
    indices: torch.Tensor,
    lengths: Optional[torch.Tensor],
    supported_widths: Sequence[int],
    *,
    trim_dspark_padding: bool = False,
    pad_to_supported: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a contiguous, instantiated FlashInfer Top-K shape."""
    indices = indices.to(torch.int32).contiguous()
    rows, width = indices.shape
    if trim_dspark_padding and width == 256:
        width = 128
        indices = indices[:, :width].contiguous()
    if width not in supported_widths:
        if not pad_to_supported:
            raise RuntimeError(
                "SM120 sparse MLA supports Top-K widths "
                f"{'/'.join(map(str, supported_widths))}, got {width}"
            )
        padded_width = next((value for value in supported_widths if value >= width), None)
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
    return indices, token_lens(lengths, rows, width, indices.device)


def pack_logical_workspace(
    pool: torch.Tensor, indices: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather arbitrary RTP slots into the contiguous page ABI used by SM120."""
    from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
        gather_k_cache_slots_packed,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
        insert_packed_k_cache_flat,
    )

    flat_indices = indices.reshape(-1)
    packed_rows = gather_k_cache_slots_packed(pool, flat_indices)
    slot_count = int(flat_indices.numel())
    packed = torch.zeros(
        (max((slot_count + page_size - 1) // page_size, 1), page_size, pool.shape[-1]),
        dtype=pool.dtype,
        device=pool.device,
    )
    local_slots = torch.arange(slot_count, dtype=torch.int64, device=pool.device)
    insert_packed_k_cache_flat(packed_rows, packed, local_slots)
    remapped = local_slots.to(torch.int32).masked_fill(flat_indices < 0, 0)
    return packed, remapped.view_as(indices)


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
    """Invoke the sole SM120 sparse-MLA kernel used by prefill and decode."""
    try:
        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4
    except ImportError as exc:
        raise RuntimeError(
            "SM120 DSV4 sparse attention requires FlashInfer's "
            "trtllm_batch_decode_sparse_mla_dsv4"
        ) from exc

    # FlashInfer's SM120 DSV4 instantiations currently start at 16 local
    # attention heads. TP=8 leaves eight heads per rank, but attention heads
    # are independent, so pad only that SM120 shape to 16 and discard the
    # dummy outputs. This keeps TP=4 and all non-SM120 paths unchanged and
    # avoids falling back to a reference implementation.
    kernel_query = query.contiguous()
    kernel_sinks = sinks.float()
    kernel_out = out
    original_heads = int(query.shape[-2])
    if original_heads == 8:
        kernel_query = torch.cat((kernel_query, torch.zeros_like(kernel_query)), dim=-2)
        kernel_sinks = torch.cat((kernel_sinks, torch.zeros_like(kernel_sinks)), dim=-1)
        kernel_out = torch.empty_like(kernel_query)

    trtllm_batch_decode_sparse_mla_dsv4(
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
