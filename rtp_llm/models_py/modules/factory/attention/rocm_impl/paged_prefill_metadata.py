"""Batch-owned CK prefill metadata; no AITER or compiled RTP-LLM dependency."""

from dataclasses import dataclass
from typing import Optional

import torch


def sanitize_and_pad_block_table(
    block_table: Optional[torch.Tensor],
    seqlen_k: Optional[torch.Tensor],
    tokens_per_block: int,
    max_seqlen_k: int,
    block_positions_cache: Optional[torch.Tensor] = None,
):
    """Preserve valid IDs and fill padding for CK's speculative V prefetch.

    This is independent of the layer's KV storage: all layers in a cache group
    use the same logical block table. Never remove the speculative padding.
    """
    if block_table is None:
        return None, block_positions_cache
    if seqlen_k is None or block_table.dim() != 2:
        return block_table, block_positions_cache
    if seqlen_k.device != block_table.device:
        seqlen_k = seqlen_k.to(block_table.device, non_blocking=True)
    max_blocks_per_seq = block_table.shape[1]
    if seqlen_k.numel() != block_table.shape[0]:
        return block_table, block_positions_cache
    if (
        block_positions_cache is None
        or block_positions_cache.numel() < max_blocks_per_seq
        or block_positions_cache.device != block_table.device
    ):
        block_positions_cache = torch.arange(
            max(max_blocks_per_seq, 1024), dtype=torch.int32, device=block_table.device
        )
    valid_blocks = torch.div(
        seqlen_k + tokens_per_block - 1, tokens_per_block, rounding_mode="floor"
    ).to(torch.int32)
    positions = block_positions_cache[:max_blocks_per_seq].unsqueeze(0)
    valid_mask = positions < valid_blocks.unsqueeze(1)
    last_valid_col = (valid_blocks - 1).clamp(min=0).unsqueeze(1)
    last_valid_block_id = block_table.gather(1, last_valid_col.to(torch.int64))
    block_table = torch.where(
        valid_mask, block_table, last_valid_block_id.expand_as(block_table)
    )
    # kN0=128 is a conservative upper bound across CK head dimensions.
    extra_pages = (128 + tokens_per_block - 1) // tokens_per_block
    required_cols = (
        max_seqlen_k + tokens_per_block - 1
    ) // tokens_per_block + extra_pages
    if block_table.shape[1] < required_cols:
        pad_cols = required_cols - block_table.shape[1]
        block_table = torch.cat(
            [block_table, block_table[:, -1:].expand(-1, pad_cols)], dim=1
        )
    return block_table, block_positions_cache


@dataclass(frozen=True)
class PagedPrefillMetadata:
    seqlen_k: torch.Tensor
    block_table: torch.Tensor
    kv_indptr: torch.Tensor
    kv_page_indices: torch.Tensor
    descale: Optional[torch.Tensor]


def prepare_paged_prefill_metadata(
    cu_seqlens_k: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    max_seqlen_k: int,
    *,
    fp8: bool = False,
    seqlen_k: Optional[torch.Tensor] = None,
) -> PagedPrefillMetadata:
    """Prepare once per eager FMHAParams, never cache across batches/groups.

    Graph replay has mutable lengths and fixed-address buffers and must keep
    using its own refresh path, not this eager snapshot.
    """
    device = cu_seqlens_k.device
    if seqlen_k is None:
        seqlen_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    seqlen_k = seqlen_k.to(device=device, dtype=torch.int32)
    block_table = block_table.to(device=device, dtype=torch.int32)
    sanitized, _ = sanitize_and_pad_block_table(
        block_table, seqlen_k, tokens_per_block, max_seqlen_k
    )
    return PagedPrefillMetadata(
        seqlen_k=seqlen_k,
        block_table=sanitized,
        kv_indptr=torch.zeros(seqlen_k.numel() + 1, dtype=torch.int32, device=device),
        kv_page_indices=torch.zeros(1, dtype=torch.int32, device=device),
        descale=torch.ones(1, dtype=torch.float32, device=device) if fp8 else None,
    )
