"""Batch-owned CK prefill metadata; no AITER or compiled RTP-LLM dependency."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def borrow_prefill_cumulative_lengths(attn_inputs, device: torch.device):
    """Reuse the runner's stream-ordered device metadata, only for eager.

    PyWrappedModel flushes its staged H2D copies before prepare_fmha_impl and
    retains their sources. Never borrow these buffers for a captured graph:
    graph refresh requires its own stable storage, independent of live inputs.
    Standalone callers without complete device metadata retain the old path.
    """
    if getattr(attn_inputs, "is_cuda_graph", False):
        return None
    lengths = attn_inputs.input_lengths
    if lengths.device.type != "cpu" or lengths.dim() != 1:
        return None
    tensors = tuple(
        getattr(attn_inputs, name, None)
        for name in ("cu_seqlens_device", "cu_kv_seqlens_device")
    )
    if not all(
        t is not None
        and t.device == device
        and t.dtype == torch.int32
        and t.dim() == 1
        and t.numel() == lengths.numel() + 1
        and t.is_contiguous()
        for t in tensors
    ):
        return None
    return tensors


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
    # Retain pinned sources until the batch is retired. HIP async H2D must not
    # outlive its source, and this storage must never be reused across batches.
    host_staging: Optional[torch.Tensor] = None


def prepare_host_paged_prefill_metadata(
    host_block_table: torch.Tensor,
    host_seqlen_k: torch.Tensor,
    tokens_per_block: int,
    max_seqlen_k: int,
    device: torch.device,
    *,
    fp8: bool = False,
) -> PagedPrefillMetadata:
    """One batch-owned H2D instead of a chain of tiny GPU setup kernels.

    Only consumes authoritative host metadata; never copies GPU data back.
    Keep CK's speculative-prefetch padding and preserve every valid block ID.
    Graph replay deliberately continues using its fixed-address refresh path.
    """
    if (
        host_block_table.device.type != "cpu"
        or host_seqlen_k.device.type != "cpu"
        or host_block_table.dtype != torch.int32
        or host_seqlen_k.dtype != torch.int32
        or host_block_table.dim() != 2
        or host_seqlen_k.dim() != 1
        or host_block_table.shape[0] != host_seqlen_k.numel()
        or host_block_table.shape[1] == 0
        or tokens_per_block <= 0
    ):
        raise ValueError("CK host metadata requires int32 host lengths and block table")
    table = host_block_table.numpy()
    lengths = host_seqlen_k.numpy().astype(np.int64, copy=False)
    if np.any(lengths < 0) or np.any(lengths > max_seqlen_k):
        raise ValueError("CK host lengths exceed the declared sequence bound")
    valid = (lengths + tokens_per_block - 1) // tokens_per_block
    if np.any(valid > table.shape[1]):
        raise ValueError("CK host block table does not cover valid KV tokens")
    batch, old_cols = table.shape
    cols = max(
        old_cols,
        (max_seqlen_k + tokens_per_block - 1) // tokens_per_block
        + (128 + tokens_per_block - 1) // tokens_per_block,
    )
    # int32 lengths, indptr, one dummy page index, then the padded table.
    table_offset = 2 * batch + 2
    host = torch.empty(
        table_offset + batch * cols,
        dtype=torch.int32,
        pin_memory=device.type == "cuda",
    )
    values = host.numpy()
    values[:batch] = lengths
    values[batch:table_offset] = 0
    padded = values[table_offset:].reshape(batch, cols)
    last = table[np.arange(batch), np.maximum(valid - 1, 0)]
    padded[:] = last[:, None]
    # Do not sanitize invalid IDs *inside* the valid range: preserve semantics
    # and let existing upstream validation expose bad input rather than hide it.
    padded[:, :old_cols] = np.where(
        np.arange(old_cols)[None, :] < valid[:, None], table, last[:, None]
    )
    packed = host.to(device=device, non_blocking=True)
    return PagedPrefillMetadata(
        seqlen_k=packed[:batch],
        kv_indptr=packed[batch : 2 * batch + 1],
        kv_page_indices=packed[2 * batch + 1 : table_offset],
        block_table=packed[table_offset:].view(batch, cols),
        descale=torch.ones(1, dtype=torch.float32, device=device) if fp8 else None,
        host_staging=host,
    )


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
