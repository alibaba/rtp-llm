"""DSV4 prefill-path paged KV write.

Writes K/V (and compressor compressed-K) directly into the framework's
``BlockPool`` paged tensors, bypassing the per-Attention dense
``register_buffer("kv_cache", ...)`` intermediate. Lets the prefill
reuse path skip ``_gather_all_layers`` / ``_scatter_all_layers`` for
the KV pools (1/2/7) entirely — only the small STATE pools (4/5/6)
still need gather/scatter for compressor running state.

The ``BlockPool`` storage for V4 KV pools is a ``[num_blocks,
stride_bytes] uint8`` tensor whose stride is exactly
``tokens_per_block * head_dim * 2`` bytes (BF16-only path; see
``DSV4ConfigCreator.cc``). We view it as
``[num_blocks, tokens_per_block, head_dim] bf16`` and index by
``(physical_block_id, slot_in_block)``.

This module is BlockPool-shape-aware but DSV4-policy-agnostic: callers
decide what global slot each new token writes to (sequential for CSA /
HCA / INDEXER compressed entries; ring-mod-window for SWA).
"""

from __future__ import annotations

from typing import Optional

import torch


def _pool_as_dense_view(pool_tensor: torch.Tensor, tokens_per_block: int, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """View a ``[num_blocks, stride_bytes] uint8`` BlockPool tensor as
    ``[num_blocks, tokens_per_block, head_dim]`` of ``dtype``.

    Asserts the stride matches ``tokens_per_block * head_dim *
    dtype.itemsize`` so a layout drift in DSV4ConfigCreator surfaces
    immediately rather than corrupting silently.
    """
    if pool_tensor.dim() != 2 or pool_tensor.dtype != torch.uint8:
        raise ValueError(
            f"pool_tensor must be [num_blocks, stride_bytes] uint8, got "
            f"shape={tuple(pool_tensor.shape)} dtype={pool_tensor.dtype}"
        )
    expected_stride = tokens_per_block * head_dim * torch.empty((), dtype=dtype).element_size()
    if pool_tensor.size(1) != expected_stride:
        raise ValueError(
            f"pool stride {pool_tensor.size(1)} != tokens_per_block({tokens_per_block}) "
            f"* head_dim({head_dim}) * esize({torch.empty((), dtype=dtype).element_size()}) "
            f"= {expected_stride}"
        )
    num_blocks = pool_tensor.size(0)
    return pool_tensor.view(dtype).view(num_blocks, tokens_per_block, head_dim)


def write_paged_kv_per_req(
    src_kv: torch.Tensor,
    pool_tensor: torch.Tensor,
    block_table_row: torch.Tensor,
    slot_offset: int,
    tokens_per_block: int,
    head_dim: int,
    ring_modulo: int = 0,
) -> None:
    """In-place paged write of one request's new tokens.

    For each token ``i in [0, seqlen)``::

        global_slot = slot_offset + i
        if ring_modulo > 0:
            global_slot = global_slot % ring_modulo
        block_idx = global_slot // tokens_per_block
        slot     = global_slot % tokens_per_block
        phys     = block_table_row[block_idx]
        pool[phys, slot, :] = src_kv[i]

    Args:
        src_kv: ``[seqlen, head_dim]`` bf16 — new K/V rows to write.
        pool_tensor: ``[num_blocks, stride_bytes] uint8`` — BlockPool
            tensor for the target pool. Stride must match
            ``tokens_per_block * head_dim * 2`` bytes.
        block_table_row: ``[max_blocks_per_req]`` int32 — physical
            block ids assigned to this request for the target pool.
        slot_offset: starting global slot index. For sequential writes
            (CSA/HCA/INDEXER compressed entries) this is the request's
            ``prefix_len // ratio`` (CSA/HCA) or ``prefix_len`` (raw
            paged KV). For SWA it is the request's ``prefix_len`` and
            ``ring_modulo`` is set to ``win``.
        tokens_per_block: entries per BlockPool block (256 for SWA, 64
            for CSA/INDEXER, 2 for HCA).
        head_dim: per-entry element count (512 for KV, 128 for INDEXER).
        ring_modulo: ``0`` = sequential write; ``>0`` = ring write
            (slot wrapped mod ``ring_modulo``). Used only by SWA.
    """
    if src_kv.numel() == 0:
        return
    seqlen = src_kv.size(0)
    if seqlen == 0:
        return
    if src_kv.size(-1) != head_dim:
        raise ValueError(f"src_kv last dim {src_kv.size(-1)} != head_dim {head_dim}")
    device = src_kv.device

    pool_view = _pool_as_dense_view(pool_tensor, tokens_per_block, head_dim, src_kv.dtype)

    pos = torch.arange(seqlen, device=device, dtype=torch.long) + int(slot_offset)
    if ring_modulo > 0:
        pos = pos % int(ring_modulo)
    block_idx = pos // int(tokens_per_block)
    slot = pos % int(tokens_per_block)
    if block_table_row.dtype != torch.long:
        block_table_row = block_table_row.to(torch.long)
    phys = block_table_row[block_idx]

    pool_view[phys, slot] = src_kv


def write_paged_kv_swa_per_req(
    src_kv: torch.Tensor,
    pool_tensor: torch.Tensor,
    block_table_row: torch.Tensor,
    start_pos: int,
    seqlen: int,
    win: int,
    head_dim: int,
    tokens_per_block: int = 256,
) -> None:
    """SWA-specific paged write that mirrors the ring-buffer policy of
    the legacy ``Attention.forward`` prefill arm.

    Final state semantics (equivalent to)::

        for t in range(seqlen):
            pool_view[phys, (start_pos + t) % win, :] = src_kv[t]

    Implementation: when ``seqlen > win``, the ring buffer's final state
    is determined entirely by the *last* ``win`` source rows (they pin
    every ring slot exactly once, in last-write-wins order).  We exploit
    that to write only those ``win`` rows — avoiding the
    PyTorch-undefined behavior of fancy-index assignment with repeated
    ``(phys, slot)`` keys (which the naive vectorized form would emit
    for every slot ``seqlen / win`` times, with only the final write
    that "wins" left to the executor's discretion).

    For ``seqlen <= win`` every slot is written at most once, so the
    vectorized form is well-defined and we use it directly.
    """
    if seqlen <= 0 or src_kv.numel() == 0:
        return
    if seqlen <= win:
        # Single-pass write — every ring slot touched at most once.
        write_paged_kv_per_req(
            src_kv=src_kv[:seqlen],
            pool_tensor=pool_tensor,
            block_table_row=block_table_row,
            slot_offset=int(start_pos),
            tokens_per_block=tokens_per_block,
            head_dim=head_dim,
            ring_modulo=int(win),
        )
        return
    # seqlen > win: keep only the last ``win`` rows.  Their global
    # positions are [start_pos + seqlen - win, start_pos + seqlen), and
    # their ring slots ``(g % win)`` form a permutation of [0, win) —
    # one-shot write, no repeated indices.
    last_offset = int(start_pos) + int(seqlen) - int(win)
    write_paged_kv_per_req(
        src_kv=src_kv[seqlen - win :],
        pool_tensor=pool_tensor,
        block_table_row=block_table_row,
        slot_offset=last_offset,
        tokens_per_block=tokens_per_block,
        head_dim=head_dim,
        ring_modulo=int(win),
    )


def gather_paged_kv_per_req(
    pool_tensor: torch.Tensor,
    block_table_row: torch.Tensor,
    slot_indices: torch.Tensor,
    tokens_per_block: int,
    head_dim: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Read ``[N, head_dim]`` from a paged pool by global slot indices.

    Used for parity testing / debug: compares against
    ``write_paged_kv_per_req`` to verify a roundtrip is bit-equal.

    Args:
        pool_tensor: ``[num_blocks, stride_bytes] uint8``.
        block_table_row: ``[max_blocks_per_req]`` int32 — request's
            physical block ids.
        slot_indices: ``[N]`` int — global slot indices to read.
        tokens_per_block: pool's entries-per-block.
        head_dim: per-entry element count.
        out_dtype: dtype to read as. Defaults to bf16.
    """
    dtype = out_dtype if out_dtype is not None else torch.bfloat16
    pool_view = _pool_as_dense_view(pool_tensor, tokens_per_block, head_dim, dtype)
    if slot_indices.dtype != torch.long:
        slot_indices = slot_indices.to(torch.long)
    if block_table_row.dtype != torch.long:
        block_table_row = block_table_row.to(torch.long)
    block_idx = slot_indices // int(tokens_per_block)
    slot = slot_indices % int(tokens_per_block)
    phys = block_table_row[block_idx]
    return pool_view[phys, slot]
