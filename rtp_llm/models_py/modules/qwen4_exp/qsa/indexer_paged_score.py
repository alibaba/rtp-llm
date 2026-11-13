"""Paged (decode) 4-head KSA indexer scoring kernel.

Contract mirrors ``fp8_paged_indexer_score``: scores a decode-time query
against the paged indexer KV pool using a ``block_table`` for logical →
physical indirection.  One program per ``(batch, token)`` row tiles over
``max_ctx_len`` output columns in chunks so large pool sizes do not blow
up register usage.

Pool layout is bf16 per entry (each entry is ``[D]`` bf16). q is also
received as bf16 -- the caller is responsible for dequantising before
calling.  :mod:`indexer_score` exercises the same score math, but its legacy
FP8 + scale input is not ABI-compatible with this production side pool.

Per row::

    logits[row, t] = sum_h weight[row, h] * relu(dot(q[row, h, :],
        pool[slot(row, t), :]))

    batch(row) = row // next_n
    slot(row, t) = block_table[batch(row), t // block_size] * block_size + t % block_size
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_N = 64


@triton.jit
def _qsa_paged_score_kernel(
    q_ptr,
    w_ptr,
    pool_ptr,
    bt_ptr,
    ctx_ptr,
    out_ptr,
    stride_q_row,
    max_ctx_len,
    max_blocks,
    pool_blocks,
    next_n: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    block_size: tl.constexpr,
    BN: tl.constexpr,
):
    rd = tl.arange(0, D)
    row = tl.program_id(0)
    blk = tl.program_id(1)
    batch_idx = row // next_n

    rn = blk * BN + tl.arange(0, BN)
    n_mask = rn < max_ctx_len

    T = tl.load(ctx_ptr + row)
    in_context = n_mask & (rn < T)

    bi = rn // block_size
    ei = rn % block_size
    in_block_table = bi < max_blocks
    physical_bt = tl.load(
        bt_ptr + batch_idx * max_blocks + bi,
        mask=in_context & in_block_table,
        other=0,
    )
    # Block zero (and negative IDs) are allocation sentinels.  The upper bound
    # is checked again in the kernel even though the wrapper validates it, so a
    # bad table can never turn into an out-of-bounds pool read.
    has_block = (
        in_context & in_block_table & (physical_bt > 0) & (physical_bt < pool_blocks)
    )
    slot = physical_bt * block_size + ei

    acc = tl.zeros((BN,), dtype=tl.float32)
    for h in tl.static_range(H):
        qh = (
            tl.load(q_ptr + row * stride_q_row + h * D + rd)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        w_h = tl.load(w_ptr + row * H + h)

        k = (
            tl.load(
                pool_ptr + slot[:, None] * D + rd[None, :],
                mask=has_block[:, None],
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        dot = tl.sum(qh[None, :] * k, axis=1)
        # The model contract is ``weight * relu(dot)``.  Applying ReLU after
        # the multiply happens to agree for the released positive dequant
        # weights, but changes the public helper's semantics for negative
        # weights and can turn a negative contribution into zero.
        acc += tl.where(has_block, tl.maximum(dot, 0.0) * w_h, 0.0)

    tl.store(
        out_ptr + row * max_ctx_len + rn,
        tl.where(has_block, acc, float("-inf")),
        mask=n_mask,
    )


def qsa_paged_indexer_score(
    q_bf16: torch.Tensor,
    weight: torch.Tensor,
    kv_pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    max_ctx_len: Optional[int] = None,
    block_n: int = _BLOCK_N,
    validate_block_table: bool = True,
) -> torch.Tensor:
    """Score one decode step against the paged indexer KV pool.

    Args:
        q_bf16: ``[B, next_n, H, D]`` bf16.
        weight: ``[B * next_n, H]`` fp32 -- per-(row, head) weight.
            Qwen4 folds the q dequant scale into it.
        kv_pool: ``[total_slots, D]`` bf16 -- flat indexer KV pool view.
        block_table: ``[B, max_blocks]`` int32 -- logical -> physical block map.
        context_lens: ``[B, next_n]`` int32 -- valid compressed entries / row.
        block_size: compressed entries per physical pool block
            (``kernel_seq_size_per_block / ratio``).  For the qwen4 production
            shape with ratio=4 and kernel_block=256, block_size=64.
        max_ctx_len: output column count.  Defaults to ``context_lens.max()``
            (requires synchronisation).
        validate_block_table: Check every physical block ID on the host.  The
            QSA runtime already validates its required blocks before scoring
            and can disable this redundant device-to-host synchronization.
            The kernel still masks out-of-range IDs to prevent invalid reads.

    Returns:
        ``[B * next_n, max_ctx_len]`` fp32 logits; -inf past the per-row
        ``context_lens`` boundary or for an unallocated / missing logical block.
    """
    if not q_bf16.is_cuda:
        raise ValueError("qsa_paged_indexer_score requires CUDA tensors")
    if q_bf16.dtype != torch.bfloat16 or q_bf16.dim() != 4:
        raise ValueError("q_bf16 must be a rank-4 bf16 CUDA tensor")
    B, next_n, H, D = q_bf16.shape
    M = B * next_n

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a positive power of two, got {block_n}")

    if weight.shape != (M, H) or weight.dtype != torch.float32:
        raise ValueError(f"weight must be fp32 [{M}, {H}]")
    if kv_pool.dim() != 2 or kv_pool.shape[1] != D or kv_pool.dtype != torch.bfloat16:
        raise ValueError(f"kv_pool must be bf16 [total_slots, {D}]")
    if (
        block_table.dim() != 2
        or block_table.shape[0] != B
        or block_table.dtype != torch.int32
    ):
        raise ValueError(f"block_table must be int32 [{B}, max_blocks]")
    if context_lens.shape != (B, next_n) or context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32 [{B}, {next_n}]")
    tensors = (weight, kv_pool, block_table, context_lens)
    if any(not tensor.is_cuda or tensor.device != q_bf16.device for tensor in tensors):
        raise ValueError(
            "all qsa_paged_indexer_score tensors must share one CUDA device"
        )
    if kv_pool.shape[0] % block_size:
        raise ValueError(
            f"kv_pool slots ({kv_pool.shape[0]}) must be divisible by block_size "
            f"({block_size})"
        )

    pool_blocks = kv_pool.shape[0] // block_size
    if validate_block_table:
        max_physical_block = int(block_table.max().item()) if block_table.numel() else 0
        if max_physical_block >= pool_blocks:
            raise ValueError(
                f"block_table physical block id {max_physical_block} is outside "
                f"kv_pool capacity [0, {pool_blocks})"
            )

    q_bf16 = q_bf16.contiguous()
    weight = weight.contiguous()
    kv_pool = kv_pool.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()

    max_blocks = block_table.shape[1]
    if max_ctx_len is None:
        max_ctx_len = int(context_lens.max())
    logits = torch.empty((M, max_ctx_len), device=q_bf16.device, dtype=torch.float32)
    if M == 0 or max_ctx_len == 0:
        return logits

    # The flattened (batch, token) rows are contiguous, each with H * D values.
    grid = (M, triton.cdiv(max_ctx_len, block_n))
    _qsa_paged_score_kernel[grid](
        q_bf16,
        weight,
        kv_pool,
        block_table,
        context_lens.view(-1),
        logits,
        stride_q_row=H * D,
        max_ctx_len=max_ctx_len,
        max_blocks=max_blocks,
        pool_blocks=pool_blocks,
        next_n=next_n,
        H=H,
        D=D,
        block_size=block_size,
        BN=block_n,
    )
    return logits
