"""Triton token-expert alignment for the fused MoE kernel.

Drop-in replacement for ``moe_align_block_size_torch`` with the same output
contract:

    moe_align_block_size_triton(topk_ids, block_size, num_experts)
        -> (sorted_token_ids, expert_ids, num_tokens_post_padded)

Semantics / shape / dtype contract:
- topk_ids: [M, top_k] int32, contiguous, values in [0, num_experts).
  Out-of-range ids are dropped by the general path (and land in a filtered
  bucket on the small-numel path), which is undefined behaviour for the
  consumer exactly like the eager reference; callers must clamp (the Triton
  fused executor already clamps EP-remapped ids).
- sorted_token_ids: [max_padded] int32, pad slots inside [0, total) hold
  ``numel`` (the consumer's invalid-token sentinel), pair slots hold the flat
  (token, slot) index. Intra-expert order is deterministic (ascending pair
  index) on the small-numel path and scheduling-dependent (atomic scatter) on
  the general path; the consumer addresses output rows by pair index, so both
  are numerically equivalent.
- expert_ids: [max_blocks] int32, expert id per block; blocks at or beyond
  ``total`` hold ``num_experts`` (never read by the consumer).
- num_tokens_post_padded: [1] int32 on GPU (no host sync, CUDA-graph safe).
- max_padded / max_blocks follow the eager reference's static worst-case
  capacity formula so the downstream grid is unchanged.

Two paths, selected by ``numel = M * top_k``:
- numel <= SMALL_NUMEL_LIMIT (decode): one single-CTA launch, ported from
  SGLang's ``moe_align_small_numel`` ([NP, NP] pairwise rank / count /
  representative formulation). Above the limit its pairwise tensors spill to
  local memory (NP=256 measured ~230us) and it stops being faster.
- general (prefill): count (reused multi-CTA atomic histogram from
  ep_kernels) + single-CTA meta (cumsum, cursors, expert_ids, sentinel fill)
  + per-pair atomic scatter, all pure Triton.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    _count_tokens_per_expert_kernel,
)

# Largest numel routed to the single-launch pairwise kernel. Its [NP, NP]
# tensors stay in registers at NP=64 but spill at NP=256 (~230us, slower than
# the multi-launch path it replaces).
SMALL_NUMEL_LIMIT = 64

_SENTINEL_FILL_BLOCK = 1024  # vector width of the pad-slot sentinel fill
_SCATTER_BLOCK = 1024  # pairs handled per scatter program
_COUNT_BLOCK = 256  # entries per histogram program (matches ep_kernels usage)
_META_ELEMS = 16384  # target element count of the [BLOCK_B, BLOCK_E] compare


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,  # [numel] int32 flat (token, slot) expert ids
    sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
    expert_ids_ptr,  # [max_num_m_blocks] int32
    num_tokens_post_pad_ptr,  # [1] int32
    num_experts,  # bucket count (num_experts + 1 under the "+1 offset")
    block_size,
    numel,
    NB_written,  # expert_ids buffer length (max blocks)
    NP: tl.constexpr,  # power-of-2 >= numel
    NB: tl.constexpr,  # power-of-2 >= NB_written
):
    """Single-CTA moe_align for tiny batches with many experts.

    Ported from SGLang ``moe_align_small_numel``; everything works on the PAIR
    axis ([NP, NP] pairwise comparisons plus a rank-0 representative per
    bucket). The "+1 offset" convention keeps -1 sentinels in bucket 0; with
    the in-range contract above, bucket ids equal expert ids + 1.

    Intra-bucket order is the stable pair index, matching the eager reference
    bit-for-bit on the used region.
    """
    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
    # Padded lanes get an out-of-range bucket and are masked out everywhere.
    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

    # Pairwise stats: stable rank within the bucket and bucket population.
    same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
    earlier = offs_p[None, :] < offs_p[:, None]
    rank = tl.sum((same & earlier).to(tl.int32), axis=1)  # [NP]
    cnt = tl.sum(same.to(tl.int32), axis=1)  # [NP], own-bucket population
    padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
    is_rep = (rank == 0) & mask_p  # one representative pair per bucket

    # Bucket-ordered exclusive offsets: sum the padded counts of every
    # representative with a strictly smaller bucket id.
    smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
    excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)  # [NP]

    total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    # expert_ids per block: representative r owns blocks
    # [excl[r], excl[r] + padded_cnt[r]); the written id is bucket - 1
    # (bucket 0 = filtered -> -1). Blocks with no owner (>= total, or the
    # tail up to the buffer length) get num_experts - 1 to match the eager
    # reference's searchsorted tail.
    offs_b = tl.arange(0, NB)
    block_start = offs_b * block_size
    in_range = (
        (block_start[:, None] >= excl[None, :])
        & (block_start[:, None] < (excl + padded_cnt)[None, :])
        & is_rep[None, :]
    )
    eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
    any_match = tl.sum(in_range.to(tl.int32), axis=1) > 0
    eid = tl.where(any_match, eid, num_experts - 1)
    tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=offs_b < NB_written)

    # Fill the used region's pad slots with `numel`, then scatter the real
    # pair indices over them. The barrier is required: fill and scatter run on
    # different warps of this CTA, and a scatter store must not be overtaken
    # by a later-warp fill store to the same address.
    n_fill = (total + NP - 1) // NP
    for it in range(n_fill):
        f_offs = it * NP + offs_p
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.zeros([NP], tl.int32) + numel,
            mask=f_offs < total,
        )
    tl.debug_barrier()
    pos = excl + rank
    tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)


@triton.jit
def _moe_align_meta_kernel(
    counts_ptr,  # [E] int32 raw per-expert counts (from the count kernel)
    cursor_ptr,  # [E] int32 out: padded exclusive start per expert; the
    # scatter kernel consumes it as an atomic write cursor
    expert_ids_ptr,  # [max_blocks] int32 out
    num_tokens_post_pad_ptr,  # [1] int32 out
    sorted_token_ids_ptr,  # [max_padded] int32 out (sentinel pre-fill)
    numel,  # num_valid = M * top_k (pad-slot sentinel value)
    block_size,
    max_blocks,
    num_experts,
    BLOCK_E: tl.constexpr,  # power-of-2 >= num_experts
    BLOCK_FILL: tl.constexpr,  # sentinel fill vector width
    BLOCK_B: tl.constexpr,  # expert_ids fill chunk (rows of the compare tile)
):
    """Single-CTA layout pass: padded cumsum, cursors, expert_ids, sentinel fill.

    expert_ids[b] = #{e : cum_incl[e] <= b * block_size}, i.e. a searchsorted
    over the inclusive padded cumsum; blocks beyond the used region resolve to
    num_experts (never read by the consumer). The [BLOCK_B, BLOCK_E] compare
    tile is bounded by the wrapper so register use stays flat for large E.
    """
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts
    cnt = tl.load(counts_ptr + offs_e, mask=mask_e, other=0)
    pad = ((cnt + block_size - 1) // block_size) * block_size
    cum_incl = tl.cumsum(pad)  # inclusive padded cumsum, masked lanes stay 0
    total = tl.sum(pad)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    # Padded exclusive start per expert doubles as the scatter cursor.
    excl = cum_incl - pad
    tl.store(cursor_ptr + offs_e, excl, mask=mask_e)

    # expert_ids fill, chunked over blocks.
    offs_b = tl.arange(0, BLOCK_B)
    for b_start in range(0, max_blocks, BLOCK_B):
        block_idx = b_start + offs_b
        block_start = block_idx * block_size
        counted = (cum_incl[None, :] <= block_start[:, None]) & mask_e[None, :]
        eid = tl.sum(counted.to(tl.int32), axis=1)
        tl.store(
            expert_ids_ptr + block_idx, eid, mask=block_idx < max_blocks
        )

    # Sentinel-fill the used region so pad slots hold `numel`; real pair
    # indices are scattered over it by the next launch (stream-ordered).
    offs_f = tl.arange(0, BLOCK_FILL)
    for f_start in range(0, total, BLOCK_FILL):
        f_offs = f_start + offs_f
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.zeros([BLOCK_FILL], tl.int32) + numel,
            mask=f_offs < total,
        )


@triton.jit
def _moe_align_scatter_kernel(
    ids_ptr,  # [numel] int32 flat (token, slot) expert ids
    cursor_ptr,  # [E] int32, initialized to padded exclusive starts
    sorted_token_ids_ptr,  # [max_padded] int32
    numel,
    num_experts,
    max_padded,
    BLOCK_P: tl.constexpr,
):
    """Per-pair atomic scatter: pair p claims the next slot of its expert.

    Intra-expert order is scheduling-dependent (atomic); every pair lands in
    its expert's padded segment exactly once, which is all the consumer needs
    (output rows are addressed by pair index).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    mask = offs < numel
    e = tl.load(ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (e >= 0) & (e < num_experts)
    e_safe = tl.where(valid, e, 0)
    dest = tl.atomic_add(cursor_ptr + e_safe, 1, mask=valid)
    tl.store(
        sorted_token_ids_ptr + dest,
        offs.to(tl.int32),
        mask=valid & (dest < max_padded),
    )


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,  # [M, top_k]
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align tokens to experts in blocks of block_size (see module docstring)."""
    assert topk_ids.dim() == 2
    assert topk_ids.dtype == torch.int32
    assert topk_ids.is_contiguous()
    M, top_k = topk_ids.shape
    num_valid = M * top_k

    # Same static worst-case capacity as the eager reference so the consumer
    # grid and buffer shapes are unchanged.
    max_active_experts = min(num_valid, num_experts)
    max_blocks = max_active_experts + (num_valid - max_active_experts) // block_size
    max_padded = max_blocks * block_size

    device = topk_ids.device
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)
    flat_ids = topk_ids.view(-1)

    if num_valid <= SMALL_NUMEL_LIMIT:
        _moe_align_small_numel_kernel[(1,)](
            flat_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            num_experts + 1,  # bucket count under the "+1 offset"
            block_size,
            num_valid,
            max_blocks,
            NP=triton.next_power_of_2(max(num_valid, 2)),
            NB=triton.next_power_of_2(max(max_blocks, 2)),
            num_warps=4,
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    _count_tokens_per_expert_kernel[(triton.cdiv(num_valid, _COUNT_BLOCK),)](
        topk_ids,
        topk_ids.stride(0),
        topk_ids.stride(1),
        counts,
        num_valid,
        topk_num=top_k,
        num_experts=num_experts,
        BLOCK_SIZE=_COUNT_BLOCK,
    )

    cursor = torch.empty(num_experts, dtype=torch.int32, device=device)
    block_e = triton.next_power_of_2(num_experts)
    _moe_align_meta_kernel[(1,)](
        counts,
        cursor,
        expert_ids,
        num_tokens_post_padded,
        sorted_token_ids,
        num_valid,
        block_size,
        max_blocks,
        num_experts,
        BLOCK_E=block_e,
        BLOCK_FILL=_SENTINEL_FILL_BLOCK,
        BLOCK_B=max(16, _META_ELEMS // block_e),
        num_warps=8,
    )

    _moe_align_scatter_kernel[(triton.cdiv(num_valid, _SCATTER_BLOCK),)](
        flat_ids,
        cursor,
        sorted_token_ids,
        num_valid,
        num_experts,
        max_padded,
        BLOCK_P=_SCATTER_BLOCK,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded
