"""Fused EB accept-mask Triton kernel (one launch per call).

Opt r5 ⑥-clearing lane: the DiffusionGemma proposer's entropy-bounded
acceptance ran the reference chain

    sorted_entropy, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumulative = torch.cumsum(sorted_entropy, dim=-1)
    sorted_mask = cumulative - sorted_entropy <= entropy_bound
    mask = torch.scatter(torch.zeros_like(sorted_mask), -1, sorted_idx, sorted_mask)

on [B, canvas] tensors — the sort alone surfaced as 3540 radixSort
invocations / 68.1 ms over one decode window, and the chain is six-plus
tiny kernel launches per denoising step (sort + cumsum + sub + compare +
zeros + scatter) whose cost is launch overhead, not work.

Semantics first (read before replacing the sort): the sort is NOT mere mask
compaction — it implements the reference acceptance RULE, "accept tokens in
ascending-entropy order while the sum of all strictly-smaller entropies
stays within the entropy budget" (transformers generation_diffusion_gemma).
A position-order prefix-sum compact would change which tokens get accepted,
so this kernel preserves the sort semantics exactly instead of removing it:
one program per row bitonic-sorts the (entropy, index) pairs in registers
(``tl.sort`` over an order-preserving packed int64 key), forms the exclusive
cumulative sum, compares against the budget and scatter-stores the accept
bits — the whole chain in one launch.

Key packing: fp32 entropy bits ``u`` are mapped to an order-preserving
int64 (negatives flipped below zero, positives offset above, so unsigned
key order == float order), then ``key << 11 | index`` — the index rides the
low bits as a stable ascending tie-break (torch's CUDA radix sort is
likewise stable, so exact-tie entropy rows agree bit-for-bit; ties are
bit-equal fp32 entropies, which real per-token rows essentially never
produce). Padded lanes (canvas < BLOCK) load +inf and sort to the tail,
outside the store mask.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

_MAX_BLOCK_C = 4096
_IDX_BITS = 11  # key low bits for the token index; BLOCK_C <= 2048


@triton.jit
def _accept_mask_kernel(
    entropy_ptr,
    out_ptr,
    canvas,
    bound,
    stride_row,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(axis=0).to(tl.int64)
    base = entropy_ptr + row * stride_row
    offs = tl.arange(0, BLOCK_C)
    lane = offs < canvas
    e = tl.load(base + offs, mask=lane, other=float("inf")).to(tl.float32)

    # order-preserving fp32 -> int64 map: negative floats flip (~), positive
    # floats offset by 2^31, so ascending key order == ascending float order.
    u = e.to(tl.int32, bitcast=True).to(tl.int64)
    key = tl.where(u < 0, u ^ -1, u + 2147483648)
    # pack the token index into the low bits as the stable tie-break.
    packed = (key << 11) | offs.to(tl.int64)

    sorted_keys = tl.sort(packed)

    # unpack the sorted values and indices.
    sorted_idx = (sorted_keys & 2047).to(tl.int32)
    sorted_key = sorted_keys >> 11
    sorted_u = tl.where(sorted_key >= 2147483648, sorted_key - 2147483648, sorted_key ^ -1)
    sorted_e = sorted_u.to(tl.int32).to(tl.float32, bitcast=True)

    # exclusive cumulative sum of the ascending entropies.
    cum = tl.cumsum(sorted_e, axis=0)
    mask = (cum - sorted_e) <= bound

    # scatter the accept bits back to the original token positions.
    tl.store(out_ptr + row * canvas + sorted_idx, mask, mask=sorted_idx < canvas)


def accept_mask(entropy: torch.Tensor, bound: float) -> torch.Tensor:
    """EB accept mask from per-token entropies, one kernel launch.

    For each row of ``entropy`` ([.., canvas] fp32): accept token t iff the
    sum of the entropies strictly smaller than ``entropy[t]`` (plus the
    tied tokens ranked before it) is <= ``bound`` — the reference
    sort-ascending / exclusive-cumsum / scatter chain, bit-faithful modulo
    torch.sort's own stability on exact ties. Requires CUDA fp32 with a
    last dim <= 2048 (the proposer's canvas is 256); the caller owns the
    gate and the torch fallback for CPU / oversize rows.
    """
    if entropy.dim() < 1:
        raise ValueError("accept_mask needs at least one dimension")
    if not entropy.is_cuda or entropy.dtype != torch.float32:
        raise ValueError("accept_mask requires CUDA fp32 entropy")
    canvas = int(entropy.shape[-1])
    if canvas < 1:
        raise ValueError("accept_mask needs a non-empty last dim")
    if canvas > 2048:
        raise ValueError("accept_mask supports a last dim of up to 2048")
    rows = entropy.numel() // canvas
    out_shape = entropy.shape
    if rows == 0:
        return torch.empty(out_shape, dtype=torch.bool, device=entropy.device)
    flat = entropy.reshape(rows, canvas)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = torch.empty(rows, canvas, dtype=torch.bool, device=flat.device)
    block_c = triton.next_power_of_2(canvas)
    _accept_mask_kernel[(rows,)](
        flat,
        out,
        canvas,
        float(bound),
        flat.stride(0),
        BLOCK_C=block_c,
        num_warps=max(1, min(8, block_c // 64)),
    )
    return out.view(out_shape)
