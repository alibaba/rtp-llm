"""Fused MoE token-expert alignment Triton kernel (one launch per call).

Opt r9 ⑥ tail-clear lane, item 1: the decode-window residual #1 kernel was
``radixSortKVInPlace`` — 70.2 ms / 3660x over one window (3660 = 30 MoE
layers x ~122 canvas forwards). The call chain is NOT the accept mask (the
r5 delivery attribution "radixSort 68.1 = accept-mask" mis-read the counts):
``DiffusionGemmaExperts._forward_fused`` -> ``moe_align_block_size_torch``
-> ``flat_ids.argsort(stable=True)`` runs once per MoE layer per forward,
and each call additionally fires ~12 small kernels (zeros/ones/full fills,
two cumsums, the stable radix sort itself, three index/gather chains, an
index_put scatter, the int32 casts, a searchsorted) — the whole 63.4 ms
index/scatter cluster, the 17.9 ms int FillFunctor cluster and part of the
copy cluster ride along with it.

This kernel replaces the entire chain with ONE launch that reproduces the
torch reference bit-exactly (all-integer semantics, no tolerance):

  * counts[e]      = #{routes with id == e}                      (one-hot)
  * padded[e]      = ceil(counts[e] / block) * block
  * offsets        = exclusive cumsum of padded                   (1D cumsum)
  * post_padded    = sum(padded)                                  (= offsets[E])
  * stable order   = per-64-route chunk ``tl.sort`` over the packed
                     (expert_id << 32 | route_index) int64 key — chunk-local
                     stable sort + a running per-expert count reproduces
                     torch's global stable argsort exactly (chunks are
                     visited in route-index order, so "earlier chunk" and
                     "smaller route index" are the same set).
  * dest           = offsets[expert] + rank-within-expert; sorted_token_ids
                     is pre-filled with the num_valid sentinel and the
                     routes are scattered into it.
  * expert_ids[b]  = #{e : (offsets[e] + padded[e]) // block <= b}
                     — torch's ``searchsorted(block_boundaries, b, right)``.

Single-program (grid=(1,)) by design: the decode/canvas shapes it targets
are small (num_valid = 256*8 = 2048 routes, E=128 experts) and the torch
chain it replaces is launch-overhead bound, not work bound. Large prefill
shapes stay on the torch chain (caller-side threshold: num_valid > 8192 —
there the multi-kernel torch chain is throughput-bound and a single CTA
would be slower); CUDA-graph replay is shape-static so the branch is stable
per capture.

The kernel requires every route id in [0, num_experts) — the same contract
as ``moe_align_block_size_torch`` (its expert_start/offsets indexing is
undefined for out-of-range ids).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

# Route-chunk width. Bounded by the [BLOCK_N, BLOCK_N] rank compare and the
# [BLOCK_N, BLOCK_E] one-hot matrices staying register-resident.
_BLOCK_N = 64
_MAX_NUM_VALID = 8192
_MAX_EXPERTS = 512


@triton.jit
def _moe_align_kernel(
    ids_ptr,  # [num_valid] route expert ids (int32 or int64, row-major flat)
    sorted_token_ids_ptr,  # [max_padded] int32 out (sentinel num_valid)
    expert_ids_ptr,  # [max_blocks] int32 out
    post_padded_ptr,  # [1] int32 out
    num_valid,
    block_size,
    num_experts,
    max_blocks,
    max_padded,
    BLOCK_E: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    e_range = tl.arange(0, BLOCK_E)
    e_mask = e_range < num_experts

    # ---- pass 1: per-expert route counts (masked lanes carry the sentinel
    # id num_experts, which never matches a valid e-range lane).
    counts = tl.zeros([BLOCK_E], tl.int32)
    for off in range(0, num_valid, BLOCK_N):
        idx = off + tl.arange(0, BLOCK_N)
        lane = idx < num_valid
        ids = tl.load(ids_ptr + idx, mask=lane, other=num_experts)
        onehot = ids[:, None] == e_range[None, :]
        counts += tl.sum(onehot.to(tl.int32), axis=0)

    # ---- expert offsets: exclusive cumsum of the block-padded counts.
    padded = ((counts + block_size - 1) // block_size) * block_size
    padded = tl.where(e_mask, padded, 0)
    offs_incl = tl.cumsum(padded, axis=0)
    offs_excl = offs_incl - padded
    tl.store(post_padded_ptr, tl.sum(padded, axis=0))

    # ---- sentinel fill: every slot starts as num_valid (the fused GEMM's
    # padding marker); real routes overwrite their own slots below.
    for off in range(0, max_padded, BLOCK_F):
        fidx = off + tl.arange(0, BLOCK_F)
        tl.store(
            sorted_token_ids_ptr + fidx,
            num_valid,
            mask=fidx < max_padded,
        )

    # ---- pass 2: stable ranks + scatter. One chunk at a time (ascending
    # route index), sort the packed (expert, index) key so each expert's
    # routes form a run in ascending original index — the rank of a route
    # within its expert = its position inside the run, which equals
    # (sorted position) - (count of strictly-smaller ids in the chunk);
    # adding the running per-expert count from earlier chunks gives the
    # global stable rank, bit-equal to argsort(stable=True).
    run_counts = tl.zeros([BLOCK_E], tl.int32)
    for off in range(0, num_valid, BLOCK_N):
        idx = off + tl.arange(0, BLOCK_N)
        lane = idx < num_valid
        ids = tl.load(ids_ptr + idx, mask=lane, other=num_experts)
        packed = (ids.to(tl.int64) << 32) | idx.to(tl.int64)
        skeys = tl.sort(packed)
        sids = (skeys >> 32).to(tl.int32)
        sorig = (skeys & 0xFFFFFFFF).to(tl.int32)
        # count of strictly-smaller expert ids per sorted slot = the run
        # start position (the sorted vector is ascending).
        smaller = tl.sum(
            (sids[None, :] < sids[:, None]).to(tl.int32), axis=1
        )
        rank = tl.arange(0, BLOCK_N) - smaller
        onehot = sids[:, None] == e_range[None, :]
        # dest = offsets[expert] + (routes of this expert seen in earlier
        # chunks) + rank-within-chunk — selected per element through the
        # one-hot (exactly one true column per valid row).
        base = offs_excl[None, :] + run_counts[None, :]
        dest = tl.sum(tl.where(onehot, base, 0), axis=1) + rank
        tl.store(
            sorted_token_ids_ptr + dest,
            sorig,
            mask=tl.arange(0, BLOCK_N) < (num_valid - off),
        )
        run_counts += tl.sum(onehot.to(tl.int32), axis=0)

    # ---- pass 3: expert id per block (searchsorted-right semantics).
    # Invalid e-range lanes get a boundary of +inf so they never count.
    boundaries = tl.where(e_mask, offs_incl, 2147483647) // block_size
    for off in range(0, max_blocks, BLOCK_B):
        bidx = off + tl.arange(0, BLOCK_B)
        bmask = bidx < max_blocks
        cnt = tl.sum(
            (boundaries[None, :] <= bidx[:, None]).to(tl.int32), axis=1
        )
        tl.store(expert_ids_ptr + bidx, cnt, mask=bmask)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA-graph-safe single-launch replacement for
    ``moe_align_block_size_torch`` (same contract, same outputs, bit-exact).

    Returns (sorted_token_ids int32 [max_padded], expert_ids int32
    [max_blocks], num_tokens_post_padded int32 [1]); every route id must lie
    in [0, num_experts). Shapes above ``num_valid > 8192`` or
    ``num_experts > 512`` raise ValueError so the caller keeps the torch
    chain there (throughput regime, single CTA would be slower).
    """
    if topk_ids.dim() != 2:
        raise ValueError("moe_align_block_size_triton needs a 2-D topk_ids")
    if not topk_ids.is_cuda:
        raise ValueError("moe_align_block_size_triton requires CUDA ids")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("moe_align_block_size_triton needs int32/int64 ids")
    if int(block_size) < 1 or int(num_experts) < 1:
        raise ValueError("block_size and num_experts must be positive")
    M, top_k = topk_ids.shape
    num_valid = M * top_k
    if num_experts > _MAX_EXPERTS:
        raise ValueError("moe_align_block_size_triton supports up to 512 experts")
    if num_valid > _MAX_NUM_VALID:
        raise ValueError(
            "moe_align_block_size_triton supports up to 8192 routes "
            "(larger shapes stay on the torch chain)"
        )

    max_active_experts = min(num_valid, num_experts)
    max_blocks = max_active_experts + (num_valid - max_active_experts) // block_size
    max_padded = max_blocks * block_size

    flat = topk_ids.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()

    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=flat.device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=flat.device)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=flat.device)

    block_e = triton.next_power_of_2(num_experts)
    block_b = max(16, triton.next_power_of_2(max(max_blocks, 1)))
    block_b = min(block_b, 256)
    _moe_align_kernel[(1,)](
        flat,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid,
        int(block_size),
        int(num_experts),
        max_blocks,
        max_padded,
        BLOCK_E=block_e,
        BLOCK_N=_BLOCK_N,
        BLOCK_B=block_b,
        BLOCK_F=1024,
        num_warps=4,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded
