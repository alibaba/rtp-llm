"""Fused router-gate epilogue for V4 Gate (P2 of plan_0427.md).

Replaces the chain
  scores = F.softplus(scores).sqrt()       # 2 elementwise launches
  original = scores
  scores = scores + bias                    # 1 elementwise
  indices = scores.topk(topk)[1]            # mbtopk: ~3 kernel launches
  weights = original.gather(1, indices)     # 1 vectorized_gather
  weights = weights / (weights.sum(-1) + eps)  # 1 reduce + 1 div
  weights = weights * route_scale           # 1 mul
with a single Triton kernel that does all of it per token.

Shapes (V4-Flash): scores [N, 256] fp32, bias [256] fp32, output
indices [N, 6] long, weights [N, 6] fp32.

Currently supports score_func='sqrtsoftplus' (V4 default).  For other
score functions, fall back to the eager path.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _v4_gate_sqrtsoftplus_topk_kernel(
    scores_ptr,  # [N, E] fp32
    bias_ptr,  # [E] fp32
    out_idx_ptr,  # [N, K] int64
    out_w_ptr,  # [N, K] fp32
    N,
    E: tl.constexpr,
    K: tl.constexpr,
    NORM_EPS: tl.constexpr,  # 1e-12
    ROUTE_SCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,  # >= E, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
):
    """One program per token row.

    For each token:
      1. Load scores row [E].
      2. Compute s = sqrt(softplus(scores)) — fp32 throughout.
      3. Find top-K of (s + bias), keeping s un-biased for the weight gather.
      4. Normalize weights by sum and scale.
    """
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    offs = tl.arange(0, BLOCK_E)
    mask = offs < E

    # Load row + bias.
    s_row = tl.load(scores_ptr + pid * E + offs, mask=mask, other=-float("inf")).to(
        tl.float32
    )
    bias_row = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # softplus(x) = log(1 + exp(x)); numerically stable for x>20: just x.
    THRESH = tl.full([1], 20.0, dtype=tl.float32)
    sp = tl.where(s_row > THRESH, s_row, tl.log(1.0 + tl.exp(s_row)))
    s_active = tl.sqrt(sp)  # original (un-biased) score, used for weights
    s_biased = s_active + bias_row  # used for ranking

    # Mask out padding lanes from being chosen.
    s_biased = tl.where(mask, s_biased, -float("inf"))

    # Insertion-sort top-K by repeatedly extracting the argmax.  K is small (=6
    # for V4), and the per-step argmax of [E] is cheap.  We blank out chosen
    # positions with -inf so subsequent argmaxes ignore them.
    # NOTE: tl.argmax over a 1-D vector returns int32; promote to int64 on store.
    cur_biased = s_biased
    for k in tl.static_range(K):
        idx = tl.argmax(cur_biased, axis=0)  # int32 scalar
        # Gather the un-biased score at this index for the weight.
        sel = tl.sum(tl.where(offs == idx, s_active, 0.0), axis=0)
        tl.store(out_idx_ptr + pid * K + k, idx.to(tl.int64))
        tl.store(out_w_ptr + pid * K + k, sel)
        # Blank out the chosen position so the next argmax ignores it.
        cur_biased = tl.where(offs == idx, -float("inf"), cur_biased)

    # Pass 2: load the K weights back, normalize, scale, store.
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w_loaded = tl.load(out_w_ptr + pid * K + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    s = tl.sum(w_loaded, axis=0) + NORM_EPS
    w_norm = w_loaded / s * ROUTE_SCALE
    tl.store(out_w_ptr + pid * K + k_offs, w_norm, mask=k_mask)


def fused_sqrtsoftplus_gate(
    scores: torch.Tensor,  # [N, E] fp32 contiguous
    bias: torch.Tensor,  # [E] fp32 contiguous
    topk: int,
    route_scale: float = 1.0,
    norm_eps: float = 1e-12,
):
    """Fused replacement for the Gate epilogue when score_func='sqrtsoftplus'.

    Returns (weights [N, topk] fp32, indices [N, topk] int64) — same shape and
    semantics as the eager-mode

        scores = F.softplus(scores).sqrt()
        scores_b = scores + bias
        indices = scores_b.topk(topk)[1]
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-12) * route_scale
    """
    assert (
        scores.dtype == torch.float32 and scores.dim() == 2 and scores.is_contiguous()
    )
    assert bias.dtype == torch.float32 and bias.dim() == 1 and bias.is_contiguous()
    N, E = scores.shape
    assert bias.numel() == E
    K = int(topk)
    assert 1 <= K <= 32, "K must be small for the per-program insertion-sort top-K"
    BLOCK_E = triton.next_power_of_2(E)
    BLOCK_K = triton.next_power_of_2(K)

    out_idx = torch.empty((N, K), dtype=torch.int64, device=scores.device)
    out_w = torch.empty((N, K), dtype=torch.float32, device=scores.device)
    if N == 0:
        return out_w, out_idx

    grid = (N,)
    _v4_gate_sqrtsoftplus_topk_kernel[grid](
        scores,
        bias,
        out_idx,
        out_w,
        N=N,
        E=E,
        K=K,
        NORM_EPS=norm_eps,
        ROUTE_SCALE=route_scale,
        BLOCK_E=BLOCK_E,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )
    return out_w, out_idx
