"""Fused router-gate epilogue for sqrt-softplus routing.

Replaces the chain
  scores = F.softplus(scores).sqrt()       # 2 elementwise launches
  original = scores
  scores = scores + bias                    # 1 elementwise
  indices = scores.topk(topk)[1]            # mbtopk: ~3 kernel launches
  weights = original.gather(1, indices)     # 1 vectorized_gather
  weights = weights / (weights.sum(-1) + eps)  # 1 reduce + 1 div
  weights = weights * route_scale           # 1 mul
with a single Triton kernel that does all of it per token.

Shapes: scores [N, E] fp32, bias [E] fp32, output indices [N, K] long,
weights [N, K] fp32.

Currently supports score_func='sqrtsoftplus'. For other
score functions, fall back to the eager path.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _gate_sqrtsoftplus_topk_kernel(
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

    # Load row + bias. A single non-finite value invalidates the whole router
    # row, matching the eager fallback and keeping bad weights out of MegaMoE.
    s_row = tl.load(scores_ptr + pid * E + offs, mask=mask, other=0.0).to(tl.float32)
    bias_row = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    score_is_finite = tl.abs(s_row) < float("inf")
    bias_is_finite = tl.abs(bias_row) < float("inf")
    bad_value = mask & (~score_is_finite | ~bias_is_finite)
    row_is_finite = tl.sum(bad_value.to(tl.int32), axis=0) == 0
    s_row = tl.where(score_is_finite, s_row, 0.0)
    bias_row = tl.where(bias_is_finite, bias_row, 0.0)

    # softplus(x) = log(1 + exp(x)); numerically stable for x>20: just x.
    THRESH = tl.full([1], 20.0, dtype=tl.float32)
    sp = tl.where(s_row > THRESH, s_row, tl.log(1.0 + tl.exp(s_row)))
    s_active = tl.sqrt(sp)  # original (un-biased) score, used for weights
    s_biased = s_active + bias_row  # used for ranking

    # Mask out padding lanes from being chosen.
    s_biased = tl.where(mask, s_biased, -float("inf"))

    # Insertion-sort top-K by repeatedly extracting the argmax.  K is small (=6
    # in supported model configurations), and the per-step argmax of [E] is
    # cheap. We blank out chosen
    # positions with -inf so subsequent argmaxes ignore them.
    # NOTE: tl.argmax over a 1-D vector returns int32; promote to int64 on store.
    cur_biased = s_biased
    for k in tl.static_range(K):
        idx = tl.argmax(cur_biased, axis=0)  # int32 scalar
        # Gather the un-biased score at this index for the weight.
        sel = tl.sum(tl.where(offs == idx, s_active, 0.0), axis=0)
        safe_idx = tl.where(row_is_finite, idx, k)
        tl.store(out_idx_ptr + pid * K + k, safe_idx.to(tl.int64))
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
    w_norm = tl.where(row_is_finite, w_norm, ROUTE_SCALE / K)
    tl.store(out_w_ptr + pid * K + k_offs, w_norm, mask=k_mask)


@triton.jit(do_not_specialize=["N"])
def _gate_sqrtsoftplus_hash_kernel(
    scores_ptr,  # [N, E] bf16
    input_ids_ptr,  # [N]
    tid2eid_ptr,  # [vocab, K]
    out_idx_ptr,  # [N, K] int64
    out_w_ptr,  # [N, K] fp32
    N,
    E: tl.constexpr,
    K: tl.constexpr,
    input_ids_stride: tl.constexpr,
    tid2eid_stride_m: tl.constexpr,
    tid2eid_stride_k: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Match the former MegaMoE hash gate-pack routing numerics."""
    row = tl.program_id(0).to(tl.int64)
    if row >= N:
        return

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    token_id = tl.load(input_ids_ptr + row * input_ids_stride).to(tl.int64)
    idx = tl.load(
        tid2eid_ptr + token_id * tid2eid_stride_m + k_offs * tid2eid_stride_k,
        mask=k_mask,
        other=0,
    ).to(tl.int64)
    selected = tl.load(
        scores_ptr + row * E + idx,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    selected_is_finite = tl.abs(selected) < float("inf")
    row_is_finite = tl.sum((k_mask & ~selected_is_finite).to(tl.int32), axis=0) == 0
    selected = tl.where(selected_is_finite, selected, 0.0)

    threshold = tl.full([1], 20.0, dtype=tl.float32)
    softplus = tl.where(
        selected > threshold,
        selected,
        tl.log(1.0 + tl.exp(selected)),
    )
    weights = tl.sqrt(softplus)
    denom = tl.sum(tl.where(k_mask, weights, 0.0), axis=0) + NORM_EPS
    weights = weights / denom * ROUTE_SCALE
    weights = tl.where(row_is_finite, weights, ROUTE_SCALE / K)
    tl.store(out_idx_ptr + row * K + k_offs, idx, mask=k_mask)
    tl.store(out_w_ptr + row * K + k_offs, weights, mask=k_mask)


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
    _gate_sqrtsoftplus_topk_kernel[grid](
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


def fused_sqrtsoftplus_hash_gate(
    scores: torch.Tensor,  # [N, E] bf16 contiguous
    input_ids: torch.Tensor,  # [N]
    tid2eid: torch.Tensor,  # [vocab, K]
    route_scale: float = 1.0,
    norm_eps: float = 1e-12,
):
    """Route hash layers with the numerics used by MegaMoE gate-pack.

    Keeping the BF16 router scores and evaluating sqrt-softplus in Triton is
    intentional: replacing this path with the eager PyTorch epilogue causes
    small per-layer drift that can change greedy decoding after many layers.
    """
    assert scores.dtype == torch.bfloat16 and scores.dim() == 2
    assert scores.is_contiguous()
    assert input_ids.dim() == 1 and input_ids.numel() == scores.size(0)
    assert tid2eid.dim() == 2
    N, E = scores.shape
    K = int(tid2eid.size(1))
    assert 1 <= K <= 32

    out_idx = torch.empty((N, K), dtype=torch.int64, device=scores.device)
    out_w = torch.empty((N, K), dtype=torch.float32, device=scores.device)
    if N == 0:
        return out_w, out_idx

    _gate_sqrtsoftplus_hash_kernel[(N,)](
        scores,
        input_ids,
        tid2eid,
        out_idx,
        out_w,
        N=N,
        E=E,
        K=K,
        input_ids_stride=input_ids.stride(0),
        tid2eid_stride_m=tid2eid.stride(0),
        tid2eid_stride_k=tid2eid.stride(1),
        NORM_EPS=norm_eps,
        ROUTE_SCALE=route_scale,
        BLOCK_K=triton.next_power_of_2(K),
        num_warps=4,
    )
    return out_w, out_idx
