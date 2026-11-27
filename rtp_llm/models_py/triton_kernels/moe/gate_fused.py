"""Fused Triton router-gate epilogues.

Replaces the unfused chain
  scores = score_fn(logits)                     # softmax / sigmoid / sqrtsoftplus / identity
  original = scores
  scores_b = scores + bias                      # optional, + optional visual bias
  [optional group_topk mask]                    # n_group / topk_group
  indices = scores_b.topk(topk)[1]
  weights = original.gather(1, indices)
  weights = weights / (weights.sum(-1) + eps)   # when normalize=True
  weights = weights * route_scale
with a single Triton kernel per token row (with ``do_not_specialize=["N"]`` so
token count changes do not retrigger Triton JIT compilation).

Also exposes backwards-compatible ``fused_sqrtsoftplus_gate`` and
``fused_sqrtsoftplus_hash_gate``.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_SCORE_IDENTITY = 0
_SCORE_SOFTMAX = 1
_SCORE_SIGMOID = 2
_SCORE_SQRTSOFTPLUS = 3

_SCORE_FUNC_TO_MODE = {
    "identity": _SCORE_IDENTITY,
    "softmax": _SCORE_SOFTMAX,
    "sigmoid": _SCORE_SIGMOID,
    "sqrtsoftplus": _SCORE_SQRTSOFTPLUS,
}


@triton.jit(do_not_specialize=["N", "scores_stride_m", "image_mask_stride"])
def _gate_moe_topk_kernel(
    scores_ptr,  # [N, E] fp32/bf16/fp16
    bias_ptr,  # [E] fp32/bf16/fp16 (dummy ptr when HAS_BIAS=False)
    bias_vl_ptr,  # [E] fp32/bf16/fp16 (dummy ptr when HAS_VL_BIAS=False)
    image_mask_ptr,  # [N] bool/uint8 (dummy ptr when HAS_VL_BIAS=False)
    out_idx_ptr,  # [N, K] int64
    out_w_ptr,  # [N, K] fp32
    N,
    scores_stride_m,
    image_mask_stride,
    E: tl.constexpr,
    K: tl.constexpr,
    NORM_EPS: tl.constexpr,
    ROUTE_SCALE: tl.constexpr,
    SCORE_MODE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_VL_BIAS: tl.constexpr,
    NORMALIZE: tl.constexpr,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_NUM_POW2: tl.constexpr,
    BLOCK_E: tl.constexpr,  # >= E, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
):
    """One program per token row supporting general scoring, bias, VL, and group top-K."""
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    offs = tl.arange(0, BLOCK_E)
    mask = offs < E

    # Load row + optional bias. A single non-finite value invalidates the whole
    # router row, matching the eager fallback and keeping bad weights out of MoE.
    s_row = tl.load(
        scores_ptr + pid * scores_stride_m + offs, mask=mask, other=0.0
    ).to(tl.float32)
    score_is_finite = tl.abs(s_row) < float("inf")

    if HAS_BIAS:
        if HAS_VL_BIAS:
            is_img = tl.load(image_mask_ptr + pid * image_mask_stride) != 0
            b_base = tl.where(is_img, bias_vl_ptr, bias_ptr)
            bias_row = tl.load(b_base + offs, mask=mask, other=0.0).to(tl.float32)
        else:
            bias_row = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bias_is_finite = tl.abs(bias_row) < float("inf")
    else:
        bias_row = tl.zeros((BLOCK_E,), dtype=tl.float32)
        bias_is_finite = tl.full((BLOCK_E,), True, dtype=tl.int1)

    bad_value = mask & (~score_is_finite | ~bias_is_finite)
    row_is_finite = tl.sum(bad_value.to(tl.int32), axis=0) == 0
    s_row = tl.where(score_is_finite, s_row, 0.0)
    bias_row = tl.where(bias_is_finite, bias_row, 0.0)

    if SCORE_MODE == 1:
        s_masked = tl.where(mask, s_row, -float("inf"))
        s_max = tl.max(s_masked, axis=0)
        exp_s = tl.where(mask, tl.exp(s_masked - s_max), 0.0)
        s_active = exp_s / (tl.sum(exp_s, axis=0) + NORM_EPS)
    elif SCORE_MODE == 2:
        exp_neg = tl.exp(-tl.abs(s_row))
        inv = 1.0 / (1.0 + exp_neg)
        s_active = tl.where(s_row >= 0.0, inv, exp_neg * inv)
    elif SCORE_MODE == 3:
        THRESH = tl.full([1], 20.0, dtype=tl.float32)
        sp = tl.where(s_row > THRESH, s_row, tl.log(1.0 + tl.exp(s_row)))
        s_active = tl.sqrt(sp)
    else:
        s_active = s_row

    s_biased = s_active + bias_row if HAS_BIAS else s_active
    s_biased = tl.where(mask, s_biased, -float("inf"))

    if N_GROUP > 1 and TOPK_GROUP < N_GROUP:
        g_offs = tl.arange(0, GROUP_NUM_POW2)
        group_scores = tl.full((GROUP_NUM_POW2,), -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            g_start = g * GROUP_SIZE
            g_end = g_start + GROUP_SIZE
            in_g = mask & (offs >= g_start) & (offs < g_end)
            g_vals = tl.where(in_g, s_biased, -float("inf"))
            idx1 = tl.argmax(g_vals, axis=0)
            val1 = tl.sum(tl.where(offs == idx1, g_vals, 0.0), axis=0)
            g_vals2 = tl.where(offs == idx1, -float("inf"), g_vals)
            idx2 = tl.argmax(g_vals2, axis=0)
            val2 = tl.sum(tl.where(offs == idx2, g_vals2, 0.0), axis=0)
            group_scores = tl.where(g_offs == g, val1 + val2, group_scores)

        group_keep = tl.zeros((BLOCK_E,), dtype=tl.int1)
        cur_g = group_scores
        for _ in tl.static_range(TOPK_GROUP):
            best_g = tl.argmax(cur_g, axis=0)
            g_lo = best_g * GROUP_SIZE
            g_hi = g_lo + GROUP_SIZE
            group_keep = group_keep | ((offs >= g_lo) & (offs < g_hi))
            cur_g = tl.where(g_offs == best_g, -float("inf"), cur_g)
        s_biased = tl.where(mask & group_keep, s_biased, -float("inf"))

    cur_biased = s_biased
    for k in tl.static_range(K):
        idx = tl.argmax(cur_biased, axis=0)
        sel = tl.sum(tl.where(offs == idx, s_active, 0.0), axis=0)
        safe_idx = tl.where(row_is_finite, idx, k)
        tl.store(out_idx_ptr + pid * K + k, safe_idx.to(tl.int64))
        tl.store(out_w_ptr + pid * K + k, sel)
        cur_biased = tl.where(offs == idx, -float("inf"), cur_biased)

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w_loaded = tl.load(out_w_ptr + pid * K + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    if NORMALIZE:
        w_sum = tl.sum(w_loaded, axis=0) + NORM_EPS
        w_loaded = w_loaded / w_sum
    w_norm = w_loaded * ROUTE_SCALE
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


def fused_moe_gate(
    scores: torch.Tensor,  # [N, E] fp32/bf16/fp16
    bias: Optional[torch.Tensor],  # [E] or None
    topk: int,
    *,
    score_func: str = "sqrtsoftplus",
    route_scale: float = 1.0,
    norm_eps: float = 1e-12,
    normalize: Optional[bool] = None,
    n_group: int = 0,
    topk_group: int = 0,
    bias_vl: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """General fused Triton MoE router-gate epilogue.

    Supports ``softmax``, ``sigmoid``, ``sqrtsoftplus``, and ``identity`` score
    functions, optional ``bias`` and VL ``bias_vl``/``image_token_mask``,
    optional ``n_group``/``topk_group`` group-topK filtering, arbitrary ``E``
    (including ``E % 4 != 0``), arbitrary 2D row stride (``stride(-1) == 1``),
    and per-row non-finite fallback to uniform ``route_scale / topk``.
    """
    if score_func not in _SCORE_FUNC_TO_MODE:
        raise ValueError(f"Unsupported MoE score_func={score_func!r}")
    if not scores.is_cuda:
        raise RuntimeError(f"fused_moe_gate requires CUDA scores, got {scores.device}")
    if scores.dim() != 2 or scores.stride(-1) != 1:
        raise ValueError(
            f"scores must be 2D with unit last stride, got shape={tuple(scores.shape)} stride={scores.stride()}"
        )
    if scores.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"Unsupported scores dtype={scores.dtype}")

    N, E = int(scores.shape[0]), int(scores.shape[1])
    K = int(topk)
    if not (1 <= K <= 32) or K > E:
        raise ValueError(f"topk must satisfy 1 <= topk <= min(32, E={E}), got {K}")

    if normalize is None:
        normalize = score_func != "softmax"

    has_bias = bias is not None
    if has_bias:
        assert bias is not None
        if not bias.is_cuda or bias.device != scores.device:
            raise ValueError(f"bias must be on {scores.device}, got {bias.device}")
        if bias.dim() != 1 or int(bias.numel()) != E:
            raise ValueError(f"bias must be [E={E}], got {tuple(bias.shape)}")
        if not bias.is_contiguous():
            bias = bias.contiguous()

    if (bias_vl is None) != (image_token_mask is None):
        raise ValueError(
            "bias_vl and image_token_mask must both be set or both be None"
        )
    has_vl_bias = has_bias and bias_vl is not None and image_token_mask is not None
    if has_vl_bias:
        assert bias_vl is not None and image_token_mask is not None
        if not bias_vl.is_cuda or bias_vl.device != scores.device:
            raise ValueError(f"bias_vl must be on {scores.device}, got {bias_vl.device}")
        if bias_vl.dim() != 1 or int(bias_vl.numel()) != E:
            raise ValueError(f"bias_vl must be [E={E}], got {tuple(bias_vl.shape)}")
        if not bias_vl.is_contiguous():
            bias_vl = bias_vl.contiguous()
        if not image_token_mask.is_cuda or image_token_mask.device != scores.device:
            raise ValueError(
                f"image_token_mask must be on {scores.device}, got {image_token_mask.device}"
            )
        if image_token_mask.dim() != 1 or int(image_token_mask.numel()) != N:
            raise ValueError(
                f"image_token_mask must be [N={N}], got {tuple(image_token_mask.shape)}"
            )

    n_group_i = int(n_group or 0)
    topk_group_i = int(topk_group or 0)
    if n_group_i > 1 and topk_group_i < n_group_i:
        if E % n_group_i != 0:
            raise ValueError(
                f"E={E} must be divisible by n_group={n_group_i}"
            )
        group_size = E // n_group_i
        if group_size < 2 or topk_group_i < 1:
            raise ValueError(
                f"group_topk requires group_size >= 2 and topk_group >= 1, "
                f"got group_size={group_size}, topk_group={topk_group_i}"
            )
        group_num_pow2 = triton.next_power_of_2(n_group_i)
    else:
        n_group_i = 0
        topk_group_i = 0
        group_size = 1
        group_num_pow2 = 1

    out_idx = torch.empty((N, K), dtype=torch.int64, device=scores.device)
    out_w = torch.empty((N, K), dtype=torch.float32, device=scores.device)
    if N == 0:
        return out_w, out_idx

    bias_ptr = bias if has_bias else scores
    bias_vl_ptr = bias_vl if has_vl_bias else bias_ptr
    image_mask_ptr = image_token_mask if has_vl_bias else scores
    image_mask_stride = (
        int(image_token_mask.stride(0)) if (has_vl_bias and image_token_mask is not None) else 0
    )

    _gate_moe_topk_kernel[(N,)](
        scores,
        bias_ptr,
        bias_vl_ptr,
        image_mask_ptr,
        out_idx,
        out_w,
        N=N,
        scores_stride_m=int(scores.stride(0)),
        image_mask_stride=image_mask_stride,
        E=E,
        K=K,
        NORM_EPS=float(norm_eps),
        ROUTE_SCALE=float(route_scale),
        SCORE_MODE=_SCORE_FUNC_TO_MODE[score_func],
        HAS_BIAS=has_bias,
        HAS_VL_BIAS=has_vl_bias,
        NORMALIZE=bool(normalize),
        N_GROUP=n_group_i,
        TOPK_GROUP=topk_group_i,
        GROUP_SIZE=group_size,
        GROUP_NUM_POW2=group_num_pow2,
        BLOCK_E=triton.next_power_of_2(E),
        BLOCK_K=triton.next_power_of_2(K),
        num_warps=4,
        num_stages=2,
    )
    return out_w, out_idx


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
    assert bias.numel() == scores.shape[1]
    assert 1 <= int(topk) <= 32, "K must be small for the per-program insertion-sort top-K"
    return fused_moe_gate(
        scores,
        bias,
        topk=int(topk),
        score_func="sqrtsoftplus",
        route_scale=float(route_scale),
        norm_eps=float(norm_eps),
        normalize=True,
    )


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
