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

The same kernel also carries the SM90 MegaMoE input staging
(``fused_sqrtsoftplus_gate_pack``): a 2-D grid ``(N, hidden/128)`` lets every
program quantize one 128-wide K-slice of its row into the collective buffer's
FP8 activations + FP32 K128 scales, while the ``pid(1) == 0`` programs run the
router body.  That removes one launch per MoE layer versus routing and packing
in two kernels.  The router body is byte-identical in both modes; the 1-D
callers (prefill, and any non-packed call) keep the original single-column
grid and the ``HAS_PACK=False`` specialization.
"""

import torch
import triton
import triton.language as tl

# SM90 MegaMoE activation staging protocol: FP8 E4M3 with FP32 per-K128
# scales, matching DeepGEMM ``per_token_cast_to_fp8(use_ue8m0=False, gran_k=128)``.
_PACK_GRAN_K = 128
_PACK_FP8_MAX = 448.0
_PACK_AMAX_FLOOR = 1e-4


@triton.jit(do_not_specialize=["N"])
def _v4_gate_sqrtsoftplus_topk_kernel(
    scores_ptr,  # [N, E] fp32 or bf16
    bias_ptr,  # [E] fp32
    out_idx_ptr,  # [N, K] int64
    out_w_ptr,  # [N, K] fp32
    x_ptr,  # [N, H] bf16 — read only when HAS_PACK
    out_fp8_ptr,  # [N, H] fp8 — written only when HAS_PACK
    out_sf_ptr,  # [N, H / 128] fp32 — written only when HAS_PACK
    N,
    E: tl.constexpr,
    K: tl.constexpr,
    NORM_EPS: tl.constexpr,  # 1e-12
    ROUTE_SCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,  # >= E, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    HAS_PACK: tl.constexpr,
    HIDDEN: tl.constexpr,  # 0 when HAS_PACK is False
    x_stride_m: tl.constexpr,
    out_fp8_stride_m: tl.constexpr,
    sf_stride_m: tl.constexpr,
    out_idx_stride_m: tl.constexpr,
    out_w_stride_m: tl.constexpr,
    AMAX_FLOOR: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One program per (token, K128 tile); the router body runs on tile 0.

    For each token:
      0. (HAS_PACK) Quantize this row's 128-wide tile to FP8 with an FP32
         K128 scale.  Skipped entirely for the router-only launch.
      1. Load scores row [E].
      2. Compute s = sqrt(softplus(scores)) — fp32 throughout.
      3. Find top-K of (s + bias), keeping s un-biased for the weight gather.
      4. Normalize weights by sum and scale.
    """
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    if HAS_PACK:
        # Tile (pid, pid_k) owns columns [pid_k * 128, pid_k * 128 + 128) of
        # row ``pid``.  Formula matches the standalone SM90 packer bit for bit:
        # amax = max(max|x|, 1e-4); scale = amax / 448; q = x * (1 / scale).
        # 128 is `_PACK_GRAN_K`; Triton kernels cannot read module globals.
        pid_k = tl.program_id(1)
        cols = pid_k * 128 + tl.arange(0, 128)
        col_mask = cols < HIDDEN
        x_tile = tl.load(
            x_ptr + pid * x_stride_m + cols, mask=col_mask, other=0.0
        ).to(tl.float32)
        # Match torch.nan_to_num(..., nan=0, posinf=0, neginf=0).
        x_tile = tl.where(tl.abs(x_tile) < float("inf"), x_tile, 0.0)
        amax = tl.maximum(tl.max(tl.abs(x_tile), axis=0), AMAX_FLOOR)
        scale = amax / FP8_MAX
        q = (x_tile * (1.0 / scale)).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + pid * out_fp8_stride_m + cols, q, mask=col_mask)
        tl.store(out_sf_ptr + pid * sf_stride_m + pid_k, scale)
        # Router staging does not depend on the K128 tile, so exactly one
        # program per row runs it and the rest retire here.
        if pid_k != 0:
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
    # for V4), and the per-step argmax of [E] is cheap.  We blank out chosen
    # positions with -inf so subsequent argmaxes ignore them.
    # NOTE: tl.argmax over a 1-D vector returns int32; promote to int64 on store.
    cur_biased = s_biased
    for k in tl.static_range(K):
        idx = tl.argmax(cur_biased, axis=0)  # int32 scalar
        # Gather the un-biased score at this index for the weight.
        sel = tl.sum(tl.where(offs == idx, s_active, 0.0), axis=0)
        safe_idx = tl.where(row_is_finite, idx, k)
        tl.store(out_idx_ptr + pid * out_idx_stride_m + k, safe_idx.to(tl.int64))
        tl.store(out_w_ptr + pid * out_w_stride_m + k, sel)
        # Blank out the chosen position so the next argmax ignores it.
        cur_biased = tl.where(offs == idx, -float("inf"), cur_biased)

    # Pass 2: load the K weights back, normalize, scale, store.
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w_loaded = tl.load(
        out_w_ptr + pid * out_w_stride_m + k_offs, mask=k_mask, other=0.0
    ).to(tl.float32)
    s = tl.sum(w_loaded, axis=0) + NORM_EPS
    w_norm = w_loaded / s * ROUTE_SCALE
    w_norm = tl.where(row_is_finite, w_norm, ROUTE_SCALE / K)
    tl.store(out_w_ptr + pid * out_w_stride_m + k_offs, w_norm, mask=k_mask)


def _launch(
    scores: torch.Tensor,
    bias: torch.Tensor,
    out_idx: torch.Tensor,
    out_w: torch.Tensor,
    x: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    *,
    grid: tuple,
    has_pack: bool,
    hidden: int,
    route_scale: float,
    norm_eps: float,
) -> None:
    """Shared launch path for the router-only and router+pack specializations."""
    n, experts = scores.shape
    topk = out_idx.shape[1]
    _v4_gate_sqrtsoftplus_topk_kernel[grid](
        scores,
        bias,
        out_idx,
        out_w,
        x,
        out_fp8,
        out_sf,
        N=n,
        E=experts,
        K=topk,
        NORM_EPS=norm_eps,
        ROUTE_SCALE=route_scale,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(topk),
        HAS_PACK=has_pack,
        HIDDEN=hidden,
        x_stride_m=x.stride(0),
        out_fp8_stride_m=out_fp8.stride(0),
        sf_stride_m=out_sf.stride(0),
        out_idx_stride_m=out_idx.stride(0),
        out_w_stride_m=out_w.stride(0),
        AMAX_FLOOR=_PACK_AMAX_FLOOR,
        FP8_MAX=_PACK_FP8_MAX,
        num_warps=4,
        num_stages=2,
    )


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

    Router-only launch: one program per row, no activation staging.
    """
    N, E = scores.shape
    K = int(topk)

    out_idx = torch.empty((N, K), dtype=torch.int64, device=scores.device)
    out_w = torch.empty((N, K), dtype=torch.float32, device=scores.device)
    if N == 0:
        return out_w, out_idx

    # The pack pointers are dead code under HAS_PACK=False; reuse the router
    # outputs so the launch never depends on an unrelated allocation.
    _launch(
        scores,
        bias,
        out_idx,
        out_w,
        scores,
        out_idx,
        out_w,
        grid=(N,),
        has_pack=False,
        hidden=0,
        route_scale=float(route_scale),
        norm_eps=float(norm_eps),
    )
    return out_w, out_idx


def fused_sqrtsoftplus_gate_pack(
    scores: torch.Tensor,  # [N, E] fp32/bf16 contiguous
    bias: torch.Tensor,  # [E] fp32 contiguous
    x: torch.Tensor,  # [N, H] bf16 contiguous
    out_fp8: torch.Tensor,  # [N, H] fp8_e4m3fn
    out_sf: torch.Tensor,  # [N, H / 128] fp32
    out_weights: torch.Tensor,  # [N, K] fp32
    out_indices: torch.Tensor,  # [N, K] int64
    topk: int,
    route_scale: float = 1.0,
    norm_eps: float = 1e-12,
) -> None:
    """Router epilogue + SM90 MegaMoE input staging in a single launch.

    Writes the top-k weights/indices and the FP8 activations with their FP32
    K128 scales straight into the collective buffer, replacing
    ``fused_sqrtsoftplus_gate(...)`` followed by the SM90 input packer.
    """
    N, _ = scores.shape
    if N == 0:
        return
    hidden = int(x.shape[1])
    if hidden <= 0 or hidden % _PACK_GRAN_K != 0:
        raise ValueError(
            "merged gate-pack requires a hidden width that is positive and "
            f"divisible by {_PACK_GRAN_K}, got {hidden}"
        )
    if x.shape[0] != N:
        raise ValueError(f"x rows ({x.shape[0]}) must match router rows ({N})")
    if out_indices.shape != out_weights.shape or out_indices.shape != (N, int(topk)):
        raise ValueError(
            "router outputs must be [N, topk] = "
            f"{(N, int(topk))}, got {tuple(out_indices.shape)} / "
            f"{tuple(out_weights.shape)}"
        )
    if out_fp8.shape != (N, hidden) or out_sf.shape != (N, hidden // _PACK_GRAN_K):
        raise ValueError(
            f"staging buffers must be {(N, hidden)} / "
            f"{(N, hidden // _PACK_GRAN_K)}, got {tuple(out_fp8.shape)} / "
            f"{tuple(out_sf.shape)}"
        )
    _launch(
        scores,
        bias,
        out_indices,
        out_weights,
        x,
        out_fp8,
        out_sf,
        grid=(N, hidden // _PACK_GRAN_K),
        has_pack=True,
        hidden=hidden,
        route_scale=float(route_scale),
        norm_eps=float(norm_eps),
    )
