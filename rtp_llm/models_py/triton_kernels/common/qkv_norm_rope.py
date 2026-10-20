"""Fused qkv-slice RMSNorm + rotate-half RoPE Triton kernel (one pass).

Contract 6 (opt r8, BinaryFunctor 138ms/39300x elimination): the
DiffusionGemma attention pre-kernel chain used to run, per layer forward,

    q = qkv[..., :q_size]                       # strided slice
    q.copy_ -> q.reshape(-1, head_dim)          # 1 strided copy (cudaMemcpy2D)
    RMSNorm(q)                                  # flashinfer kernel launch
    rope: cos/sin gather x2 + positions.long()  # 3 index kernels
    rotate(q): x1*cos - x2*sin (4 muls + sub)   # BinaryFunctor x4
    torch.cat((o1, o2))                         # 1 copy (CatArrayBatchedCopy)

— the same for k (plus the v slice: 1 copy + norm), i.e. ~16 launches and
6+ full tensor passes per layer forward, all launch-bound at the canvas
M=256 shapes (r7 trace: BinaryFunctor Mul 137.8ms/39300, CatArrayBatchedCopy
15.5ms/7860, index_elementwise 35.7ms/11790, ~84 cudaMemcpy events/layer).

This kernel reads each qkv head slice ONCE straight from the strided qkv
buffer (arbitrary row stride), applies the per-head RMSNorm (gamma) in fp32,
rounds the normed value to bf16 exactly at the norm-output boundary (the
value the torch chain produces), applies the rotate-half rope in fp32 with
the cos/sin gathered in-kernel at the row's position, and writes one fresh
contiguous [M, H, D] bf16 output. HAS_ROPE=0 degenerates to the norm-only
path (the v slice). One launch per tensor.

Value contract (A/B referee ``[dgm-qkv-ab]``, DGM_QKV_AB=1):
  * RMSNorm: fp32 accumulation on both sides; the only deviation is the
    reduction order (tl.sum tree vs flashinfer's warp scan) — bf16 output
    differs at most in the last ulp (the dsv4/dsv41 fused-rmsnorm-rope
    precedent, and every fused norm kernel in this repo).
  * RoPE: the torch chain rounds to bf16 after each BinaryFunctor (mul,
    sub/add, cat) while the kernel rounds once at the store — <= 2 ulp
    bf16 noise (the r5 fused add-norm ruling class).
  * The normed value is rounded to bf16 IN REGISTER before the rotation so
    the norm->rope boundary matches the torch chain bit-for-bit.

The kernel compiles through the normal Triton JIT (TRITON_CACHE_DIR,
managed by jit_cache_manager as the "triton" component, the categorical_*
kernel pattern).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_NUM_WARPS = 4


@triton.jit
def _qkv_norm_rope_kernel(
    x_ptr,
    g_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    out_ptr,
    x_stride_m,
    x_stride_h,
    out_stride_m,
    out_stride_h,
    cos_stride,
    sin_stride,
    eps,
    M,
    HAS_GAMMA: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    HALF: tl.constexpr,
):
    # int64 offsets: the ragged row count x strides can exceed 2^31 for the
    # encoder prefill (M=4096 x row_stride ~10k stays < 2^31, but the same
    # kernel serves any canvas/prefill mix — keep it safe).
    m = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    dd = tl.arange(0, HALF)
    x_base = x_ptr + m * x_stride_m + h.to(tl.int64) * x_stride_h
    x1 = tl.load(x_base + dd).to(tl.float32)
    x2 = tl.load(x_base + HALF + dd).to(tl.float32)
    # per-head RMSNorm: fp32 sum of squares over the FULL head dim.
    var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / (2.0 * HALF)
    inv = tl.math.rsqrt(var + eps)
    if HAS_GAMMA:
        g1 = tl.load(g_ptr + dd).to(tl.float32)
        g2 = tl.load(g_ptr + HALF + dd).to(tl.float32)
        n1 = x1 * inv * g1
        n2 = x2 * inv * g2
    else:
        n1 = x1 * inv
        n2 = x2 * inv
    # round to bf16 exactly like the separate norm kernel's output store, so
    # the norm -> rope boundary values match the torch chain bit-for-bit.
    n1 = n1.to(tl.bfloat16).to(tl.float32)
    n2 = n2.to(tl.bfloat16).to(tl.float32)
    if HAS_ROPE:
        pos = tl.load(pos_ptr + m).to(tl.int64)
        # cos/sin cache rows are [head_dim] with the two halves duplicated
        # (emb = cat(freqs, freqs)); only the first half is ever read.
        c = tl.load(cos_ptr + pos * cos_stride + dd).to(tl.float32)
        s = tl.load(sin_ptr + pos * sin_stride + dd).to(tl.float32)
        o1 = n1 * c - n2 * s
        o2 = n2 * c + n1 * s
    else:
        o1 = n1
        o2 = n2
    out_base = out_ptr + m * out_stride_m + h.to(tl.int64) * out_stride_h
    tl.store(out_base + dd, o1.to(tl.bfloat16))
    tl.store(out_base + HALF + dd, o2.to(tl.bfloat16))


def norm_rope_usable(
    x: torch.Tensor,
    head_num: int,
    gamma: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
) -> bool:
    """Gate: CUDA bf16 strided [M, head_num*D] slice with a bf16 cos cache."""
    if not (x.is_cuda and x.dim() == 2 and x.dtype == torch.bfloat16):
        return False
    if x.stride(1) != 1:
        return False
    if head_num < 1 or x.shape[0] < 1 or x.shape[1] % head_num != 0:
        return False
    half = x.shape[1] // head_num // 2
    if half < 1 or (half & (half - 1)) != 0:
        # HALF must be a power of two for tl.arange.
        return False
    if gamma is not None:
        if gamma.dim() != 1 or gamma.numel() != 2 * half or gamma.stride(0) != 1:
            return False
        if gamma.dtype not in (torch.bfloat16, torch.float32):
            return False
        if gamma.device != x.device:
            return False
    if cos is None and sin is None and positions is None:
        # norm-only mode (the v slice): no rope operands to validate.
        return True
    if cos is None or sin is None or positions is None:
        return False
    for cache in (cos, sin):
        if cache.dim() != 2 or cache.dtype != torch.bfloat16:
            return False
        if cache.stride(1) != 1 or cache.stride(0) < 2 * half:
            return False
        if cache.device != x.device:
            return False
    if positions.dim() != 1 or positions.numel() != x.shape[0]:
        return False
    if positions.dtype not in (torch.int64, torch.int32):
        return False
    if positions.device != x.device:
        return False
    # the cache must cover the max position (host-side check avoided: the
    # rope module guarantees it via its own memo; we bound only stride use).
    return True


def qkv_norm_rope(
    x: torch.Tensor,
    head_num: int,
    eps: float,
    gamma: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    positions: Optional[torch.Tensor],
) -> torch.Tensor:
    """RMSNorm (+ optional rotate-half rope) over a strided qkv slice.

    ``x``: [M, head_num * D] (a strided slice view of the fused qkv buffer
    is fine — only the row stride may be larger than the slice width).
    Returns a fresh contiguous [M, head_num, D] bf16 tensor: the normed
    (and optionally rope-rotated) heads. ``gamma=None`` -> no-scale norm
    (the v-slice / router class); ``cos/sin/positions=None`` -> norm only.
    cos/sin rows are [D] (the duplicated-halves cache layout).
    """
    d = x.shape[1] // head_num
    half = d // 2
    out = torch.empty(
        (x.shape[0], head_num, d), dtype=torch.bfloat16, device=x.device
    )
    _qkv_norm_rope_kernel[(x.shape[0], head_num)](
        x,
        gamma if gamma is not None else x,
        cos if cos is not None else x,
        sin if sin is not None else x,
        positions if positions is not None else x,
        out,
        x.stride(0),
        d,
        out.stride(0),
        out.stride(1),
        cos.stride(0) if cos is not None else 1,
        sin.stride(0) if sin is not None else 1,
        float(eps),
        x.shape[0],
        HAS_GAMMA=gamma is not None,
        HAS_ROPE=cos is not None,
        HALF=half,
        num_warps=_NUM_WARPS,
    )
    return out
