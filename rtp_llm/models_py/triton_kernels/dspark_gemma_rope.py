"""DSpark Gemma Q/K normalization and partial NeoX RoPE.

Keep normalized values in FP32 through rotation, matching the MiniMax training
demo's fused Q/K path. A BF16 norm materialization before RoPE changes rounding.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@lru_cache(maxsize=None)
def dspark_rope_inv_freq(rope_theta, device):
    """Build the demo's FP32 frequency vector once (128 persistent bytes).

    Prewarm before graph capture, or pass a precomputed demo frequency vector
    explicitly. Pow implementations differ by ULPs across CUDA toolchains;
    using torch's frequency equation prevents long-position phase drift.
    """
    return 1.0 / (
        float(rope_theta) ** (torch.arange(0, 64, 2, device=device).float() / 64)
    )


@triton.jit
def _dspark_gemma_qk_norm_rope(
    Q,
    K,
    WQ,
    WK,
    POS,
    FREQ,
    OQ,
    OK,
    Q_TOKEN_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    K_HEAD_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    EPS: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    if head < HQ:
        x = tl.load(Q + token * Q_TOKEN_STRIDE + head * Q_HEAD_STRIDE + d).to(
            tl.float32
        )
        w = tl.load(WQ + d).to(tl.float32)
    else:
        x = tl.load(K + token * K_TOKEN_STRIDE + (head - HQ) * K_HEAD_STRIDE + d).to(
            tl.float32
        )
        w = tl.load(WK + d).to(tl.float32)
    # The demo reduces the rotary and non-rotary halves separately.
    square = x * x
    total = tl.sum(tl.where(d < 64, square, 0.0), 0) + tl.sum(
        tl.where(d >= 64, square, 0.0), 0
    )
    normed = (x * tl.rsqrt(total / 128.0 + EPS)) * (1.0 + w)
    partner = tl.gather(
        normed, tl.where(d < 32, d + 32, tl.where(d < 64, d - 32, d)), 0
    )
    # libdevice trig retains range reduction at long-context positions.
    frequency = tl.load(FREQ + d % 32)
    angle = tl.load(POS + token * POS_STRIDE).to(tl.float32) * frequency
    cosine, sine = libdevice.cos(angle), libdevice.sin(angle)
    rotated = tl.where(
        d < 32,
        normed * cosine - partner * sine,
        tl.where(d < 64, normed * cosine + partner * sine, normed),
    )
    if head < HQ:
        tl.store(OQ + (token * HQ + head) * 128 + d, rotated)
    else:
        tl.store(OK + (token * HK + head - HQ) * 128 + d, rotated)


def dspark_gemma_qk_norm_rope(
    q,
    k,
    q_weight,
    k_weight,
    positions,
    *,
    eps=1e-6,
    rope_theta=10000000.0,
    rotary_dim=64,
    inv_freq=None,
):
    """Return normalized/rotated BF16 Q/K from ``[tokens, heads, 128]``.

    Q and K can have independent head counts and interleaved QKV row strides.
    Empty Q supports context-only K projection; zero-token inputs launch no
    kernel. Raw BF16 Gemma weights stay raw: ``1 + weight`` is computed in FP32.
    Inputs are not mutated. Outputs are contiguous, with no FP32 workspace or
    trig cache. A shared 32-element FP32 frequency vector uses 128 persistent
    bytes per theta/device; positions and raw weights are read on every replay.
    """
    if q.ndim != 3 or k.ndim != 3 or q.shape[0] != k.shape[0]:
        raise ValueError("Q/K must be [tokens, heads, 128] with equal token counts")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or rotary_dim != 64:
        raise ValueError("DSpark fused Gemma RoPE requires head_dim=128, rotary_dim=64")
    if eps <= 0 or rope_theta <= 0:
        raise ValueError("eps and rope_theta must be positive")
    if positions.shape != (q.shape[0],) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("positions must be int32/int64 [tokens]")
    tensors = (q, k, q_weight, k_weight, positions)
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("Q/K, weights and positions must share a CUDA device")
    if any(t.dtype != torch.bfloat16 for t in tensors[:4]):
        raise TypeError("Q/K and raw Gemma weights must be BF16")
    for x in (q, k):
        if x.numel() and (
            x.stride(-1) != 1
            or x.stride(1) < 128
            or x.stride(0) < x.shape[1] * x.stride(1)
        ):
            raise ValueError(
                "Q/K require nonoverlapping rows and contiguous head channels"
            )
    if any(w.shape != (128,) or not w.is_contiguous() for w in (q_weight, k_weight)):
        raise ValueError("raw Gemma weights must be contiguous [128]")
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if q.shape[0] and q.shape[1] + k.shape[1]:
        if inv_freq is None:
            inv_freq = dspark_rope_inv_freq(float(rope_theta), q.device)
        if (
            inv_freq.shape != (32,)
            or inv_freq.dtype != torch.float32
            or inv_freq.device != q.device
            or not inv_freq.is_contiguous()
        ):
            raise ValueError("inv_freq must be contiguous FP32 [32] on the Q/K device")
        _dspark_gemma_qk_norm_rope[(q.shape[0], q.shape[1] + k.shape[1])](
            q,
            k,
            q_weight,
            k_weight,
            positions,
            inv_freq,
            out_q,
            out_k,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            positions.stride(0),
            q.shape[1],
            k.shape[1],
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out_q, out_k
