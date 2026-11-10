"""Pre-SM89 fused dequantization GEMMs for DSV4.1 checkpoint weights.

A100 (SM80) has no FP8/FP4 tensor cores and no DeepGEMM kernels for the
native 32x32 MXFP8 / packed-FP4 checkpoint layouts. Materializing a BF16
copy of every expert or dense weight is both slow and, for this model,
larger than the remaining device memory. These kernels decode each weight
tile inside the GEMM inner loop, so the only persistent copies remain the
original checkpoint tensors.

The arithmetic matches the reference dequantize-then-F.linear path:
  FP8: w = e4m3 * 2**(ue8m0_scale - 127)          (32x32 blocks)
  FP4: w = e2m1(nibble) * 2**(ue8m0_scale - 127)  (1x32 blocks)
and accumulates in FP32 before converting to BF16.
"""

from __future__ import annotations

import os

import torch

try:  # Triton is a runtime dependency of the cuda image, not of CPU imports.
    import triton
    import triton.language as tl

    _TRITON = True
except Exception:  # pragma: no cover - CPU/import-only environments
    triton = None
    tl = None
    _TRITON = False


def triton_dequant_enabled() -> bool:
    """Default on for devices without native FP8/FP4 tensor cores."""
    if not _TRITON or not torch.cuda.is_available():
        return False
    if getattr(torch.version, "hip", None) is not None:
        return False
    if os.getenv("DSV41_TRITON_DEQUANT", "1") == "0":
        return False
    return tuple(torch.cuda.get_device_capability()) < (8, 9)


@triton.jit
def _dequant_matmul_kernel(
    X,
    W,
    S,
    Y,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_ym,
    stride_yn,
    IS_FP4: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rn_mask = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    x_ptrs = X + rm[:, None] * stride_xm + rk[None, :] * stride_xk

    rk_s = tl.arange(0, BLOCK_K // 32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptrs, mask=rm[:, None] < M, other=0.0)
        if IS_FP4:
            # Contiguous [BLOCK_N, BLOCK_K/2] packed bytes; interleave the two
            # nibbles into the K order instead of gathering one byte per K.
            rk2 = tl.arange(0, BLOCK_K // 2)
            wb = tl.load(
                W + rn[:, None] * stride_wn + (k0 // 2 + rk2)[None, :] * stride_wk,
                mask=rn_mask[:, None],
                other=0,
            ).to(tl.uint8)
            low = (wb & 0x0F).to(tl.uint8)
            high = ((wb >> 4) & 0x0F).to(tl.uint8)
            nib = tl.interleave(low, high)
            idx = nib & 7
            mag = tl.where(
                idx == 0,
                0.0,
                tl.where(
                    idx == 1,
                    0.5,
                    tl.where(
                        idx == 2,
                        1.0,
                        tl.where(
                            idx == 3,
                            1.5,
                            tl.where(
                                idx == 4,
                                2.0,
                                tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)),
                            ),
                        ),
                    ),
                ),
            )
            w = tl.where((nib & 8) != 0, -mag, mag)
        else:
            # SM80 Triton has no fp8e4nv load type. Decode E4M3 bytes with
            # integer ops: sign | 4-bit exponent (bias 7) | 3-bit mantissa.
            wb8 = tl.load(
                W + rn[:, None] * stride_wn + kk[None, :] * stride_wk,
                mask=rn_mask[:, None],
                other=0,
            ).to(tl.uint8)
            sign = tl.where((wb8 & 0x80) != 0, -1.0, 1.0)
            exp = ((wb8 >> 3) & 0x0F).to(tl.float32)
            mant = (wb8 & 0x07).to(tl.float32)
            normal = (1.0 + mant / 8.0) * tl.exp2(exp - 7.0)
            subnormal = (mant / 8.0) * tl.exp2(-6.0)
            w = tl.where(exp == 0, subnormal, normal) * sign
        s_rn = rn if IS_FP4 else (rn // 32)
        sb = tl.load(
            S + s_rn[:, None] * stride_sn + (k0 // 32 + rk_s)[None, :] * stride_sk,
            mask=rn_mask[:, None],
            other=127,
        ).to(tl.uint8)
        # One UE8M0 scale per 32 K elements; broadcast it to the tile width.
        sb = tl.reshape(
            tl.broadcast_to(sb[:, :, None], (BLOCK_N, BLOCK_K // 32, 32)),
            (BLOCK_N, BLOCK_K),
        )
        scale = tl.exp2(sb.to(tl.float32) - 127.0)
        w = (w * scale).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk

    y = acc.to(tl.bfloat16)
    tl.store(
        Y + rm[:, None] * stride_ym + rn[None, :] * stride_yn,
        y,
        mask=(rm[:, None] < M) & rn_mask[None, :],
    )


def _launch(x: torch.Tensor, w: torch.Tensor, s: torch.Tensor, is_fp4: bool):
    if not triton_dequant_enabled():
        return None
    if x.dim() < 2:
        return None
    k = x.shape[-1]
    n = w.shape[0]
    if is_fp4:
        if w.dtype != torch.int8 or w.shape[1] * 2 != k:
            return None
        if s.shape != (n, k // 32):
            return None
    else:
        if w.shape != (n, k):
            return None
        if tuple(s.shape) != (n // 32, k // 32):
            return None
    if k % 128 or n % 32:
        return None
    if s.dtype != torch.float8_e8m0fnu:
        return None
    x2 = x.reshape(-1, k)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    x2 = x2.contiguous()
    m = x2.shape[0]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    w_view = w.view(torch.uint8)
    s_view = s.view(torch.uint8)
    if m <= 16:
        block_m, block_n, block_k, warps, stages = 16, 32, 256, 4, 3
    else:
        block_m, block_n, block_k, warps, stages = 64, 64, 64, 4, 3
    # Tuning hook used by the A100 bring-up sweep; defaults are the measured
    # best shapes for decode. Values stay overridable for future devices.
    block_n = int(os.getenv("DSV41_DEQUANT_BN", block_n))
    block_k = int(os.getenv("DSV41_DEQUANT_BK", block_k))
    warps = int(os.getenv("DSV41_DEQUANT_WARPS", warps))
    stages = int(os.getenv("DSV41_DEQUANT_STAGES", stages))
    while block_k > 64 and k % block_k:
        block_k //= 2
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _dequant_matmul_kernel[grid](
        x2,
        w_view,
        s_view,
        out,
        m,
        n,
        k,
        x2.stride(0),
        x2.stride(1),
        w_view.stride(0),
        w_view.stride(1),
        s_view.stride(0),
        s_view.stride(1),
        out.stride(0),
        out.stride(1),
        IS_FP4=is_fp4,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )
    return out.reshape(*x.shape[:-1], n)


def fp8_dequant_matmul(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    """x [..., K] BF16 @ dequant(weight [N, K]) -> [..., N] BF16."""
    return _launch(x, weight, scale, is_fp4=False)


def fp4_dequant_matmul(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor):
    """x [..., K] BF16 @ dequant(weight [N, K/2] packed, scale [N, K/32])."""
    return _launch(x, weight, scale, is_fp4=True)
