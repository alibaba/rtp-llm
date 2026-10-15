"""Fused Xing4 hc-row RMSNorm + hc_fn skinny GEMV Triton kernel (fusion round-2, K1+GEMM).

Numerical authority (do not edit): ``Xing4HCUnit._pre_impl`` /
``_pre_fused`` in ``rtp_llm/models_py/model_desc/xing4_0_model.py``::

    x_flat      = x.flatten(-2)            # [T, hc*H] bf16 (14336 for hc=4, H=3584)
    rsqrt       = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + eps)
    flat_normed = (x_flat.float() * rsqrt).to(x.dtype)   # bf16, feeds F.linear
    mixes       = F.linear(flat_normed, fn)              # [T, (hc+2)*hc] bf16

The eager chain is K1 (norm, 6-8 micro kernels, see norm_triton.py) plus the
cuBLAS ``F.linear`` on the skinny [T, 14336] x [24, 14336] shape. Measured on
B300/SM103 in CUDA-graph mode (build_logs/k41_chain_cost.py,
k41_gemv_proto.py), at decode sizes the cuBLAS lowering picks a split-K pair
whose wall cost is 11-19 us per call — the single largest stage of the mHC
pre-chain (K1 2.6 us + GEMM 11.2 us = 13.8 us at T=1). This kernel replaces
BOTH stages with ONE launch (grid = (rows, N_out))::

    mixes[t, n] = bf16( sum_c fp32(bf16round(fp32(x[t, c]) * rstd[t]))
                             * fp32(fn[n, c]) )          # ascending-c fp32 acc
    rstd[t]     = rsqrt(mean_c(fp32(x[t, c])**2) + eps)  # block-cyclic, as K1

Rounding structure — identical to the eager chain except the fp32 summation
orders (the same equivalence class every fused K lane gates with
rtol = atol = 2**-7):

* ``rstd`` — K1's block-cyclic fp32 square-sum, already tolerance-class
  vs torch's ``mean`` (norm_triton.py documents the 1-ulp behavior).
* ``flat_normed`` — the reference rounds ``fp32(x) * rsqrt`` to bf16 BEFORE
  the GEMM; the kernel reproduces exactly that rounding through the opaque
  RTNE bitcast (``_rtne_round_bf16``; dropping it flips ~40x more output
  bits vs the reference — measured 85/192 vs 2/192 at T=8).
* ``mixes`` — products of two bf16 values are exact in fp32; the reference
  GEMM (cuBLAS split-K) accumulates fp32 and rounds ONCE to bf16; the kernel
  accumulates the 14336 products in fp32 in ascending-c order and rounds
  once — only a cuBLAS reordering of the sum can differ, bounded by one bf16
  ulp after rounding.

Why the K4 -> K1 rsqrt-precompute chain was NOT taken instead (measured,
rejected): having K4's epilogue export row partial square-sums so the next
K1 skips its pass 1 costs a cross-warp tile reduction on the issue-bound K4
(+0.6-1.2 us at T=1, +85 us at T=4096) while K1's pass 1 is worth only
~0.08 us at decode (the kernel is latency-bound; the pass-2 re-read is
L2-resident) — a net loss at every T (build_logs/k41_t1_graph_bench.py).
Computing rstd inside this kernel, over data every program already loads,
is the cheaper form of the same fusion.

Grid/geometry: grid = (rows, N_out); each program owns one (row, output)
pair — the fn row re-read scales with rows (~0.16 us/row on top of a ~3.8 us
base) while the cuBLAS pair is a flat ~14 us through T=64, so the fused
form wins through ~48 rows and loses from ~64 on (measured crossover,
CUDA-graph device time, 3 repetitions each: T=1-12 +10.4 us, T=16 +8.9,
T=32 +5.4, T=48 +3.4, T=64 +0.2, T=512 ~-130). The dispatch is therefore
rows-gated by ``XING4_HC_FN_GEMV_MAX_TOKENS`` (default 32 — the standard
decode-graph ladder 1/2/4/8/16/32, all measured positive; 0 disables,
resolved per call so ops can tune and tests can force both paths).
``pid`` and every row-dependent offset are int64 because
rows * 14336 passes INT32_MAX from ~149726 rows (same contract as
norm_triton.py).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_MAX_TOKENS_ENV = "XING4_HC_FN_GEMV_MAX_TOKENS"
_DEFAULT_MAX_TOKENS = 32
_DEFAULT_BLOCK_N = 2048  # 14336 = 7 * 2048 -> block-cyclic row loops, as K1
_DEFAULT_NUM_WARPS = 8


def hc_fn_gemv_max_tokens() -> int:
    """Resolve the T gate: rows <= this use the fused GEMV (0 disables).

    Mirrors the ``XING4_HC_FALLBACK_CHUNK_TOKENS`` discipline: silent
    fallback to the default on unparsable values, resolved per call.
    """
    raw = os.environ.get(_MAX_TOKENS_ENV)
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_TOKENS
    try:
        return max(int(raw), 0)
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS


@triton.jit
def _rtne_round_bf16(x):
    """RTNE-round an fp32 tile to bf16-representable values, kept as fp32.

    Same helper and rationale as post_triton.py: Triton 3.6 canonicalizes
    ``x.to(tl.bfloat16).to(tl.float32)`` back to ``x`` (extf(truncf) -> x),
    which would delete exactly the flat_normed bf16 rounding this kernel
    must reproduce. The bit-manipulation form is the standard fp32->bf16
    round-to-nearest-even; NaN payloads are not preserved and this path
    never sees NaN (normed streams are finite by construction).
    """
    u = x.to(tl.int32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    u = u & -65536  # 0xFFFF0000: sign+exponent+top-7 mantissa bits of bf16
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _hc_fn_gemv_kernel(
    x_ptr,  # [rows, n_cols] bf16 (row stride may exceed n_cols, unit col stride)
    fn_ptr,  # [n_out, n_cols] bf16
    out_ptr,  # [rows, n_out] bf16
    eps,
    n_cols,
    x_row_stride,
    fn_row_stride,
    out_row_stride,
    BLOCK_N: tl.constexpr,
):
    # int64 BEFORE the row-stride multiply: rows * 14336 exceeds INT32_MAX
    # from ~149726 rows on (same contract as norm_triton.py).
    pid_t = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    x_row = x_ptr + pid_t * x_row_stride
    fn_row = fn_ptr + pid_n * fn_row_stride

    # Pass 1: fp32 row square-sum, block-cyclic (14336 = 7*2048) — the same
    # accumulation structure as _hc_row_rmsnorm_kernel's pass 1.
    acc_sq = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc_sq += v * v
    rstd = tl.rsqrt(tl.sum(acc_sq, axis=0) / n_cols + eps)

    # Pass 2: mixes[t, n] = sum_c bf16round(fp32(x)*rstd) * fp32(fn[n, c]),
    # fp32 ascending-c accumulation, one bf16 round after the full sum
    # (the cuBLAS bf16 GEMM structure).
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < n_cols
        v = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        f = tl.load(fn_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += _rtne_round_bf16(v * rstd) * f
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_t * out_row_stride + pid_n, total.to(tl.bfloat16))


def hc_fn_gemv_triton(
    x: torch.Tensor,
    fn: torch.Tensor,
    eps: float,
    *,
    block_n: int = _DEFAULT_BLOCK_N,
) -> torch.Tensor:
    """Fused row RMSNorm + hc_fn GEMV: ``mixes = F.linear(rmsnorm(x), fn)``.

    Args:
        x: ``[..., hc, H]`` or ``[..., hc*H]`` bf16 CUDA tensor (the HC
            residual stream); trailing dims are flattened into rows exactly
            like ``x.flatten(-2)``. Row slices (row stride > cols, unit
            column stride) launch without copies; other non-row-major
            layouts fall back to one contiguous copy.
        fn: ``[n_out, hc*H]`` bf16 CUDA contiguous weight (``Xing4HCUnit.fn``
            cast to the stream dtype, i.e. ``_fn_for(dtype)``).
        eps: the RMSNorm epsilon, added inside the rsqrt (fp32, matching the
            reference).
        block_n: column block of the block-cyclic loops (power of two).

    Returns:
        ``[..., n_out]`` bf16 CUDA tensor — the same shape and dtype
        ``F.linear(flat_normed, fn)`` produces for the flattened ``x``.
    """
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bfloat16, got {x.dtype}")
    if fn.dtype != torch.bfloat16:
        raise ValueError(f"fn must be bfloat16, got {fn.dtype}")
    if fn.dim() != 2 or not fn.is_contiguous():
        raise ValueError(
            f"fn must be a contiguous 2D [n_out, hc*H] tensor, "
            f"got shape={tuple(fn.shape)} strides={fn.stride()}"
        )
    if x.dim() < 2:
        raise ValueError(
            f"hc_fn_gemv_triton expects dim >= 2 ([T, hc, H] or [T, hc*H]), "
            f"got {x.dim()}"
        )
    x_flat = x.flatten(-2) if x.dim() > 2 else x
    leading = x_flat.shape[:-1]
    cols = int(x_flat.shape[-1])
    rows = x_flat.numel() // max(cols, 1)
    n_out = int(fn.shape[0])
    if cols != int(fn.shape[1]):
        raise ValueError(
            f"x row width {cols} != fn columns {int(fn.shape[1])} "
            "(x must be the hc*H-flattened stream)"
        )
    if not x.is_cuda:
        raise ValueError("hc_fn_gemv_triton requires CUDA tensors")
    if fn.device != x.device:
        raise ValueError(f"fn on {fn.device}, expected {x.device}")
    if cols < 1 or n_out < 1:
        shape = (*leading, n_out)
        return torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    if block_n < 1 or (block_n & (block_n - 1)) != 0:
        raise ValueError(f"block_n must be a power of two, got {block_n}")

    x_rows = x_flat.reshape(rows, cols)
    if x_rows.stride(-1) != 1:
        x_rows = x_rows.contiguous()
    out = torch.empty(
        (*leading, n_out), dtype=torch.bfloat16, device=x.device
    )
    out_rows = out.reshape(rows, n_out)
    if out_rows.stride(-1) != 1:
        out_rows = out_rows.contiguous()

    block = block_n if cols >= block_n else triton.next_power_of_2(cols)
    num_warps = 8 if block >= 2048 else (4 if block >= 512 else 1)
    _hc_fn_gemv_kernel[(rows, n_out)](
        x_rows,
        fn,
        out_rows,
        eps,
        cols,
        x_rows.stride(0),
        fn.stride(0),
        out_rows.stride(0),
        BLOCK_N=block,
        num_warps=num_warps,
    )
    return out
