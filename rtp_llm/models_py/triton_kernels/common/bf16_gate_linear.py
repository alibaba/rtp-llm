"""Single-launch BF16 Qwen3.5 gates with a batch-independent reduction.

Each CTA computes all K partitions for its output tile and reduces them locally.
No atomics, intermediate tensor, or M-dependent autotuning is used. Two contiguous
K partitions reproduce the tested 64-row router baseline; BA uses four. The
caller selects this opt-in implementation only during an explicit decode phase.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.prefill_input_log import trace_triton


@triton.jit(do_not_specialize=["M"])
def _bf16_gate_gemm(
    X,
    W,
    B,
    Y,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    XS: tl.constexpr,
    XK: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    splits = tl.arange(0, SK)
    acc = tl.zeros((SK, BM, BN), tl.float32)
    for block in range(tl.cdiv(K // SK, BK)):
        k = splits[:, None] * (K // SK) + block * BK + ks[None, :]
        x = tl.load(
            X + rows[None, :, None] * XS + k[:, None, :] * XK,
            rows[None, :, None] < M,
            0,
        )
        w = tl.load(
            W + k[:, :, None] * WK + cols[None, None, :] * WN,
            cols[None, None, :] < N,
            0,
        )
        acc = tl.dot(x, w, acc)
    if SK == 4:
        # Keep the final partial-sum additions ordered, independent of M/layout.
        paired = tl.reshape(tl.permute(acc, (1, 2, 0)), (BM, BN, 2, 2))
        even, odd = tl.split(paired)
        p0, p2 = tl.split(even)
        p1, p3 = tl.split(odd)
        total = ((p0 + p1) + p2) + p3
    else:
        total = tl.sum(acc, 0)
    if HAS_BIAS:
        total += tl.load(B + cols, cols < N, 0).to(tl.float32)[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        total,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def is_supported(x, weight, bias=None) -> bool:
    return (
        torch.version.hip is None
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == weight.ndim == 2
        and x.shape[1] == 4096
        and weight.shape in ((128, 4096), (512, 4096))
        and (
            bias is None
            or (
                bias.device == x.device
                and bias.dtype == x.dtype
                and bias.shape == (weight.shape[0],)
                and bias.stride(0) == 1
            )
        )
    )


def maybe_bf16_gate_linear(x, weight, bias=None) -> Optional[torch.Tensor]:
    """Return None outside the supported BF16 shapes; accept any row count.

    Reduction geometry depends only on the output dimension, never on M.
    Row/column-major weights and strided input rows/columns are supported.
    """
    if not is_supported(x, weight, bias):
        return None
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m:
        bm = 16
        bn, bk, sk = (32, 128, 2) if n == 512 else (16, 256, 4)
        trace_triton(
            "bf16_gate_linear:fixed_geometry",
            _bf16_gate_gemm,
            (triton.cdiv(m, bm), triton.cdiv(n, bn)),
            x,
            weight,
            bias if bias is not None else output,
            output,
            m,
            N=n,
            K=k,
            XS=x.stride(0),
            XK=x.stride(1),
            WN=weight.stride(0),
            WK=weight.stride(1),
            HAS_BIAS=bias is not None,
            BM=bm,
            BN=bn,
            BK=bk,
            SK=sk,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
    return output
