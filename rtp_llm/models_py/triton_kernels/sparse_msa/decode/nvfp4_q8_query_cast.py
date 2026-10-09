"""Fused saturating BF16 query pre-rounding into existing E4M3 buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_query_cast(
    Q,
    IQ,
    Q8,
    IQ8,
    NQ: tl.constexpr,
    NI: tl.constexpr,
    Q_BLOCKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    if pid < Q_BLOCKS:
        x = pid * BLOCK + lane
        value = tl.load(Q + x, x < NQ, other=0).to(tl.float32)
        tl.store(Q8 + x, value.to(tl.float8e4nv), x < NQ)
    else:
        x = (pid - Q_BLOCKS) * BLOCK + lane
        value = tl.load(IQ + x, x < NI, other=0).to(tl.float32)
        tl.store(IQ8 + x, value.to(tl.float8e4nv), x < NI)


def fused_query_cast(q, idx_q, q8, idx_q8, *, cast_main_query=True):
    """One launch, no tensor allocations; all captured rows including padding.

    Inputs remain unmodified, unlike the baseline's BF16 round trip. Integration
    is restricted to callers with no other rounded-carrier consumer.
    BF16-native attention can disable main-Q conversion; index-Q is still cast.
    """
    for value, out in ((q, q8), (idx_q, idx_q8)):
        if (
            value.dtype != torch.bfloat16
            or out.dtype != torch.float8_e4m3fn
            or not value.is_cuda
            or value.device != out.device
            or value.shape != out.shape
            or value.ndim != 3
            or not value.is_contiguous()
            or not out.is_contiguous()
        ):
            raise ValueError(
                "requires contiguous CUDA BF16 input and matching E4M3 output"
            )
    if q.device != idx_q.device or q.shape[0] != idx_q.shape[0]:
        raise ValueError("Q/index-Q must share device and row count")
    nq, ni = q.numel() if cast_main_query else 0, idx_q.numel()
    if nq + ni:
        block = 256
        qb = triton.cdiv(nq, block)
        _fused_query_cast[(qb + triton.cdiv(ni, block),)](
            q, idx_q, q8, idx_q8, nq, ni, qb, block, num_warps=4
        )
