"""Experimental FP32 head-gate projection fused with indexer Q quantization."""

import torch, triton, triton.language as tl
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    _had128_inline,
)


@triton.jit
def gate_quant_kernel(
    X, W, Q, O, F, N: tl.constexpr, K: tl.constexpr, BH: tl.constexpr
):
    token = tl.program_id(0)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    k = tl.arange(0, K)
    x = tl.load(X + token * K + k).to(tl.float32)
    w = tl.load(W + head[:, None] * K + k[None, :], head[:, None] < N, other=0)
    gate = tl.sum(w * x[None, :], axis=1)
    d = tl.arange(0, 128)
    q = tl.load(
        Q + (token * N + head[:, None]) * 128 + d[None, :], head[:, None] < N, other=0
    ).to(tl.float32)
    q = (_had128_inline(q, BH, 128) * (128**-0.5)).to(tl.bfloat16).to(tl.float32)
    scale = tl.exp2(
        tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(q), axis=1), 1e-4) * (1.0 / 448.0)))
    )
    tl.store(
        O + (token * N + head[:, None]) * 128 + d[None, :],
        tl.minimum(tl.maximum(q / scale[:, None], -448.0), 448.0).to(tl.float8e4nv),
        head[:, None] < N,
    )
    tl.store(F + token * N + head, gate * scale, head < N)


def gate_quant(x, weight, q, head_tile=4, warps=4):
    assert x.dtype == q.dtype == torch.bfloat16 and weight.dtype == torch.float32
    assert x.is_contiguous() and weight.is_contiguous() and q.is_contiguous()
    assert x.ndim == 2 and x.shape[1] == 4096 and q.ndim == 3 and q.shape[-1] == 128
    assert q.shape[:2] == (x.shape[0], weight.shape[0]) and weight.shape[1] == 4096
    output = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    folded = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    gate_quant_kernel[(x.shape[0], triton.cdiv(weight.shape[0], head_tile))](
        x, weight, q, output, folded, weight.shape[0], 4096, head_tile, num_warps=warps
    )
    return output, folded
