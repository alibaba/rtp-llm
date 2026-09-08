"""Batched CUDA-core Hadamard, BF16 rounding, FP8 quantization and gate fold."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    _had128_inline,
)

ROWS = 16


@triton.jit
def indexer_hadamard_quant_fold_kernel(
    q, w, o, wf, R: tl.constexpr, BR: tl.constexpr, ROTATE: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    d = tl.arange(0, 128)
    x = tl.load(q + r[:, None] * 128 + d[None, :], mask=r[:, None] < R, other=0).to(
        tl.float32
    )
    if ROTATE:
        x = (_had128_inline(x, BR, 128) * (128**-0.5)).to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-12)
    y = (x / scale[:, None]).to(tl.float8e4nv)
    tl.store(o + r[:, None] * 128 + d[None, :], y, mask=r[:, None] < R)
    gate = tl.load(w + r, mask=r < R, other=0).to(tl.float32)
    tl.store(wf + r, gate * scale, mask=r < R)


def indexer_hadamard_quant_fold(q, w, rotate=True):
    if q.dtype != torch.bfloat16 or q.shape[-1] != 128 or w.shape != q.shape[:-1]:
        raise ValueError(
            "Indexer fused quantization expects BF16 [..., 128] and matching head gates"
        )
    if (
        not q.is_contiguous()
        or not w.is_contiguous()
        or not q.is_cuda
        or w.device != q.device
    ):
        raise ValueError(
            "Indexer fused quantization requires contiguous tensors on the same CUDA device"
        )
    o = torch.empty_like(q, dtype=torch.float8_e4m3fn)
    wf = torch.empty_like(w, dtype=torch.float32)
    r = q.numel() // 128
    if r == 0:
        return o, wf
    indexer_hadamard_quant_fold_kernel[(triton.cdiv(r, ROWS),)](
        q, w, o, wf, r, ROWS, rotate, num_warps=4
    )
    return o, wf
