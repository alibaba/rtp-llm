"""Preserve the official FP8 QAT boundary for BF16 Prefill SWA reads."""

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_qat(X):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    value = tl.load(X + row * 512 + col).to(tl.float32).reshape(8, 64)
    amax = tl.maximum(tl.max(tl.abs(value), 1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    quant = (value / scale[:, None]).to(tl.float8e4nv).to(tl.float32)
    result = (quant * scale[:, None]).reshape(512)
    tl.store(X + row * 512 + col, result, col < 448)


def quantize_dequantize_swa_kv_(kv):
    """Quantize non-RoPE groups of 64 in place; keep 64 RoPE values unchanged."""
    if (
        not kv.is_cuda
        or kv.dtype != torch.bfloat16
        or kv.ndim < 2
        or kv.shape[-1] != 512
        or not kv.is_contiguous()
    ):
        raise ValueError("SWA QAT requires contiguous CUDA BF16 KV with head_dim=512")
    rows = kv.numel() // 512
    if rows:
        _kv_qat[(rows,)](kv, num_warps=4)
    return kv
