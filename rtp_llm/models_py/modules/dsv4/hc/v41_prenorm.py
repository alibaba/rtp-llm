"""Small-token V4.1 mHC projection with split-K and FP32 RMS reduction.

Like vLLM's delayed mHC, use DeepGEMM's combined projection/square-sum
primitive to distribute the long K dimension across SMs. Preserve RTP's
existing TF32 weight rounding and leave delayed mixing to the caller.
"""

import importlib
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def _has_prenorm_gemm() -> bool:
    try:
        module = importlib.import_module("deep_gemm")
    except ModuleNotFoundError as error:
        if error.name == "deep_gemm":
            return False
        raise
    return callable(getattr(module, "tf32_hc_prenorm_gemm", None))


def is_supported(residual: torch.Tensor, fn: torch.Tensor) -> bool:
    """Static inference gate; large prefills retain the existing implementation."""
    if os.environ.get("DSV41_FUSED_MHC_PRENORM", "1") == "0":
        return False
    if (
        residual.device.type != "cuda"
        or residual.dtype != torch.bfloat16
        or residual.ndim not in (3, 4)
        or residual.shape[-2:] != (4, 5120)
        or not residual.is_contiguous()
        or fn.device != residual.device
        or fn.dtype != torch.float32
        or fn.shape != (24, 20480)
        or not fn.is_contiguous()
        or residual.numel() // 20480 > 64
        or (torch.is_grad_enabled() and (residual.requires_grad or fn.requires_grad))
    ):
        return False
    return (
        torch.cuda.get_device_capability(residual.device)[0] == 10
        and _has_prenorm_gemm()
    )


def prepare_tf32_weight(fn: torch.Tensor) -> torch.Tensor:
    """Cache the TF32 values consumed by the old biased/truncating MMA path.

    The old TileLang wrapper adds 0x1000 to FP32 bits before TensorCore
    truncation. Explicitly clear the low 13 bits as well, so DeepGEMM's TMA
    conversion cannot round those weights a second time. Inference tensors
    without version counters follow the existing immutable-weight contract.
    """
    try:
        version = fn._version
    except RuntimeError:
        version = None
    key = (version, fn.data_ptr(), tuple(fn.shape), tuple(fn.stride()), fn.device)
    cached = getattr(fn, "_dsv41_prenorm_tf32_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    weight = ((fn.view(torch.int32) + 0x1000) & -8192).view(torch.float32)
    fn._dsv41_prenorm_tf32_cache = (key, weight)
    return weight


@triton.jit
def _reduce_prenorm_kernel(
    partial,
    square_sum,
    output,
    TOKENS: tl.constexpr,
    SPLITS: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    columns = tl.arange(0, 32)
    projection = tl.full((32,), 0, tl.float32)
    total_square = tl.full((), 0, tl.float32)
    # Match the existing TileLang normalization's ordered split accumulation.
    for split in tl.static_range(SPLITS):
        projection += tl.load(
            partial + (split * TOKENS + token) * 24 + columns,
            columns < 24,
            other=0,
        )
        total_square += tl.load(square_sum + split * TOKENS + token)
    norm = tl.rsqrt(total_square / 20480 + EPS)
    tl.store(output + token * 24 + columns, projection * norm, columns < 24)


def prenorm(
    residual: torch.Tensor, fn: torch.Tensor, eps: float
) -> torch.Tensor | None:
    """Return FP32 [..., 24] mixes, or None for the original-path fallback."""
    if not is_supported(residual, fn):
        return None
    tokens = residual.numel() // 20480
    output = torch.empty(
        (*residual.shape[:-2], 24), dtype=torch.float32, device=residual.device
    )
    if tokens == 0:
        return output

    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    # vLLM bounds its small-token prenorm specialization to 16 K partitions.
    splits = 16
    weight = prepare_tf32_weight(fn)
    partial = torch.empty(
        (splits, tokens, 24), dtype=torch.float32, device=residual.device
    )
    square_sum = torch.empty(
        (splits, tokens), dtype=torch.float32, device=residual.device
    )
    tf32_hc_prenorm_gemm(
        residual.view(tokens, 20480), weight, partial, square_sum, splits
    )
    _reduce_prenorm_kernel[(tokens,)](
        partial,
        square_sum,
        output,
        tokens,
        splits,
        eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
