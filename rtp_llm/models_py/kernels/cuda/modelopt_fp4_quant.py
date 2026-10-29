"""Fuse dense SwiGLU and TP2 residual norm producers into NVFP4 quantization."""

import functools
import os
from pathlib import Path

import torch

from rtp_llm.models_py.utils.arch import get_sm

try:
    from flashinfer import silu_and_mul_scaled_nvfp4_experts_quantize
except ImportError:
    silu_and_mul_scaled_nvfp4_experts_quantize = None

_FUSED_PRODUCER_ENABLED = os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"


@functools.lru_cache(maxsize=1)
def _norm_module():
    from flashinfer.jit import env, gen_jit_spec

    return gen_jit_spec(
        "rtp_modelopt_ar_norm_fp4_v1",
        [Path(__file__).with_name("modelopt_norm_fp4_quant.cu")],
        extra_include_paths=[
            str(env.FLASHINFER_CSRC_DIR),
            str(env.FLASHINFER_CSRC_DIR / "nv_internal"),
            str(env.FLASHINFER_CSRC_DIR / "nv_internal/include"),
        ],
        extra_cuda_cflags=[
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-DENABLE_FP4",
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
    ).build_and_load()


def residual_norm_fp4_quant(input, residual, weight, scale, eps, communicator=None):
    """Sum TP2 partials, update BF16 norm/residual and return packed NVFP4."""
    if (
        not _FUSED_PRODUCER_ENABLED
        or communicator is None
        or not input.is_cuda
        or input.ndim != 2
        or input.dtype != torch.bfloat16
        or input.shape[1] != 5120
        or not (input.shape[0] == 5 or 17 <= input.shape[0] <= 512)
        or not input.is_contiguous()
        or input.requires_grad
        or get_sm(input.device) != (12, 0)
        or residual.shape != input.shape
        or residual.dtype != input.dtype
        or residual.device != input.device
        or not residual.is_contiguous()
        or weight.shape != (5120,)
        or weight.dtype != input.dtype
        or weight.device != input.device
        or not weight.is_contiguous()
        or scale.device != input.device
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or not communicator.is_supported(input)
    ):
        return None
    size_bytes = input.numel() * input.element_size()
    if (
        input.data_ptr() < residual.data_ptr() + size_bytes
        and residual.data_ptr() < input.data_ptr() + size_bytes
    ):
        return None
    m = input.shape[0]
    q = torch.empty((m, 2560), device=input.device, dtype=torch.uint8)
    scales = torch.empty(
        ((m + 127) // 128 * 128, 320), device=input.device, dtype=torch.uint8
    )
    _norm_module().run(
        input,
        residual,
        weight,
        scale,
        q,
        scales,
        input,
        communicator.trt_pointers,
        communicator.rank,
        eps,
    )
    return q, scales


def silu_fp4_quant(input, scale, mask):
    """Return dense packed data/scales, or None to retain the old producer.

    A single expert is a dense matrix. Undo only the grouped GEMM views: the
    physical 128x4 scale layout already matches the dense FP4 GEMM contract.
    FlashInfer rounds the SiLU product to BF16 before computing block scales.
    """
    if (
        not _FUSED_PRODUCER_ENABLED
        or silu_and_mul_scaled_nvfp4_experts_quantize is None
        or not input.is_cuda
        or input.ndim != 2
        or input.dtype != torch.bfloat16
        or input.shape[0] == 0
        or input.numel() >= 2**31
        or input.shape[1] not in (17408, 34816)
        or not input.is_contiguous()
        or get_sm(input.device) != (12, 0)
        or scale.device != input.device
        or scale.dtype != torch.float32
        or scale.numel() != 1
        or not scale.is_contiguous()
        or mask is None
        or mask.device != input.device
        or mask.dtype != torch.int32
        or mask.shape != (1,)
    ):
        return None
    m, two_k = input.shape
    k = two_k // 2
    q, scales = silu_and_mul_scaled_nvfp4_experts_quantize(
        input.unsqueeze(0), mask, scale.reshape(1)
    )
    # FlashInfer 0.6.9 grouped output; retain the original producer if changed.
    if q.shape != (m, k // 2, 1) or scales.shape != (
        32,
        4,
        (m + 127) // 128,
        4,
        (k // 16 + 3) // 4,
        1,
    ):
        return None
    q = q.permute(2, 0, 1).reshape(m, k // 2)
    scales = scales.permute(5, 2, 4, 0, 1, 3).reshape(-1, k // 16)
    return q, scales.view(torch.uint8)
