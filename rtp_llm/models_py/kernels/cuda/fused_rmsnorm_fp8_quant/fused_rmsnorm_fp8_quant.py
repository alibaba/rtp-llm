from __future__ import annotations

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_max,
    fp8_min,
)

_SUPPORTED_K = {1024, 1536, 2048, 3072, 4096, 7168}


def is_supported(x: torch.Tensor, weight: torch.Tensor, group_size: int = 128) -> bool:
    return (
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.is_cuda
        and weight.dim() == 1
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and weight.numel() == x.shape[-1]
        and weight.device == x.device
        and group_size == 128
        and x.shape[-1] in _SUPPORTED_K
    )


def fused_rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    quant_eps: float = 1e-4,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_supported(x, weight, group_size):
        raise ValueError(
            "unsupported fused_rmsnorm_fp8_quant input: "
            f"x_shape={tuple(x.shape)} x_dtype={x.dtype} "
            f"weight_shape={tuple(weight.shape)} weight_dtype={weight.dtype} "
            f"group_size={group_size}"
        )

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    output_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=output_q.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    rtp_llm_ops.fused_rmsnorm_fp8_quant(
        x,
        weight,
        output_q,
        output_s,
        norm_eps,
        quant_eps,
        fp8_min,
        fp8_max,
    )
    return output_q, output_s


def fused_rmsnorm_bf16_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    norm_eps: float = 1e-6,
    quant_eps: float = 1e-4,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not is_supported(x, weight, group_size):
        raise ValueError(
            "unsupported fused_rmsnorm_bf16_fp8_quant input: "
            f"x_shape={tuple(x.shape)} x_dtype={x.dtype} "
            f"x_stride={tuple(x.stride())} "
            f"weight_shape={tuple(weight.shape)} weight_dtype={weight.dtype} "
            f"weight_stride={tuple(weight.stride())} group_size={group_size}"
        )

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    output_y = torch.empty_like(x, dtype=torch.bfloat16)
    output_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    output_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=output_q.shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    rtp_llm_ops.fused_rmsnorm_bf16_fp8_quant(
        x,
        weight,
        output_y,
        output_q,
        output_s,
        norm_eps,
        quant_eps,
        fp8_min,
        fp8_max,
    )
    return output_y, output_q, output_s
