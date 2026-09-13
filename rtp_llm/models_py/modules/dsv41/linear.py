"""Explicit CUDA13/SM100 FP8 block32 linears for V4.1 components.

Activations cross the official BF16 -> group32 FP8/UE8M0 boundary. The
checkpoint's two-dimensional 32x32 weight scales remain exact FP32 values
when supplied to DeepGEMM with its raw-scale recipe.
"""

import logging
import os
from functools import partial

import torch
from torch import nn


def is_supported(values: torch.Tensor) -> bool:
    return (
        values.is_cuda
        and torch.version.cuda is not None
        and torch.version.cuda.split(".")[0] == "13"
        and torch.cuda.get_device_capability(values.device)[0] == 10
    )


def _require_execution(values: torch.Tensor) -> None:
    if os.environ.get("DSV41_BLOCK32_LINEAR", "0") != "1":
        raise RuntimeError("V4.1 block32 linear requires DSV41_BLOCK32_LINEAR=1")
    if not is_supported(values):
        raise RuntimeError("V4.1 block32 linear requires CUDA13 on a Blackwell GPU")
    if torch.is_autocast_enabled():
        raise RuntimeError("V4.1 block32 activation quantization requires autocast off")


def quantize_block32(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return E4M3 values and exact power-of-two FP32 scales for [M,K]."""
    _require_execution(values)
    if (
        values.ndim != 2
        or values.dtype != torch.bfloat16
        or not values.is_contiguous()
        or values.shape[1] == 0
        or values.shape[1] % 32
    ):
        raise ValueError("V4.1 activations must be contiguous BF16 [M,K] with K%32=0")
    from rtp_llm.models_py.modules.dsv41._linear_triton import quantize_block32_kernel

    rows, columns = values.shape
    encoded = torch.empty_like(values, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (rows, columns // 32), dtype=torch.float32, device=values.device
    )
    if rows:
        valid = torch.empty_like(scales, dtype=torch.bool)
        quantize_block32_kernel[(rows, columns // 32)](
            values, encoded, scales, valid, K=columns, num_warps=1
        )
        torch._assert_async(valid.all(), "nonfinite V4.1 block32 activation")
    return encoded, scales


class V41Block32Linear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        _require_execution(weight)
        if (
            weight.ndim != 2
            or weight.dtype != torch.float8_e4m3fn
            or not weight.is_contiguous()
            or min(weight.shape) <= 0
            or weight.shape[0] % 32
            or weight.shape[1] % 32
        ):
            raise ValueError(
                "V4.1 dense weight must be contiguous E4M3 [N,K], N/K%32=0"
            )
        self.out_features, self.in_features = weight.shape
        if (
            scale.shape != (self.out_features // 32, self.in_features // 32)
            or scale.dtype != torch.float8_e8m0fnu
            or scale.device != weight.device
            or not scale.is_contiguous()
        ):
            raise ValueError(
                "V4.1 dense scales must be the checkpoint UE8M0 32x32 grid"
            )
        scales = scale.float()
        torch._assert_async(
            (torch.isfinite(scales) & (scales > 0)).all(),
            "V4.1 checkpoint contains invalid UE8M0 scales",
        )
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scales)

    @torch.inference_mode()
    def forward(self, values: torch.Tensor, *, out=None) -> torch.Tensor:
        _require_execution(values)
        if (
            values.ndim < 2
            or values.shape[-1] != self.in_features
            or values.dtype != torch.bfloat16
            or values.device != self.weight.device
            or not values.is_contiguous()
        ):
            raise ValueError(
                "V4.1 linear input must be contiguous BF16 on its weight GPU"
            )
        shape = (*values.shape[:-1], self.out_features)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=values.device)
        elif (
            out.shape != shape
            or out.dtype != torch.bfloat16
            or out.device != values.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "V4.1 linear output shape, dtype, device or layout mismatch"
            )
        rows = values.numel() // self.in_features
        if rows:
            import deep_gemm

            encoded, scales = quantize_block32(values.view(rows, self.in_features))
            deep_gemm.fp8_gemm_nt(
                (encoded, scales),
                (self.weight, self.weight_scale),
                out.view(rows, self.out_features),
                recipe=(1, 32, 32),
                disable_ue8m0_cast=False,
            )
        return out


@torch.inference_mode()
def warmup_block32_linears(model: nn.Module, *, max_rows: int) -> None:
    """Prepare each distinct V4.1 dense shape using its actual raw32 path."""
    from rtp_llm.utils.warmup import model_warm_up_enabled

    if not model_warm_up_enabled():
        return
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("V4.1 dense warmup requires a positive row budget")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("V4.1 dense warmup cannot run inside CUDA Graph capture")
    from rtp_llm.models_py.modules.dsv4 import dsv4_kernel_jit_warmup as warmup

    linears = {}
    for module in model.modules():
        if isinstance(module, V41Block32Linear):
            key = (module.weight.device, module.out_features, module.in_features)
            linears.setdefault(key, module)

    def prepare():
        for (device, n, k), linear in linears.items():
            grid = warmup._generate_dense_gemm_warmup_m_grid(
                max_m=max_rows,
                n_value=n,
                k_value=k,
                kind="fp8",
                num_sms=warmup._get_deep_gemm_num_sms(device),
                uses_deepjit=True,
            )
            logging.info(
                "[V41 Block32] startup dense n=%d k=%d max_rows=%d device=%s rows=%s",
                n,
                k,
                max_rows,
                device,
                grid,
            )
            for rows in grid:
                values = torch.ones((rows, k), dtype=torch.bfloat16, device=device)
                warmup._run_deepgemm_warmup_launch_with_retry(
                    "V41 Block32",
                    f"n={n} k={k} rows={rows}",
                    partial(linear, values),
                    device=device,
                )
            torch.cuda.synchronize(device)

    warmup._run_deepgemm_warmup_launches_serialized("V41 Block32", prepare)
