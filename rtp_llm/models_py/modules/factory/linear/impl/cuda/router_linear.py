"""Opt-in BF16 gate projections with a fixed GEMM row count."""

from typing import Optional

import torch
from torch.nn import functional as F


def _maybe_fixed64_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], outputs: int
) -> Optional[torch.Tensor]:
    """Use the validated 64-row GEMM for a supported Qwen3.5 gate.

    cuBLAS can change BF16 rounding once M exceeds 64. Keep smaller batches
    and large prefill on their existing path; the caller bounds decode sizes.
    Only metadata is inspected here, so this also works during graph capture.
    """
    if not (
        x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.ndim == weight.ndim == 2
        and x.shape[1] == 4096
        and weight.shape == (outputs, 4096)
        and 64 < x.shape[0] <= 256
        and (
            bias is None
            or (
                bias.device == x.device
                and bias.dtype == x.dtype
                and bias.shape == (outputs,)
            )
        )
    ):
        return None

    output = torch.empty((x.shape[0], outputs), dtype=x.dtype, device=x.device)
    for start in range(0, x.shape[0], 64):
        end = min(start + 64, x.shape[0])
        part = x[start:end]
        if end - start < 64:
            padded = torch.zeros((64, 4096), dtype=x.dtype, device=x.device)
            padded[: end - start].copy_(part)
            part = padded
        output[start:end].copy_(F.linear(part, weight, bias)[: end - start])
    return output


def maybe_bf16_router_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """512-expert MoE router; unsupported sizes retain the original projection."""
    return _maybe_fixed64_linear(x, weight, bias, 512)


def maybe_bf16_gdn_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
) -> Optional[torch.Tensor]:
    """Qwen3.5 GDN b/a projection for 64 value heads (128 output gates)."""
    return _maybe_fixed64_linear(x, weight, bias, 128)
