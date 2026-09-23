"""K3's BF16 projections with FP32 intermediate reductions."""

from math import prod

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.ops.compute_ops import rtp_llm_ops


def bf16_linear(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply an [out, in] weight without changing PyTorch's global math policy."""
    if input.is_cuda and input.dtype == weight.dtype == torch.bfloat16:
        shape = (*input.shape[:-1], weight.shape[0])
        rows = prod(input.shape[:-1])
        result = rtp_llm_ops.cublas_gemm_bf16_fp32_accum(
            input.reshape(rows, input.shape[-1]), weight
        )
        return result.reshape(shape)
    return F.linear(input, weight)


class KimiK3Bf16Linear(CudaF16Linear):
    """Explicit K3 selection; deliberately absent from the public registry."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            raise ValueError("K3 BF16 projections require bias-free weights")
        return bf16_linear(input, self.weight)
