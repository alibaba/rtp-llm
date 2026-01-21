"""CUDA-specific activation function implementations."""

import torch

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase
from rtp_llm.ops.compute_ops import rtp_llm_ops


class FusedSiluAndMul(SiluAndMulBase):
    """CUDA implementation of silu_and_mul using rtp_llm_ops."""

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication using CUDA kernel.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        stream_id = torch.cuda.current_stream().cuda_stream
        rtp_llm_ops.silu_and_mul(output, gate_up, stream_id)
        return output
