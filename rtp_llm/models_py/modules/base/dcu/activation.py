"""DCU-specific activation function implementations."""

import torch
#from vllm import _custom_ops as ops
from lightop import fuse_silu_and_mul

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase


class FusedSiluAndMul(SiluAndMulBase):
    """DCU implementation of silu_and_mul using custom op."""

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        """
        Perform SiLU activation and element-wise multiplication using ROCm kernel.

        Args:
            output: Output tensor to write result to
            gate_up: Input tensor with concatenated gate and up projections
        """
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        fuse_silu_and_mul(gate_up, output)
        return output
