"""XPU-specific activation function implementations."""
import torch

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase

try:
    from rtp_llm.models_py.modules.base.xpu.vllm_xpu_ops import is_available as _vllm_available
except ImportError:
    _vllm_available = lambda: False


class FusedSiluAndMul(SiluAndMulBase):
    """XPU SiLU-and-mul using vllm-xpu-kernels."""

    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        if _vllm_available():
            torch.ops._C.silu_and_mul(output, gate_up)
        else:
            x, gate = gate_up[..., :d], gate_up[..., d:]
            output.copy_(torch.nn.functional.silu(x) * gate)
        return output
