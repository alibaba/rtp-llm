import torch

from rtp_llm.models_py.modules.base.common.activation import SiluAndMulBase


class FusedSiluAndMul(SiluAndMulBase):
    def forward(self, gate_up: torch.Tensor) -> torch.Tensor:
        d = gate_up.shape[-1] // 2
        gate = torch.nn.functional.silu(gate_up[..., :d])
        up = gate_up[..., d:]
        return gate * up
