import torch
from torch import nn

from rtp_llm.ops import ActivationType


def linear(input: torch.Tensor, weight: torch.Tensor):
    return torch.nn.functional.linear(input, weight.T)


class DenseMLP(nn.Module):
    def __init__(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
        down: torch.Tensor,
        activation_type: ActivationType,
    ):
        super().__init__()
        self.gate = gate
        self.up = up
        self.down = down
        if activation_type == ActivationType.Swiglu:
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    def forward(self, x: torch.Tensor):
        gate_output = linear(x, self.gate)
        up_output = linear(x, self.up)
        activated = self.act_fn(gate_output)
        product = activated * up_output
        down_output = linear(product, self.down)
        return down_output
