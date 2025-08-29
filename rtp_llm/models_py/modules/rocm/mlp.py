from typing import Dict
import torch
from libth_transformer import rtp_llm_ops
from torch import nn
import aiter

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Linear
from rtp_llm.utils.model_weight import W

class DenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()

        # 拆分gate_proj和up_proj的权重
        ffn13 = weights[W.ffn_w13]
        gate_w, up_w = torch.chunk(ffn13, 2, dim=-1)

        # 拆分gate_proj和up_proj的bias，如果有的话
        ffn13_bias = weights.get(W.ffn_b13, None)
        if ffn13_bias is not None:
            gate_b, up_b = torch.chunk(ffn13_bias, 2, dim=-1)
        else:
            gate_b = None
            up_b = None

        self.gate_proj = Linear(gate_w, gate_b)
        self.up_proj = Linear(up_w, up_b)
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

        if config.activation_type == "SiGLU":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation type: {config.activation_type}")

    def forward(self, x: torch.Tensor):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class FusedSiluActDenseMLP(nn.Module):
    def __init__(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ):
        super().__init__()
        assert (
            config.activation_type == "SiGLU"
        ), "FusedSiluActDenseMLP only supports SiGLU activation"
        self.gate_up_proj = Linear(weights[W.ffn_w13], weights.get(W.ffn_b13, None))
        self.down_proj = Linear(weights[W.ffn_w2], weights.get(W.ffn_b2, None))

    def forward(self, x: torch.Tensor):
        gate_up = self.gate_up_proj(x)

        d = gate_up.shape[-1] // 2
        output_shape = gate_up.shape[:-1] + (d,)
        output = torch.empty(output_shape, dtype=gate_up.dtype, device=gate_up.device)
        aiter.silu_and_mul(output, gate_up)
        down_proj = self.down_proj(output)
        return down_proj
