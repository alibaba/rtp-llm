"""MTP (Multi-Token Prediction) block for new-loader models."""

import torch

from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.layers.norm import RMSNorm
from rtp_llm.models_py.module_base import RtpModule


class MTPBlock(RtpModule):
    """Multi-Token Prediction block.

    Flow: embed_norm(inputs_embeds) + hidden_norm(last_hidden_states) -> concat -> fc -> output.
    reverse_concat=True: [h_norm, e_norm] (DeepSeek V3 style)
    reverse_concat=False: [e_norm, h_norm] (Qwen style)
    """

    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float = 1e-6,
        reverse_concat: bool = False,
        bias: bool = False,
        params_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.reverse_concat = reverse_concat
        self.e_norm = RMSNorm(hidden_size, eps=rms_norm_eps, params_dtype=params_dtype)
        self.h_norm = RMSNorm(hidden_size, eps=rms_norm_eps, params_dtype=params_dtype)
        self.fc = ColumnParallelLinear(
            input_size=hidden_size * 2,
            output_size=hidden_size,
            tp_size=1,
            tp_rank=0,
            quant_config=None,
            prefix="fc",
            bias=bias,
            params_dtype=params_dtype,
        )

    def forward(
        self, inputs_embeds: torch.Tensor, last_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        e_norm = self.e_norm(inputs_embeds)
        h_norm = self.h_norm(last_hidden_states)
        if self.reverse_concat:
            concat = torch.cat([h_norm, e_norm], dim=-1)
        else:
            concat = torch.cat([e_norm, h_norm], dim=-1)
        return self.fc(concat)
