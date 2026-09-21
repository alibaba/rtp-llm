"""Native recurrent K3 nextn layer; logits and recurrence have distinct outputs."""

import torch
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.kimi_k3.attention import linear
from rtp_llm.ops.compute_ops import PyModelOutputs


class KimiK3MtpModel(KimiK3Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights = self.weight.weights[0]
        eps = self.config.layernorm_eps
        self.enorm = RMSNorm(weights["kimi_k3.mtp.enorm"], eps)
        self.hnorm = RMSNorm(weights["kimi_k3.mtp.hnorm"], eps)
        self.eh_proj = linear(weights, "kimi_k3.mtp.eh_proj", self.py_hw_kernel_config)

    def forward(self, inputs, fmha_impl=None):
        embedded = self.embed_tokens(inputs.input_ids)
        positions = inputs.combo_position_ids
        if positions is None or positions.numel() != embedded.shape[0]:
            raise ValueError(
                "Native K3 MTP requires a position for every physical token"
            )
        embedded = embedded * (positions.reshape(-1, 1) != 0)
        hidden = self.eh_proj(
            torch.cat((self.enorm(embedded), self.hnorm(inputs.input_hiddens)), dim=-1)
        )
        recurrent = self._forward_layers(hidden, inputs, fmha_impl)
        return PyModelOutputs(self.norm(recurrent), recurrent)
