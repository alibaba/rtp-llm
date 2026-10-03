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

    def _project_local_mtp_input(self, inputs):
        physical_rows = inputs.input_ids.shape[0]
        if physical_rows % self.tp_size:
            raise ValueError("Native K3 MTP requires physical token padding for SP")
        local_rows = physical_rows // self.tp_size
        local_start = self.tp_rank * local_rows
        positions = inputs.combo_position_ids
        if positions is None or positions.numel() != physical_rows:
            raise ValueError(
                "Native K3 MTP requires a position for every physical token"
            )
        if inputs.input_hiddens.shape[0] != physical_rows:
            raise ValueError("Native K3 MTP requires a hidden state for every physical token")
        # The embedding gathers hidden shards across TP. Project only this
        # rank's token rows after that gather, before entering the SP layer.
        embedded = self.embed_tokens(inputs.input_ids)
        embedded = embedded.narrow(0, local_start, local_rows).contiguous()
        local_positions = positions.narrow(0, local_start, local_rows)
        previous_h = inputs.input_hiddens.narrow(0, local_start, local_rows)
        embedded = embedded * (local_positions.reshape(-1, 1) != 0)
        return self.eh_proj(
            torch.cat((self.enorm(embedded), self.hnorm(previous_h.contiguous())), dim=-1)
        )

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._project_local_mtp_input(inputs)
        recurrent = self._forward_layers(
            hidden, inputs, fmha_impl, sequence_parallel_input=True
        )
        return PyModelOutputs(self.norm(recurrent), recurrent)
