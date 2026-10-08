"""Native recurrent K3 nextn layer; logits and recurrence have distinct outputs."""

from math import gcd

import torch

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.kimi_k3.attention import linear
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3MtpModel(KimiK3Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights = self.weight.weights[0]
        # The BF16 MLA prefix path uses skip-head-mid KV-up. Keep its source
        # [in, out] weight contiguous so the linear's [out, in] transpose has
        # the layout required by the packed GEMM, including after Host reuse.
        kv_up = weights[W.mla_kv_b_w]
        if kv_up.dtype == torch.bfloat16 and not kv_up.is_contiguous():
            weights[W.mla_kv_b_w] = kv_up.contiguous()
        eps = self.config.layernorm_eps
        self.enorm = RMSNorm(weights["kimi_k3.mtp.enorm"], eps)
        self.hnorm = RMSNorm(weights["kimi_k3.mtp.hnorm"], eps)
        self.eh_proj = linear(weights, "kimi_k3.mtp.eh_proj", self.py_hw_kernel_config)

    def initialize(self, init_resource):
        ready = super().initialize(init_resource)
        if self.tp_size > 1 and init_resource.is_decode_role:
            from rtp_llm.models_py.modules.kimi_k3.mtp_collectives import (
                KimiK3MtpBf16Collectives,
            )

            q = max(int(self.config.gen_num_per_cycle) + 1, 1)
            batch = max(
                self._max_generate_batch_size,
                int(getattr(init_resource, "max_decode_graph_batch_size", 1)),
            )
            request_alignment = self.tp_size // gcd(self.tp_size, q)
            batch = (
                (batch + request_alignment - 1) // request_alignment * request_alignment
            )
            attention = self.layers[0].attention
            attention._mtp_bf16_collectives = KimiK3MtpBf16Collectives(
                attention.input.weight.device,
                max_tokens=batch * q,
                hidden_size=self.config.hidden_size,
            )
        return ready

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
            raise ValueError(
                "Native K3 MTP requires a hidden state for every physical token"
            )
        # The embedding gathers hidden shards across TP. Project only this
        # rank's token rows after that gather, before entering the SP layer.
        embedded = self.embed_tokens(inputs.input_ids)
        embedded = embedded.narrow(0, local_start, local_rows).contiguous()
        local_positions = positions.narrow(0, local_start, local_rows)
        previous_h = inputs.input_hiddens.narrow(0, local_start, local_rows)
        embedded = embedded * (local_positions.reshape(-1, 1) != 0)
        return self.eh_proj(
            torch.cat(
                (self.enorm(embedded), self.hnorm(previous_h.contiguous())), dim=-1
            )
        )

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._project_local_mtp_input(inputs)
        recurrent = self._forward_layers(
            hidden, inputs, fmha_impl, sequence_parallel_input=True
        )
        return PyModelOutputs(self.norm(recurrent), recurrent)
