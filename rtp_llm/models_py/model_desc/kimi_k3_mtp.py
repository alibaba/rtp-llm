"""Native recurrent K3 nextn layer; logits and recurrence have distinct outputs."""

from math import gcd

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.model_desc.block_map import get_primary_attention_inputs
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3MtpModel(KimiK3Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        weights = self.weight.weights[0]
        # The BF16 MLA prefix path uses skip-head-mid KV-up. Keep its source
        # [in, out] weight contiguous so the linear's [out, in] transpose has
        # the layout required by the packed GEMM, including after Host reuse.
        weights[W.mla_kv_b_w] = weights[W.mla_kv_b_w].contiguous()
        eps = self.config.layernorm_eps
        self.enorm = RMSNorm(weights["kimi_k3.mtp.enorm"], eps)
        self.hnorm = RMSNorm(weights["kimi_k3.mtp.hnorm"], eps)
        self.eh_proj = LinearFactory.create_linear_from_weights(
            weights, "kimi_k3.mtp.eh_proj", hw_kernel_config=self.py_hw_kernel_config
        )

    def initialize(self, init_resource):
        ready = super().initialize(init_resource)
        if self.tp_size > 1 and init_resource.is_decode_role:
            from rtp_llm.models_py.distributed.collective_torch import (
                KimiK3MtpBf16Collectives,
            )

            q = int(self.config.gen_num_per_cycle) + 1
            batch = self._max_generate_batch_size
            request_alignment = self.tp_size // gcd(self.tp_size, q)
            batch = (
                (batch + request_alignment - 1) // request_alignment * request_alignment
            )
            attention = self.layers[0].attention
            # Proposal and update runners initialize this shared model.
            # Captured kernels retain raw pointers into the first workspace.
            if attention._mtp_bf16_collectives is None:
                attention._mtp_bf16_collectives = KimiK3MtpBf16Collectives(
                    attention.input.weight.device,
                    max_tokens=batch * q,
                    hidden_size=self.config.hidden_size,
                )
        return ready

    def _gather_modeling_boundary(self, local_input):
        if self.tp_size == 1:
            return local_input
        collectives = self.layers[0].attention._mtp_bf16_collectives
        if collectives is not None:
            return collectives.all_gather_modeling_boundary(local_input)
        return all_gather(local_input, Group.TP)

    def _local_mtp_embedding_mask(self, inputs, local_start, local_rows):
        """K3 suppresses absolute source-position zero before embedding RMSNorm.

        NoPE inputs omit explicit positions. Reconstruct their local rows from
        graph-stable device metadata; never read device lengths on the host.
        """
        rows = inputs.input_ids.shape[0]
        device = inputs.input_ids.device
        attn = get_primary_attention_inputs(inputs, self.kv_cache)
        positions = inputs.combo_position_ids
        if positions is not None and positions.numel():
            if positions.ndim != 1 or positions.numel() != rows or positions.device != device:
                raise ValueError("K3 MTP requires one source position per physical token")
            keep = positions.narrow(0, local_start, local_rows) != 0
        elif attn.is_prefill or attn.is_target_verify or attn.is_mtp_draft_update:
            offsets = attn.cu_seqlens_device
            prefixes = attn.prefix_lengths_device
            if (
                offsets is None or prefixes is None
                or offsets.ndim != 1 or prefixes.ndim != 1
                or prefixes.numel() == 0 or offsets.numel() != prefixes.numel() + 1
                or offsets.device != device or prefixes.device != device
                or offsets.dtype not in (torch.int32, torch.int64)
                or prefixes.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("K3 NoPE MTP requires device query offsets and prefix lengths")
            global_rows = torch.arange(local_start, local_start + local_rows, device=device)
            # Graph buckets can have repeated offsets for empty request slots.
            # Clamp only the gather index; validity remains a device-side mask.
            requests = torch.searchsorted(offsets[1:], global_rows, right=True)
            requests = requests.clamp(max=prefixes.numel() - 1)
            positions = prefixes[requests] + global_rows - offsets[requests]
            keep = (positions != 0) & (global_rows < offsets[-1])
        else:
            lengths = attn.sequence_lengths_plus_1_device
            if (
                lengths is None or lengths.ndim != 1 or lengths.numel() != rows
                or lengths.device != device
                or lengths.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("K3 NoPE MTP Decode requires one device cache length per token")
            keep = lengths.narrow(0, local_start, local_rows) != 1
        valid = attn.valid_token_mask
        if valid is not None:
            if valid.ndim != 1 or valid.numel() != rows or valid.device != device or valid.dtype != torch.bool:
                raise ValueError("K3 MTP requires one device validity entry per physical token")
            keep = keep & valid.narrow(0, local_start, local_rows)
        return keep

    def _project_local_mtp_input(self, inputs):
        physical_rows = inputs.input_ids.shape[0]
        local_rows = physical_rows // self.tp_size
        local_start = self.tp_rank * local_rows
        # The embedding gathers hidden shards across TP. Project only this
        # rank's token rows after that gather, before entering the SP layer.
        embedded = self.embed_tokens(
            inputs.input_ids, tp_gather=self._gather_modeling_boundary
        )
        embedded = embedded.narrow(0, local_start, local_rows).contiguous()
        keep_embedding = self._local_mtp_embedding_mask(inputs, local_start, local_rows)
        embedded = embedded.masked_fill(~keep_embedding[:, None], 0)
        previous_h = inputs.input_hiddens.narrow(0, local_start, local_rows)
        return self.eh_proj(
            torch.cat(
                (self.enorm(embedded), self.hnorm(previous_h.contiguous())), dim=-1
            )
        )

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._project_local_mtp_input(inputs)
        recurrent = self._forward_layers(
            hidden, inputs, fmha_impl, sequence_parallel_input=True, gather_output=False
        )
        recurrent = self._gather_modeling_boundary(recurrent)
        return PyModelOutputs(self.norm(recurrent), recurrent)
