"""Kimi K3 text target. Cache ownership remains in RTP's tagged cache groups."""

from collections.abc import Mapping
from math import gcd
import logging
import os

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
)
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_attention_inputs_value,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.kimi_linear import KimiLinearMetadata
from rtp_llm.models_py.modules import Embedding, RMSNorm
from rtp_llm.models_py.modules.kimi_k3.collectives import reduce_scatter
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3KDA, KimiK3MLA, linear
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE, situ
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops import HybridAttentionType, RoleType
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3DenseMLP(nn.Module):
    def __init__(self, config, parallelism, weights, hardware):
        super().__init__()
        self.tp_size = parallelism.tp_size
        self.gate = linear(weights, W.ffn_w1, hardware)
        self.up = linear(weights, W.ffn_w3, hardware)
        self.down = linear(weights, W.ffn_w2, hardware)
        self.beta = config.k3_runtime_config.activation_situ_beta
        self.linear_beta = config.k3_runtime_config.activation_situ_linear_beta

    def forward(self, hidden, valid_mask=None):
        full = all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden
        out = self.down(
            situ(
                self.gate(full), self.up(full), self.beta, self.linear_beta,
                inplace=True,
            )
        )
        return reduce_scatter(out, Group.TP) if self.tp_size > 1 else out


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self, config, parallelism, weights, index, moe_config, hardware, moe_capacity
    ):
        super().__init__()
        self.index = index
        self.block_size = config.k3_runtime_config.attn_res_block_size
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[index]
        self.attention_norm = RMSNorm(weights[W.pre_ln_gamma], config.layernorm_eps)
        self.mlp_norm = RMSNorm(weights[W.post_ln_gamma], config.layernorm_eps)
        if self.block_size:
            self.attention_residual = KimiK3AttentionResidual(
                weights[K3W.SELF_ATTN_RES_NORM],
                weights[K3W.SELF_ATTN_RES_PROJ],
                config.layernorm_eps,
            )
            self.mlp_residual = KimiK3AttentionResidual(
                weights[K3W.MLP_RES_NORM],
                weights[K3W.MLP_RES_PROJ],
                config.layernorm_eps,
            )
        self.attention = (
            KimiK3KDA(config, parallelism, weights, hardware)
            if self.layer_type == HybridAttentionType.LINEAR
            else KimiK3MLA(config, parallelism, weights, index, hardware)
        )
        self.mlp = (
            KimiK3LatentMoE(
                config, parallelism, weights, index, moe_config, hardware, moe_capacity
            )
            if index in config.moe_layer_index
            else KimiK3DenseMLP(config, parallelism, weights, hardware)
        )

    def forward(
        self, hidden, anchors, fmha, cache, attention_inputs, metadata, valid_mask
    ):
        if self.block_size:
            previous = (self.index + self.block_size - 1) // self.block_size
            writes = self.index % self.block_size == 0
            attn_input = self.attention_residual(
                hidden, anchors, num_blocks=previous,
                output_norm_weight=self.attention_norm.weight,
                output_norm_eps=self.attention_norm.variance_epsilon,
            )
            if writes:
                anchors[:, previous].copy_(hidden)
            attended = self.attention(
                attn_input, fmha, cache, attention_inputs, metadata
            )
            hidden = attended if writes else hidden + attended
            mlp_input = self.mlp_residual(
                hidden, anchors, num_blocks=previous + int(writes),
                output_norm_weight=self.mlp_norm.weight,
                output_norm_eps=self.mlp_norm.variance_epsilon,
            )
        else:
            hidden = hidden + self.attention(
                self.attention_norm(hidden), fmha, cache, attention_inputs, metadata
            )
            mlp_input = self.mlp_norm(hidden)
        return hidden + self.mlp(mlp_input, valid_mask)


class KimiK3Model(GptModelBase):
    requires_sequence_parallel_padding = True
    requires_token_position_ids = True

    def __init__(
        self,
        model_config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
        moe_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
            device_resource_config,
        )
        if (
            parallelism_config.dp_size != 1
            or parallelism_config.tp_size != parallelism_config.ep_size
        ):
            raise ValueError("K3 supports TP=EP with DP=1")
        if (
            parallelism_config.pp_size != 1
            or parallelism_config.get_attn_tp_size() != parallelism_config.tp_size
        ):
            raise ValueError(
                "K3 migration requires PP=1 and context parallelism disabled"
            )
        if parallelism_config.tp_size > 1 and not parallelism_config.enable_sp:
            raise ValueError("K3 TP execution requires sequence parallelism")
        self.tp_size, self.tp_rank = (
            parallelism_config.tp_size,
            parallelism_config.tp_rank,
        )
        self.chunk_prefill_budget = int(
            os.environ.get("KIMI_K3_PREFILL_CHUNK_TOKENS", "65536")
        )
        if self.chunk_prefill_budget <= 0 or self.chunk_prefill_budget % self.tp_size:
            raise ValueError("K3 chunk budget must be positive and divisible by TP")
        # The scheduler bound is global; SP routes only the local token shard.
        global_prefill = model_config.moe_prefill_max_tokens_per_rank
        if global_prefill is None:
            global_prefill = model_config.max_seq_len
        prefill_capacity = (int(global_prefill) + self.tp_size - 1) // self.tp_size
        decode_capacity = 1
        for width in (1, max(int(model_config.gen_num_per_cycle) + 1, 1)):
            unit = self.tp_size // gcd(self.tp_size, width)
            requests = (max_generate_batch_size + unit - 1) // unit * unit
            decode_capacity = max(decode_capacity, requests * width // self.tp_size)
        moe_capacity = (
            decode_capacity
            if parallelism_config.role_type == RoleType.DECODE
            else max(prefill_capacity, decode_capacity)
        )
        logging.info(
            "K3 SP MoE capacity: global_prefill=%d local_capacity=%d tp=%d",
            global_prefill,
            moe_capacity,
            self.tp_size,
        )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            KimiK3DecoderLayer(
                model_config,
                parallelism_config,
                weights.weights[i],
                i,
                moe_config,
                py_hw_kernel_config,
                moe_capacity,
            )
            for i in range(self.layer_num)
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), model_config.layernorm_eps
        )
        block_size = model_config.k3_runtime_config.attn_res_block_size
        self.num_blocks = (
            (self.layer_num + block_size - 1) // block_size if block_size else 0
        )
        if block_size:
            self.final_residual = KimiK3AttentionResidual(
                weights.get_global_weight(K3W.OUTPUT_ATTN_RES_NORM),
                weights.get_global_weight(K3W.OUTPUT_ATTN_RES_PROJ),
                model_config.layernorm_eps,
            )

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        if primary.is_target_verify or primary.is_mtp_draft_update:
            from rtp_llm.models_py.modules.kimi_k3.mla_verify import KimiK3MlaVerifyImpl

            implementation = KimiK3MlaVerifyImpl
        elif primary.is_prefill:
            from rtp_llm.models_py.modules.kimi_k3.mla_prefill import KimiK3MlaPrefillImpl

            implementation = KimiK3MlaPrefillImpl
        else:
            return super().prepare_fmha_impl(
                inputs, is_cuda_graph, cuda_graph_selection_mode
            )

        def create(attention_inputs):
            return implementation(
                self.config,
                self.parallelism_config,
                self.weight,
                attention_inputs,
                self.fmha_config,
                is_cuda_graph,
            )

        tagged = get_attention_inputs_value(inputs)
        if isinstance(tagged, Mapping):
            tags = self._get_fmha_group_tags()
            return {tag: create(tagged[tag]) for tag in (tagged if tags is None else tags)}
        return create(tagged)

    def _get_fmha_group_tags(self):
        return (
            None
            if self.kv_cache is None
            else get_group_tags_for_layers(
                self.kv_cache,
                (
                    i
                    for i, layer in enumerate(self.layers)
                    if layer.layer_type != HybridAttentionType.LINEAR
                ),
            )
        )

    def _forward_layers(self, hidden, inputs, fmha_impl):
        if hidden.shape[0] % self.tp_size:
            raise ValueError("K3 requires physical token padding before SP execution")
        local_rows = hidden.shape[0] // self.tp_size
        hidden = hidden.narrow(0, self.tp_rank * local_rows, local_rows).contiguous()
        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        # A device mask must be refreshed at replay; Python logical row counts
        # cannot be captured into the graph. The runner owns this metadata.
        valid_mask = getattr(primary, "valid_token_mask", None)
        if valid_mask is not None:
            valid_mask = valid_mask.narrow(0, self.tp_rank * local_rows, local_rows)
        conv_meta = None
        # Native MTP is MLA-only. Conv metadata performs host-side sequence
        # inspection and must not run in its prefill CUDA graph.
        if (
            primary.is_prefill
            and not primary.is_target_verify
            and any(
                layer.layer_type == HybridAttentionType.LINEAR for layer in self.layers
            )
        ):
            conv_meta = prepare_causal_conv1d_metadata(
                query_start_loc=primary.cu_seqlens_device, device=hidden.device
            )
        metadata = KimiLinearMetadata(conv_meta, primary.is_target_verify)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        anchors = hidden.new_empty((local_rows, self.num_blocks, hidden.shape[-1]))
        for index, layer in enumerate(self.layers):
            attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, index
            )
            fmha = (
                None
                if layer.layer_type == HybridAttentionType.LINEAR
                else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, index)
            )
            cache = self.kv_cache.get_layer_cache(index) if self.kv_cache else None
            hidden = layer(
                hidden, anchors, fmha, cache, attention_inputs, metadata, valid_mask
            )
        if self.num_blocks:
            hidden = self.final_residual(hidden, anchors)
        return all_gather(hidden, Group.TP) if self.tp_size > 1 else hidden

    def forward(self, inputs, fmha_impl=None):
        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        if (
            primary.is_prefill
            and not primary.is_target_verify
            and not primary.is_mtp_draft_update
            and inputs.input_ids.shape[0] > self.chunk_prefill_budget
        ):
            from rtp_llm.models_py.modules.kimi_k3.chunk_forward import (
                forward_prefill_chunks,
            )

            return forward_prefill_chunks(self, inputs)
        return self._forward_single(inputs, fmha_impl)

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._forward_layers(
            self.embed_tokens(inputs.input_ids), inputs, fmha_impl
        )
        return PyModelOutputs(self.norm(hidden), hidden)
