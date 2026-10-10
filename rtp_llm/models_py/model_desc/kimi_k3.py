"""Kimi K3 text target. Cache ownership remains in RTP's tagged cache groups."""

import logging
import os
from collections.abc import Mapping
from functools import partial
from math import gcd

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    all_gather,
    reduce_scatter_bf16 as reduce_scatter,
)
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.modules import KdaMetadata
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import Embedding, RMSNorm
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.kimi_k3.attention import (
    KimiK3KDA,
    KimiK3MLA,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    prepare_paged_short_conv_metadata,
)
from rtp_llm.models_py.triton_kernels.moe.activation import situ_and_mul
from rtp_llm.ops import HybridAttentionType, RoleType
from rtp_llm.ops.compute_ops import PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3DenseMLP(nn.Module):
    """Replicated dense weights applied to the local sequence-parallel tokens."""

    fuses_layer_residual = False

    def __init__(self, config, parallelism, weights, hardware):
        super().__init__()
        self.gate_up = None
        decode_bf16 = (
            parallelism.role_type == RoleType.DECODE
            and weights[W.ffn_w1].dtype == weights[W.ffn_w3].dtype == torch.bfloat16
        )
        tp_dense = decode_bf16 and parallelism.tp_size > 1
        if tp_dense:
            gate, up, down = (weights[W.ffn_w1], weights[W.ffn_w3], weights[W.ffn_w2])
            width = gate.shape[1] // parallelism.tp_size
            start = parallelism.tp_rank * width
            stop = start + width
            weights[W.ffn_w1] = gate[:, start:stop].contiguous()
            weights[W.ffn_w3] = up[:, start:stop].contiguous()
            weights[W.ffn_w2] = down[start:stop, :].contiguous()
        self.input_transition = (
            partial(all_gather, group=Group.TP) if tp_dense else lambda value: value
        )
        self.output_transition = (
            partial(reduce_scatter, group=Group.TP) if tp_dense else lambda value: value
        )
        if (
            parallelism.role_type != RoleType.DECODE
            and weights[W.ffn_w1].dtype == weights[W.ffn_w3].dtype == torch.bfloat16
        ):
            gate_weight, up_weight = weights[W.ffn_w1], weights[W.ffn_w3]
            width = gate_weight.shape[1]
            merged = torch.cat((gate_weight.T, up_weight.T), dim=0)
            self.gate_up = CudaF16Linear(merged.T)
            # ModelWeights retains these logical keys. Keep them as views into
            # the merged GEMM weight so the original storages can be released.
            weights[W.ffn_w1] = merged.narrow(0, 0, width).T
            weights[W.ffn_w3] = merged.narrow(0, width, width).T
        else:
            self.gate = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w1, hw_kernel_config=hardware
            )
            self.up = LinearFactory.create_linear_from_weights(
                weights, W.ffn_w3, hw_kernel_config=hardware
            )
        self.down = LinearFactory.create_linear_from_weights(
            weights, W.ffn_w2, hw_kernel_config=hardware
        )
        self.beta = config.k3_runtime_config.activation_situ_beta
        self.linear_beta = config.k3_runtime_config.activation_situ_linear_beta

    def forward(self, hidden, valid_mask=None):
        hidden = self.input_transition(hidden)
        if self.gate_up is not None:
            gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        else:
            gate = self.gate(hidden)
            up = self.up(hidden)
        activated = situ_and_mul(
            gate,
            up,
            self.beta,
            self.linear_beta,
            inplace=self.gate_up is None,
        )
        del gate, up
        return self.output_transition(self.down(activated))


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
        self.apply_mlp = (
            self._apply_fused_shared_mlp
            if self.mlp.fuses_layer_residual
            else self._apply_regular_mlp
        )

    def _apply_regular_mlp(self, hidden, mlp_input, valid_mask):
        return hidden + self.mlp(mlp_input, valid_mask)

    def _apply_fused_shared_mlp(self, hidden, mlp_input, valid_mask):
        return self.mlp(mlp_input, valid_mask, residual=hidden)

    def forward(
        self, hidden, anchors, fmha, cache, attention_inputs, metadata, valid_mask
    ):
        if self.block_size:
            previous = (self.index + self.block_size - 1) // self.block_size
            writes = self.index % self.block_size == 0
            fp8_ag = self.attention._fp8_collective
            produce_fp8 = (
                fp8_ag is not None
                and fp8_ag.enable_ag
                and hidden.is_cuda
                and hidden.shape[0] >= 4096
                and fp8_ag.can_run_ag(hidden.shape[0])
                and attention_inputs.is_prefill
                and not metadata.is_target_verify
                and not attention_inputs.is_mtp_draft_update
                and not torch.cuda.is_current_stream_capturing()
            )
            residual = (
                self.attention_residual.forward_fp8
                if produce_fp8
                else self.attention_residual
            )
            attn_input = residual(
                hidden,
                anchors,
                num_blocks=previous,
                block_write_idx=previous if writes else -1,
                output_norm_weight=self.attention_norm.weight,
                output_norm_eps=self.attention_norm.variance_epsilon,
                metadata=metadata,
            )
            attended = self.attention(
                attn_input, fmha, cache, attention_inputs, metadata
            )
            hidden = attended if writes else hidden + attended
            mlp_input = self.mlp_residual(
                hidden,
                anchors,
                num_blocks=previous + int(writes),
                output_norm_weight=self.mlp_norm.weight,
                output_norm_eps=self.mlp_norm.variance_epsilon,
            )
        else:
            hidden = hidden + self.attention(
                self.attention_norm(hidden), fmha, cache, attention_inputs, metadata
            )
            mlp_input = self.mlp_norm(hidden)
        return self.apply_mlp(hidden, mlp_input, valid_mask)


class KimiK3Model(GptModelBase):
    requires_sequence_parallel_padding = True

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
        self.tp_size, self.tp_rank = (
            parallelism_config.tp_size,
            parallelism_config.tp_rank,
        )
        self.fp8_collective_max_tokens = int(
            os.environ.get("RTP_LLM_FP8_COLLECTIVE_MAX_TOKENS", "65536")
        )
        self._max_generate_batch_size = int(max_generate_batch_size)
        # The scheduler bound is global; SP routes only the local token shard.
        global_prefill = model_config.moe_prefill_max_tokens_per_rank
        if global_prefill is None:
            global_prefill = model_config.max_seq_len
        prefill_capacity = (int(global_prefill) + self.tp_size - 1) // self.tp_size
        decode_capacity = 1
        for width in (1, int(model_config.gen_num_per_cycle) + 1):
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
        self._fp8_collective = None
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

    def initialize(self, init_resource) -> bool:
        ready = super().initialize(init_resource)
        if self._fp8_collective is not None:
            return ready
        # Native MTP subclasses this model but keeps its projections in BF16.
        if type(self) is not KimiK3Model or self.tp_size == 1:
            return ready
        decode_staging = self.parallelism_config.role_type == RoleType.DECODE
        enable_ag = os.environ.get("RTP_LLM_FP8_AG_GEMM", "0") == "1"
        enable_rs = os.environ.get("RTP_LLM_FP8_GEMM_RS", "0") == "1"
        if not (enable_ag or enable_rs):
            return ready
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion import (
            Fp8CollectiveProjection,
        )

        device = self.layers[0].attention.input.weight.device
        max_m = self.fp8_collective_max_tokens
        if decode_staging:
            rows = self._max_generate_batch_size * (
                int(self.config.gen_num_per_cycle) + 1
            )
            max_m = (rows + self.tp_size - 1) // self.tp_size * self.tp_size
        self._fp8_collective = Fp8CollectiveProjection(
            _get_group(Group.TP),
            device,
            max_m=max_m,
            hidden_size=self.config.hidden_size,
            enable_ag=enable_ag,
            enable_rs=enable_rs,
            decode_staging=decode_staging,
        )
        for layer in self.layers:
            layer.attention._fp8_collective = self._fp8_collective
            if layer.block_size:
                layer.attention_residual.configure_verify_fp8(
                    decode_staging and enable_ag
                )
            if decode_staging and enable_rs and isinstance(layer.mlp, KimiK3DenseMLP):
                layer.mlp.output_transition = self._fp8_collective.reduce_scatter_bf16
        logging.info(
            "FP8 TP projection fusion enabled: ag=%s rs=%s max_m=%d",
            enable_ag,
            enable_rs,
            max_m,
        )
        return ready

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        from rtp_llm.models_py.modules.factory.attention.attn_factory import get_mla_impl
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MlaImpl

        attention = self.config.getAttentionConfigs(
            self.parallelism_config.get_attn_tp_size()
        )

        def create(attention_inputs):
            attention_inputs.headwise_config = getattr(self.config, "headwise_config", None)
            return get_mla_impl(
                attention,
                self.weight,
                attention_inputs,
                fmha_config=self.fmha_config,
                quant_config=self.config.attention_projection_quant_config,
                is_cuda_graph=is_cuda_graph,
                max_seq_len=self.config.max_seq_len,
                parallelism_config=self.parallelism_config,
                cuda_graph_selection_mode=cuda_graph_selection_mode,
                implementations=(KimiK3MlaImpl,),
            )

        tagged = get_attention_inputs_value(inputs)
        if isinstance(tagged, Mapping):
            tags = self._get_fmha_group_tags()
            return {
                tag: create(tagged[tag]) for tag in (tagged if tags is None else tags)
            }
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

    def _forward_layers(
        self,
        hidden,
        inputs,
        fmha_impl,
        sequence_parallel_input=False,
        gather_output=True,
    ):
        if hidden is None:
            # Construct the target embedding in this frame. Passing the
            # full-token tensor from the caller keeps its storage alive until
            # all layers return, even after SP narrows it to local rows.
            # feat/k3_dev releases that storage during the layer loop.
            hidden = self.embed_tokens(inputs.input_ids)
        physical_rows = inputs.input_ids.shape[0]
        local_rows = physical_rows // self.tp_size
        if not sequence_parallel_input:
            hidden = hidden.narrow(
                0, self.tp_rank * local_rows, local_rows
            ).contiguous()
        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        # A device mask must be refreshed at replay; Python logical row counts
        # cannot be captured into the graph. The runner owns this metadata.
        valid_mask = primary.valid_token_mask
        if valid_mask is not None:
            valid_mask = valid_mask.narrow(0, self.tp_rank * local_rows, local_rows)
        paged_conv_meta = None
        # Native MTP is MLA-only. Conv metadata performs host-side sequence
        # inspection and must not run in its prefill CUDA graph.
        if (
            primary.is_prefill
            and not primary.is_target_verify
            and any(
                layer.layer_type == HybridAttentionType.LINEAR for layer in self.layers
            )
        ):
            paged_conv_meta = prepare_paged_short_conv_metadata(
                primary.cu_seqlens, hidden.device
            )
        metadata = KdaMetadata(
            is_target_verify=primary.is_target_verify,
            prefill_paged_conv_meta=paged_conv_meta,
        )
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
        return (
            all_gather(hidden, Group.TP)
            if gather_output and self.tp_size > 1
            else hidden
        )

    def forward(self, inputs, fmha_impl=None):
        return self._forward_single(inputs, fmha_impl)

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._forward_layers(None, inputs, fmha_impl)
        return PyModelOutputs(self.norm(hidden), hidden)
