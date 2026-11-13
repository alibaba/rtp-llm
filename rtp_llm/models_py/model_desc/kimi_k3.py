"""Kimi K3 text target. Cache ownership remains in RTP's tagged cache groups."""

import logging
from collections.abc import Mapping
from functools import partial

import torch
from torch import nn

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.distributed.collective_torch import Group, _get_group, all_gather
from rtp_llm.models_py.distributed.collective_torch import (
    reduce_scatter_bf16 as reduce_scatter,
)
from rtp_llm.models_py.model_desc.block_map import (
    ensure_host_prefill_metadata,
    get_attention_inputs_value,
    get_group_tags_for_layers,
    get_layer_caches_for_groups,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_attention_inputs_for_tag,
    select_fmha_impl_for_group,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    Embedding,
    KdaMetadata,
    MultimodalEmbeddingInjector,
    RMSNorm,
)
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.kimi_k3.attention import KimiK3KDA, KimiK3MLA
from rtp_llm.models_py.modules.kimi_k3.dsa_model_adapter import KimiK31ModelAdapter
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    prepare_paged_short_conv_metadata,
)
from rtp_llm.models_py.triton_kernels.moe.activation import situ_and_mul
from rtp_llm.models_py.utils.module_profiler import profile_scope
from rtp_llm.ops import HybridAttentionType, RoleType
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class KimiK3DenseMLP(nn.Module):
    """Replicated dense weights applied to the local sequence-parallel tokens."""

    fuses_layer_residual = False

    def __init__(self, config, parallelism, weights, hardware):
        super().__init__()
        self.gate_up = None
        decode_bf16 = (
            parallelism is not None
            and parallelism.role_type == RoleType.DECODE
            and weights[W.ffn_w1].dtype == weights[W.ffn_w3].dtype == torch.bfloat16
        )
        tp_dense = decode_bf16 and parallelism.tp_size > 1
        if tp_dense:
            gate, up, down = (weights[W.ffn_w1], weights[W.ffn_w3], weights[W.ffn_w2])
            if (
                gate.shape != up.shape
                or gate.shape[1] != down.shape[0]
                or gate.shape[0] != down.shape[1]
                or gate.shape[1] % parallelism.tp_size
            ):
                raise ValueError(
                    "Decode TP dense MLP has incompatible weight dimensions"
                )
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
            parallelism is None or parallelism.role_type != RoleType.DECODE
        ) and weights[W.ffn_w1].dtype == weights[W.ffn_w3].dtype == torch.bfloat16:
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
        if self.layer_type == HybridAttentionType.LINEAR:
            self.attention = KimiK3KDA(config, parallelism, weights, hardware)
        elif getattr(config.k3_runtime_config, "dsa", None) is not None:
            from rtp_llm.models_py.modules.kimi_k3.dsa import KimiK31DSA

            self.attention = KimiK31DSA(config, parallelism, weights, index, hardware)
        else:
            self.attention = KimiK3MLA(config, parallelism, weights, index, hardware)
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
        self,
        hidden,
        anchors,
        fmha,
        cache,
        attention_inputs,
        metadata,
        valid_mask,
        *,
        mlp_delta=None,
        mlp_shared_delta=None,
        defer_mlp_residual=False,
    ):
        if self.block_size:
            previous = (self.index + self.block_size - 1) // self.block_size
            writes = self.index % self.block_size == 0
            fp8_ag = self.attention._fp8_input_collective
            produce_fp8 = (
                fp8_ag is not None
                and fp8_ag.enable_ag
                and getattr(self.attention, "supports_fp8_activation", False)
                and hidden.is_cuda
                and hidden.shape[0] >= 4096
                and fp8_ag.can_run_ag(hidden.shape[0])
                and attention_inputs.is_prefill
                and not metadata.is_target_verify
                and not attention_inputs.is_mtp_draft_update
                and not torch.cuda.is_current_stream_capturing()
            )
            natural_nvfp4 = getattr(self.attention, "_natural_nvfp4_enabled", False)
            natural_fp8 = (
                (
                    natural_nvfp4
                    or getattr(self.attention, "natural_kv_indexer_input", None)
                    is not None
                )
                and getattr(metadata, "outer_layout", None) is not None
                and attention_inputs.is_prefill
                and not metadata.is_target_verify
                and not attention_inputs.is_mtp_draft_update
                and hidden.is_cuda
                and hidden.shape[0] >= (1 if natural_nvfp4 else 4096)
                and torch.cuda.get_device_capability(hidden.device)[0] == 10
                and not torch.cuda.is_current_stream_capturing()
            )
            natural_nvfp4 = natural_nvfp4 and natural_fp8
            eligible_nvfp4_ag = getattr(self.attention, "eligible_nvfp4_ag", None)
            decode_nvfp4 = (
                getattr(self.attention, "_decode_nvfp4_ag", None) is not None
                and (metadata.is_target_verify or not attention_inputs.is_prefill)
                and not attention_inputs.is_mtp_draft_update
                and hidden.shape[0] > 0
            )
            produce_fp4 = (
                (
                    decode_nvfp4
                    or (
                        eligible_nvfp4_ag is not None
                        and eligible_nvfp4_ag(hidden, attention_inputs, metadata)
                    )
                )
                and hidden.shape[1] == 7168
                and torch.cuda.get_device_capability(hidden.device)[0] == 10
            )
            residual_scope = (
                "attention_residual_nvfp4"
                if produce_fp4 or natural_nvfp4
                else "attention_residual"
            )
            with profile_scope(f"RTP::layers.{self.index}.{residual_scope}"):
                residual = (
                    self.attention_residual.forward_fp8
                    if produce_fp8
                    else self.attention_residual
                )
                attn_input = residual(
                    hidden,
                    anchors,
                    delta=mlp_delta,
                    shared_delta=mlp_shared_delta,
                    num_blocks=previous,
                    block_write_idx=previous if writes else -1,
                    output_norm_weight=self.attention_norm.weight,
                    output_norm_eps=self.attention_norm.variance_epsilon,
                    metadata=metadata,
                    **({"also_quantize": True} if natural_fp8 else {}),
                    **(
                        {"quantize_nvfp4": True, "split_nvfp4_quant": not decode_nvfp4}
                        if produce_fp4 or natural_nvfp4
                        else {}
                    ),
                    **({"rowwise_nvfp4": True} if produce_fp4 else {}),
                    **(
                        {
                            "decode_head_gate_columns": self.attention._decode_head_gate_columns
                        }
                        if decode_nvfp4
                        and getattr(self.attention, "_decode_head_gate_columns", 0)
                        else {}
                    ),
                )
                if natural_nvfp4:
                    attn_input, encoded, packed = attn_input
                elif natural_fp8:
                    attn_input, encoded = attn_input
            with profile_scope(f"RTP::layers.{self.index}.attention"):
                attended = self.attention(
                    attn_input,
                    fmha,
                    cache,
                    attention_inputs,
                    metadata,
                    **({"natural_fp8": encoded} if natural_fp8 else {}),
                    **({"natural_nvfp4": packed} if natural_nvfp4 else {}),
                )
            if natural_fp8:
                del encoded
            if natural_nvfp4:
                del packed
            delta = None
            if writes:
                hidden = attended
            elif (
                (
                    (defer_mlp_residual and metadata.is_target_verify)
                    or (
                        attention_inputs.is_prefill
                        and not metadata.is_target_verify
                        and not attention_inputs.is_mtp_draft_update
                        and metadata.query_layout is not None
                    )
                )
                and hidden.is_cuda
                and hidden.dtype == attended.dtype == torch.bfloat16
                and hidden.shape == attended.shape
                and hidden.is_contiguous()
                and attended.is_contiguous()
            ):
                # AttnRes updates hidden in place before the MLP residual uses it.
                delta = attended
            else:
                hidden = hidden + attended
            with profile_scope(f"RTP::layers.{self.index}.mlp_residual"):
                mlp_input = self.mlp_residual(
                    hidden,
                    anchors,
                    delta=delta,
                    num_blocks=previous + int(writes),
                    output_norm_weight=self.mlp_norm.weight,
                    output_norm_eps=self.mlp_norm.variance_epsilon,
                )
        else:
            with profile_scope(f"RTP::layers.{self.index}.attention"):
                attended = self.attention(
                    self.attention_norm(hidden), fmha, cache, attention_inputs, metadata
                )
                if (
                    (
                        not attention_inputs.is_prefill
                        or attention_inputs.is_mtp_draft_update
                    )
                    and hidden.is_cuda
                    and hidden.dtype
                    == attended.dtype
                    == self.mlp_norm.weight.dtype
                    == torch.bfloat16
                    and hidden.shape == attended.shape
                    and hidden.is_contiguous()
                    and attended.is_contiguous()
                    and self.mlp_norm.weight.is_contiguous()
                ):
                    from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
                        rounded_add_rmsnorm,
                    )

                    hidden, mlp_input = rounded_add_rmsnorm(
                        hidden,
                        attended,
                        self.mlp_norm.weight,
                        self.mlp_norm.variance_epsilon,
                    )
                else:
                    hidden = hidden + attended
                    mlp_input = self.mlp_norm(hidden)
        with profile_scope(f"RTP::layers.{self.index}.mlp"):
            if defer_mlp_residual:
                if self.mlp.fuses_layer_residual:
                    routed, shared = self.mlp(
                        mlp_input, valid_mask, defer_output_add=True
                    )
                    return hidden, routed, shared
                mlp_output = self.mlp(mlp_input, valid_mask)
                if (
                    hidden.dtype == mlp_output.dtype == torch.bfloat16
                    and hidden.shape == mlp_output.shape
                    and hidden.is_contiguous()
                    and mlp_output.is_contiguous()
                ):
                    return hidden, mlp_output, None
                return hidden + mlp_output, None, None
            return self.apply_mlp(hidden, mlp_input, valid_mask)


class KimiK3Model(KimiK31ModelAdapter, GptModelBase):
    requires_sequence_parallel_padding = True

    def __init__(
        self,
        model_config,
        parallelism_config,
        weights,
        max_generate_batch_size,
        fmha_config=None,
        py_hw_kernel_config=None,
        moe_config=None,
        *,
        max_moe_tokens: int,
        prefill_staging_tokens: int | None = None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
        )
        if (
            parallelism_config.ep_size
            != parallelism_config.tp_size * parallelism_config.dp_size
        ):
            raise ValueError("K3 requires EP=TP×DP for the native expert placement")
        if (
            parallelism_config.pp_size != 1
            or parallelism_config.get_attn_tp_size() != parallelism_config.tp_size
        ):
            raise ValueError(
                "K3 migration requires PP=1 and context parallelism disabled"
            )
        self.tp_size, self.tp_rank = (
            parallelism_config.tp_size,
            parallelism_config.tp_rank,
        )
        cp = parallelism_config.prefill_cp_config
        self.prefill_query_cp = (
            getattr(model_config.k3_runtime_config, "dsa", None) is not None
            and parallelism_config.role_type == RoleType.PREFILL
            and cp.is_enabled()
            and cp.preserve_tp_layout
        )
        self.requires_sequence_parallel_padding = not self.prefill_query_cp
        self._max_generate_batch_size = int(max_generate_batch_size)
        # The owner passes the global physical input bound. Projection staging
        # only needs to cover the regular prefill shape.
        self._max_moe_tokens = int(max_moe_tokens)
        if self._max_moe_tokens <= 0 or self._max_moe_tokens % self.tp_size:
            raise ValueError("K3 token capacity must be positive and divisible by TP")
        self._projection_staging_tokens = self._max_moe_tokens
        moe_capacity = self._max_moe_tokens // self.tp_size
        if self.prefill_query_cp and prefill_staging_tokens is not None:
            if prefill_staging_tokens <= 0 or prefill_staging_tokens % self.tp_size:
                raise ValueError(
                    "K3 Prefill staging must be positive and divisible by TP"
                )
            self._projection_staging_tokens = min(
                self._max_moe_tokens, prefill_staging_tokens
            )
            # The FIFO scheduler enforces the padded row budget for DP1.
            if parallelism_config.dp_size == 1:
                moe_capacity = self._projection_staging_tokens // self.tp_size
        logging.info(
            "K3 SP MoE capacity: global_capacity=%d local_capacity=%d tp=%d",
            self._max_moe_tokens,
            moe_capacity,
            self.tp_size,
        )
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
            replicated=self.prefill_query_cp,
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
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
        self._setup_dsa()
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
        self._initialize_mtp_hidden_handoff(init_resource)
        self._initialize_natural_outer(init_resource)
        self._initialize_fp8_collective()
        self._initialize_bf16_collective(init_resource)
        self._initialize_nvfp4_direct_ag(init_resource)
        self._initialize_decode_nvfp4_ag(init_resource)
        from rtp_llm.models_py.modules.kimi_k3.kernel_jit_warmup import (
            warmup_kimi_k3_kernel_jit,
        )

        warmup_kimi_k3_kernel_jit(self, init_resource)
        return ready

    def _initialize_bf16_collective(self, init_resource):
        if self.tp_size == 1 or (
            init_resource.is_decode_role and type(self) is not KimiK3Model
        ):
            return
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.bf16_comm_fusion import (
            Bf16CollectiveProjection,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )

        attentions = [
            layer.attention
            for layer in self.layers
            if not getattr(layer.attention, "prefill_query_cp", False)
        ]
        if not attentions or any(
            not isinstance(projection, CudaF16Linear)
            or projection.weight.dtype != torch.bfloat16
            or not projection.weight.is_cuda
            or projection.bias is not None
            for attention in attentions
            for projection in (attention.input, attention.output)
        ):
            return
        device = attentions[0].input.weight.device
        if torch.cuda.get_device_capability(device) not in ((10, 0), (10, 3)):
            return
        collective = Bf16CollectiveProjection(
            _get_group(Group.TP),
            device,
            max_m=self._projection_staging_tokens,
            hidden_size=self.config.hidden_size,
            output_projections=[attention.output for attention in attentions],
            decode_staging=init_resource.is_decode_role,
        )
        for attention in attentions:
            attention._bf16_collective = collective

    def _initialize_fp8_collective(self) -> None:
        if self._fp8_collective is not None:
            return
        # Native MTP subclasses this model but keeps its projections in BF16.
        if type(self) is not KimiK3Model or self.tp_size == 1:
            return
        quant = getattr(self.config, "attention_projection_quant_config", None)
        grouped_fp8 = quant is not None and quant.get_method() == "FP8_PER_BLOCK"
        decode_staging = self.parallelism_config.role_type == RoleType.DECODE
        fp8_ag_layers, fp8_rs_layers = [], []
        for layer in self.layers:
            attention = layer.attention
            fp8_output = getattr(attention.output, "scale_ue8m0", False)
            fp8_input = getattr(attention.input, "scale_ue8m0", False)
            if grouped_fp8 and (
                not fp8_output
                or (
                    not fp8_input
                    and not getattr(attention.input, "dynamic_input_scale", False)
                )
            ):
                raise ValueError(
                    "FP8 collective fusion requires grouped E4M3 projections"
                )
            if fp8_output:
                fp8_rs_layers.append(layer)
                if fp8_input:
                    fp8_ag_layers.append(layer)
        if not fp8_rs_layers:
            return
        enable_ag = bool(fp8_ag_layers)
        if enable_ag and self.config.k3_runtime_config.attn_res_block_size:
            if self.config.hidden_size != 7168 or any(
                (layer.index + layer.block_size - 1) // layer.block_size > 8
                for layer in fp8_ag_layers
            ):
                raise ValueError(
                    "FP8 AttnRes fusion requires hidden_size=7168 and at most "
                    "8 residual blocks; validate the checkpoint before serving"
                )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_comm_fusion import (
            Fp8CollectiveProjection,
        )

        device = self.layers[0].attention.input.weight.device
        max_m = self._projection_staging_tokens
        self._fp8_collective = Fp8CollectiveProjection(
            _get_group(Group.TP),
            device,
            max_m=max_m,
            hidden_size=self.config.hidden_size,
            enable_ag=enable_ag,
            enable_rs=True,
            decode_staging=decode_staging,
        )
        collective = self._fp8_collective
        for layer in fp8_rs_layers:
            layer.attention._fp8_collective = collective
        for layer in fp8_ag_layers:
            layer.attention._fp8_input_collective = collective
            if layer.block_size:
                layer.attention_residual.configure_verify_fp8(
                    decode_staging
                    and getattr(layer.attention, "supports_fp8_activation", False)
                )
        if decode_staging and grouped_fp8:
            for layer in self.layers:
                if isinstance(layer.mlp, KimiK3DenseMLP):
                    layer.mlp.output_transition = collective.reduce_scatter_bf16
        logging.info(
            "K3 TP projection collectives: fp8_layers=%d decode_push_rs=%s "
            "max_m=%d fp8_ag_layers=%d",
            len(fp8_rs_layers),
            decode_staging,
            max_m,
            len(fp8_ag_layers),
        )
        return

    def prepare_fmha_impl(
        self, inputs, is_cuda_graph=False, cuda_graph_selection_mode=None
    ):
        self._prepared_kda_prefill = None
        if self.dsa_workspace is not None:
            fmha_impl = self._prepare_dsa(inputs, is_cuda_graph)
            if (
                not is_cuda_graph
                and self._query_layout is not None
                and self.kv_cache is not None
            ):
                layout = self._outer_layout or self._query_layout
                for index, layer in enumerate(self.layers):
                    if layer.layer_type != HybridAttentionType.LINEAR:
                        continue
                    cache = self.kv_cache.get_layer_cache(index)
                    if cache is None:
                        continue
                    metadata = KdaMetadata(
                        query_layout=self._query_layout,
                        outer_layout=self._outer_layout,
                    )
                    attention_inputs = select_attention_inputs_for_layer(
                        inputs, self.kv_cache, index
                    )
                    layer.attention.prefill.prepare_cuda_inputs(
                        layout.natural_attention(attention_inputs),
                        cache,
                        metadata,
                    )
                    self._prepared_kda_prefill = (
                        fmha_impl,
                        metadata,
                        str(cache.tag),
                        attention_inputs,
                    )
                    break
            return fmha_impl
        from rtp_llm.models_py.modules.factory.attention.attn_factory import (
            get_mla_impl,
        )
        from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MlaImpl

        attention = self.config.getAttentionConfigs(
            self.parallelism_config.get_attn_tp_size()
        )

        def create(attention_inputs):
            attention_inputs.headwise_config = getattr(
                self.config, "headwise_config", None
            )
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
                prefill_kv_workspace_budget_bytes=self.config.prefill_kv_workspace_budget_bytes,
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

    def _embed_inputs(self, inputs: PyModelInputs, *, tp_gather=None) -> torch.Tensor:
        """Gather token embeddings and inject image features before SP slicing."""
        multimodal_inputs = inputs.multimodal_inputs
        features = multimodal_inputs.multimodal_features
        text_mask = inputs.embedding_inputs.text_tokens_mask
        if features and (
            text_mask is None or text_mask.shape != inputs.input_ids.shape
        ):
            raise ValueError(
                "K3 image features require one text mask entry per input token"
            )
        # Image rows contain cache hashes, so the embedding kernel must skip
        # those rows before looking up vocabulary weights.
        embeddings = self.embed_tokens(
            inputs.input_ids, text_tokens_mask=text_mask, tp_gather=tp_gather
        )
        if not features:
            return embeddings
        # Features span the full hidden dimension: inject after TP gather and
        # before the caller slices token rows for sequence parallelism.
        return self.multimodal_embedding_injector(
            embeddings, features, multimodal_inputs.mm_features_locs
        )

    def _forward_layers(
        self,
        hidden,
        inputs,
        fmha_impl,
        sequence_parallel_input=False,
        gather_output=True,
    ):
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        prepared = getattr(self, "_prepared_kda_prefill", None)
        self._prepared_kda_prefill = None
        if prepared is not None and prepared[0] is not fmha_impl:
            prepared = None
        layout = self._query_layout
        if hidden is None:
            # Construct the target embedding in this frame. Passing the
            # full-token tensor from the caller keeps its storage alive until
            # all layers return, even after SP narrows it to local rows.
            # feat/k3_dev releases that storage during the layer loop.
            hidden = self._embed_inputs(inputs)
        physical_rows = (
            inputs.input_ids.shape[0] if layout is None else layout.physical_rows
        )
        local_rows = physical_rows // self.tp_size
        if layout is not None:
            if hidden.shape[0] != local_rows:
                raise ValueError("CP and SP must share local row ownership")
            hidden.masked_fill_(~layout.local_valid[:, None], 0)
        elif not sequence_parallel_input:
            hidden = hidden.narrow(
                0, self.tp_rank * local_rows, local_rows
            ).contiguous()
        outer_layout = self._outer_layout
        if outer_layout is not None:
            # _embed_inputs has already injected multimodal features. All
            # residuals/anchors and layer inputs become natural at this point.
            with profile_scope("RTP::decoder.zigzag_to_natural"):
                hidden = self._row_exchange.exchange(
                    hidden, self._routing_plan, "to_natural"
                )
        ensure_host_prefill_metadata(inputs)
        primary = get_primary_attention_inputs(inputs, self.kv_cache)
        # A device mask must be refreshed at replay; Python logical row counts
        # cannot be captured into the graph. The runner owns this metadata.
        valid_mask = primary.valid_token_mask
        if layout is not None:
            valid_mask = (
                outer_layout if outer_layout is not None else layout
            ).local_valid
        elif valid_mask is not None:
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
                primary.cu_seqlens if layout is None else layout.cu_host, hidden.device
            )
        if prepared is not None:
            _, metadata, linear_tag, linear_inputs = prepared
            metadata.prefill_paged_conv_meta = paged_conv_meta
        else:
            metadata = KdaMetadata(
                is_target_verify=primary.is_target_verify,
                prefill_paged_conv_meta=paged_conv_meta,
                query_layout=layout,
                outer_layout=outer_layout,
            )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        anchors = hidden.new_empty((local_rows, self.num_blocks, hidden.shape[-1]))
        defer_mlp_residual = (
            self.num_blocks > 0
            and (
                primary.is_target_verify
                or (
                    primary.is_prefill
                    and layout is not None
                    and not primary.is_mtp_draft_update
                )
            )
            and hidden.is_cuda
            and hidden.dtype == torch.bfloat16
            and hidden.shape[-1] == 7168
            and torch.cuda.get_device_capability(hidden.device)[0] == 10
        )
        mlp_delta = None
        mlp_shared_delta = None
        for index, layer in enumerate(self.layers):
            if getattr(layer.attention, "indexer", None) is not None:
                tags = ("full", "indexer_kv")
                attention_inputs = select_attention_inputs_for_tag(
                    get_attention_inputs_value(inputs), "full"
                )
                fmha = {tag: select_fmha_impl_for_group(fmha_impl, tag) for tag in tags}
                cache = (
                    get_layer_caches_for_groups(self.kv_cache, index, tags)
                    if self.kv_cache
                    else None
                )
            else:
                cache = self.kv_cache.get_layer_cache(index) if self.kv_cache else None
                if (
                    prepared is not None
                    and layer.layer_type == HybridAttentionType.LINEAR
                    and cache is not None
                    and str(cache.tag) == linear_tag
                ):
                    # The natural view and cuLA lookup are keyed by the original
                    # inputs object. Reuse it even if the binding wraps it again.
                    attention_inputs = linear_inputs
                else:
                    attention_inputs = select_attention_inputs_for_layer(
                        inputs, self.kv_cache, index
                    )
                fmha = (
                    None
                    if layer.layer_type == HybridAttentionType.LINEAR
                    else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, index)
                )
            if defer_mlp_residual:
                hidden, mlp_delta, mlp_shared_delta = layer(
                    hidden,
                    anchors,
                    fmha,
                    cache,
                    attention_inputs,
                    metadata,
                    valid_mask,
                    mlp_delta=mlp_delta,
                    mlp_shared_delta=mlp_shared_delta,
                    defer_mlp_residual=True,
                )
            else:
                hidden = layer(
                    hidden, anchors, fmha, cache, attention_inputs, metadata, valid_mask
                )
        if self.num_blocks:
            hidden = (
                self.final_residual(
                    hidden, anchors, delta=mlp_delta, shared_delta=mlp_shared_delta
                )
                if mlp_delta is not None
                else self.final_residual(hidden, anchors)
            )
        if outer_layout is not None:
            # Target MTP publication and the public final-norm/last-row ABI
            # consume the complete original Zigzag shard, including padding.
            with profile_scope("RTP::decoder.natural_to_zigzag"):
                hidden = self._row_exchange.exchange(
                    hidden, self._routing_plan, "to_zigzag"
                )
        hidden = self._publish_mtp_prefill_hidden(hidden)
        return (
            all_gather(hidden, Group.TP)
            if gather_output and self.tp_size > 1 and layout is None
            else hidden
        )

    def forward(self, inputs, fmha_impl=None):
        return self._forward_single(inputs, fmha_impl)

    def _forward_single(self, inputs, fmha_impl=None):
        hidden = self._forward_layers(None, inputs, fmha_impl)
        return PyModelOutputs(self.norm(hidden), hidden)

    def _initialize_nvfp4_direct_ag(self, init_resource):
        if self.tp_size == 1 or init_resource.is_decode_role:
            return
        attentions = [
            layer.attention
            for layer in self.layers
            if getattr(layer.attention, "_nvfp4_ag_mode", None) == "direct"
        ]
        if not attentions:
            return
        from rtp_llm.models_py.distributed.collective_torch import (
            create_custom_all_gather,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
            nvfp4_packet_capacity,
        )

        width = self.config.hidden_size
        max_m = self._projection_staging_tokens
        direct = create_custom_all_gather(
            _get_group(Group.TP),
            attentions[0].input.weight.device,
            max_m=max_m,
            k=width,
            fp8=False,
            staging_only=False,
            packed_bytes=nvfp4_packet_capacity(
                max_m // self.tp_size, width, rowwise=True
            ),
        )
        for attention in attentions:
            attention._nvfp4_direct_ag = direct

    def _initialize_decode_nvfp4_ag(self, init_resource):
        if not init_resource.is_decode_role or self.tp_size == 1:
            return
        attentions = [
            layer.attention
            for layer in self.layers
            if layer.block_size
            and getattr(layer.attention.input, "dynamic_input_scale", False)
        ]
        if not attentions:
            return
        from rtp_llm.models_py.distributed.collective_torch import (
            create_custom_all_gather,
        )
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_comm_fusion import (
            DecodeNvfp4AllGather,
            decode_packet_capacity,
        )

        group = _get_group(Group.TP)
        width = self.config.hidden_size
        max_m = self._projection_staging_tokens
        gate_columns = max(
            getattr(attention, "_decode_head_gate_columns", 0)
            for attention in attentions
        )
        direct = create_custom_all_gather(
            group,
            attentions[0].input.weight.device,
            max_m=max_m,
            k=width,
            fp8=False,
            staging_only=False,
            packed_bytes=decode_packet_capacity(
                max_m // self.tp_size,
                width,
                gate_columns,
                rowwise=True,
            ),
        )
        collective = DecodeNvfp4AllGather(group, direct)
        for attention in attentions:
            attention._decode_nvfp4_ag = collective
