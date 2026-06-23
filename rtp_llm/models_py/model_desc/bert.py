import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AddBiasResLayerNorm,
    AttnImplFactory,
    CausalAttention,
    DenseMLP,
    EmbeddingBert,
    FMHAImplBase,
    LayerNorm,
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.factory.attention.block_mask import (
    build_flashinfer_block_mask,
    derive_segment_ab,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class BertDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        attn_configs.need_rope_kv_cache = False
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = AddBiasResLayerNorm(
            weights[W.post_ln_gamma],
            beta=weights[W.post_ln_beta],
            eps=config.layernorm_eps,
        )
        self.post_attention_layernorm = AddBiasResLayerNorm(
            weights[W.post_ffn_ln_gamma],
            beta=weights[W.post_ffn_ln_beta],
            eps=config.layernorm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = self.input_layernorm(hidden_states, residual, torch.empty(0))

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(
            hidden_states, residual, torch.empty(0)
        )
        return hidden_states


class BertModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = EmbeddingBert(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_decoder_layernorm = LayerNorm(
            weight=weights.get_global_weight(W.pre_decoder_ln_gamma),
            beta=weights.get_global_weight(W.pre_decoder_ln_beta),
            eps=config.layernorm_eps,
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.layers = nn.ModuleList(
            [
                BertDecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        # 用户画像分支: 显式 opt-in, 默认关 -> 普通 BERT 老路逐字节不变。
        # 开启后 A 段(Q+I)看不到 B 段(User), B 段从 CLS_UQI(token id) 起。
        self.use_user_profile_mask = (
            os.environ.get("USE_USER_PROFILE_BLOCK_MASK", "0") == "1"
        )
        self.cls_uqi_token_id = int(os.environ.get("CLS_UQI_TOKEN_ID", "2"))

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        # 关闭(默认)或非分块场景 -> 走原 factory 选择, 老路逐字节不变。
        if self.use_user_profile_mask and not is_cuda_graph:
            attn_inputs = inputs.attention_inputs
            if attn_inputs.is_prefill:
                # 单次 D2H: cu_seqlens 很小, 两个纯-CPU helper 共用一份拷贝即可。
                cu_seqlens_cpu = attn_inputs.cu_seqlens.cpu()
                seg = derive_segment_ab(
                    inputs.input_ids.cpu(),
                    cu_seqlens_cpu,
                    self.cls_uqi_token_id,
                )
                if int(seg.max()) > 0:  # 真有 B 段才接 block mask
                    attn_configs = self.config.getAttentionConfigs(
                        self.parallelism_config.get_attn_tp_size()
                    )
                    # FlashInfer custom_mask: 喂逻辑布尔 mask, 库内部按架构 swizzle/打包,
                    # 硬件可移植 + fused-快 + 已 GPU 对拍 eager oracle 验证
                    # (test_py_flashinfer_ragged_mha_prefill.test_block_mask_matches_eager_oracle)。
                    from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                        PyFlashinferPrefillImpl,
                    )

                    custom_mask = build_flashinfer_block_mask(
                        seg, cu_seqlens_cpu
                    ).to(inputs.input_ids.device)
                    return PyFlashinferPrefillImpl(
                        attn_configs,
                        attn_inputs,
                        self.parallelism_config,
                        custom_mask=custom_mask,
                    )
        return super().prepare_fmha_impl(inputs, is_cuda_graph)

    def forward(
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)
        hidden_states = self.multimodal_embedding_injector(
            hidden_states,
            bert_embedding_inputs.multimodal_features,
            bert_embedding_inputs.mm_features_locs,
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
