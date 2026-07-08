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
    build_bert_uqi_flashinfer_mask,
    build_bert_uqi_two_pass_schedule,
    derive_bert_uqi_segment_ids,
    derive_bert_uqi_segment_ids_hostlen,
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
            os.environ.get("USE_VISION_BERT_UQI_BLOCK_MASK", "0") == "1"
        )
        self.cls_uqi_token_id = int(
            os.environ.get("VISION_BERT_CLS_UQI_TOKEN_ID", "2")
        )
        # 两趟 native attention (无 custom_mask, 免同步 plan, 常驻 wrapper)。
        # 默认开; =0 回退旧 dense-mask 路径 (A/B 对比用)。
        self.use_uqi_two_pass = (
            os.environ.get("VISION_BERT_UQI_TWO_PASS", "1") == "1"
        )
        self._uqi_two_pass_op = None  # 常驻 op(双 wrapper), 首请求惰性创建

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        # 关闭(默认)或非分块场景 -> 走原 factory 选择, 老路逐字节不变。
        if self.use_user_profile_mask and not is_cuda_graph:
            attn_inputs = inputs.attention_inputs
            if attn_inputs.is_prefill:
                if self.use_uqi_two_pass:
                    return self._prepare_uqi_two_pass_impl(inputs, attn_inputs)
                # 段标记与 mask 全在 GPU 上向量化构建(无全量 input_ids D2H / mask H2D
                # 往返)。P1: 去掉原 `bool(uqi_segment_ids.any())` 那次 per-request
                # GPU->CPU 标量同步 —— 无 B 段时 mask 全可见 = no-op, 结果不变;
                # 无条件建 mask 既省一次同步(并发下打断流水线的元凶之一),
                # 又不再 fallback 到 super() 的 TRT 路径。
                uqi_segment_ids = derive_bert_uqi_segment_ids(
                    inputs.input_ids,
                    attn_inputs.cu_seqlens,
                    self.cls_uqi_token_id,
                )
                attn_configs = self.config.getAttentionConfigs(
                    self.parallelism_config.get_attn_tp_size()
                )
                # FlashInfer custom_mask: 喂逻辑布尔 mask, 库内部按架构 swizzle/打包,
                # 硬件可移植 + fused-快 + 已 GPU 对拍 eager oracle 验证
                # (test_py_flashinfer_ragged_mha_prefill.test_block_mask_matches_eager_oracle)。
                from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                    PyFlashinferPrefillImpl,
                )

                custom_mask = build_bert_uqi_flashinfer_mask(
                    uqi_segment_ids, attn_inputs.cu_seqlens
                )
                return PyFlashinferPrefillImpl(
                    attn_configs,
                    attn_inputs,
                    self.parallelism_config,
                    custom_mask=custom_mask,
                )
        return super().prepare_fmha_impl(inputs, is_cuda_graph)

    def _prepare_uqi_two_pass_impl(
        self, inputs: PyModelInputs, attn_inputs: Any
    ) -> Any:
        """两趟 native attention: 无 custom_mask (保 FA3), plan 吃 host indptr
        (免同步), wrapper 常驻复用。唯一 D2H = schedule 里 [N] int 的 b_lens。"""
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass import (
            BertUqiTwoPassAttnOp,
            BertUqiTwoPassImpl,
        )

        batch_size = attn_inputs.input_lengths.size(0)
        cu_host = attn_inputs.cu_seqlens_host[: batch_size + 1]
        cu_dev = attn_inputs.cu_seqlens[: batch_size + 1]
        input_ids = inputs.input_ids[: int(cu_host[-1])]
        seg_ids = derive_bert_uqi_segment_ids_hostlen(
            input_ids, cu_dev, cu_host, self.cls_uqi_token_id
        )
        schedule = build_bert_uqi_two_pass_schedule(seg_ids, cu_dev, cu_host)
        if self._uqi_two_pass_op is None:
            attn_configs = self.config.getAttentionConfigs(
                self.parallelism_config.get_attn_tp_size()
            )
            self._uqi_two_pass_op = BertUqiTwoPassAttnOp(attn_configs)
        return BertUqiTwoPassImpl(self._uqi_two_pass_op, attn_inputs, schedule)

    def forward(
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        if bert_embedding_inputs.multimodal_features:
            # Vision positions carry out-of-vocab placeholder/hash ids that would index
            # past the embedding table (illegal CUDA access). Clamp to a valid row; the
            # multimodal injector below overwrites these rows with the real features.
            input_ids = input_ids.clamp(0, self.embed_tokens.weight.size(0) - 1)
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
        # 两趟 attention 需要每条序列重排为 [A...|B...] (层栈前 permute 一次,
        # 层栈后逆 permute 回原布局; 位置信息已在 embedding 阶段加完, attention
        # 对 token 顺序等变, 数学上恒等)。uqi_perm=None => 恒等, 零开销。
        uqi_perm = getattr(fmha_impl, "uqi_perm", None)
        if uqi_perm is not None:
            hidden_states = hidden_states.index_select(0, uqi_perm)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        if uqi_perm is not None:
            hidden_states = hidden_states.index_select(0, fmha_impl.uqi_inv_perm)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
