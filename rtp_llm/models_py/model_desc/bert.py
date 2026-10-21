import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType
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
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.bert_user_profile import (
    build_bert_uqi_flashinfer_mask,
    derive_bert_uqi_segment_ids,
)
from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
    BertUqiTwoPassSchedule,
    build_bert_uqi_two_pass_schedule,
    build_bert_uqi_two_pass_schedule_from_bounds,
    derive_bert_uqi_segment_ids_hostlen,
)
from rtp_llm.utils.model_weight import W


class BertDecoderLayer(nn.Module):
    @staticmethod
    def _quant_bias(bias: torch.Tensor) -> torch.Tensor:
        return bias.reshape(-1) if bias.ndim == 2 and bias.shape[0] == 1 else bias

    @staticmethod
    def _can_fuse_layernorm_quant(hidden_states: torch.Tensor, linear) -> bool:
        # This is a BERT-only producer capability, deliberately narrower than
        # the linear backend's ability to consume prequantized activations.
        return (
            linear.supports_prequantized_activation
            and linear.fused_activation_quant_format == "fp8_ue8m0_block128_colmajor"
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and hidden_states.ndim == 2
            and hidden_states.is_contiguous()
            and 0 < hidden_states.shape[1] <= 1024
            and hidden_states.shape[1] % 128 == 0
        )

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
        quantized_hidden_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        quantize_output: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        empty_bias = torch.empty(
            0, device=hidden_states.device, dtype=hidden_states.dtype
        )

        residual = hidden_states
        hidden_states, attention_bias = self.self_attn.forward_without_output_bias(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            quantized_hidden_states=quantized_hidden_states,
        )
        attention_bias_tensor = (
            attention_bias.to(hidden_states.dtype)
            if attention_bias is not None
            else empty_bias
        )
        use_quantized_mlp_input = self._can_fuse_layernorm_quant(
            hidden_states, self.mlp.up_proj
        )
        if use_quantized_mlp_input:
            hidden_states, mlp_input, mlp_input_scales = (
                self.input_layernorm.forward_quantized(
                    hidden_states, residual, self._quant_bias(attention_bias_tensor)
                )
            )
            quantized_mlp_input = (mlp_input, mlp_input_scales)
        else:
            hidden_states = self.input_layernorm(
                hidden_states, residual, attention_bias_tensor
            )
            quantized_mlp_input = None

        residual = hidden_states
        hidden_states, ffn_bias = self.mlp.forward_without_output_bias(
            hidden_states, quantized_mlp_input
        )
        ffn_bias_tensor = (
            ffn_bias.to(hidden_states.dtype) if ffn_bias is not None else empty_bias
        )
        if quantize_output and self._can_fuse_layernorm_quant(
            hidden_states, self.self_attn.qkv_proj
        ):
            hidden_states, output_fp8, output_scales = (
                self.post_attention_layernorm.forward_quantized(
                    hidden_states, residual, self._quant_bias(ffn_bias_tensor)
                )
            )
            return hidden_states, (output_fp8, output_scales)
        hidden_states = self.post_attention_layernorm(
            hidden_states, residual, ffn_bias_tensor
        )
        return hidden_states, None


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
        vision_bert = config.model_type in ("vision_bert", "vision_bert_uqi")
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector(
            allow_reused_prefix=vision_bert, enable_fused_copy=vision_bert
        )
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
        self.cls_uqi_token_id = int(os.environ.get("VISION_BERT_CLS_UQI_TOKEN_ID", "2"))

        self.use_uqi_two_pass = os.environ.get("VISION_BERT_UQI_TWO_PASS", "1") == "1"
        self._uqi_two_pass_op = None

    def validate_embedding_input_features(self, inputs):
        mm = inputs.multimodal_inputs
        mask = inputs.embedding_inputs.text_tokens_mask
        features = bool(mm.multimodal_features)
        locs = mm.mm_features_locs is not None and mm.mm_features_locs.numel() > 0
        has_mask = mask is not None and mask.numel() > 0
        if not (features == locs == has_mask):
            return False
        # Metadata-only canRun calls may omit IDs. Actual forward must provide
        # one position and token-type ID for every token before embedding lookup.
        input_ids = getattr(inputs, "input_ids", None)
        tokens = input_ids.numel() if input_ids is not None else 0
        if tokens:
            for ids in (
                inputs.bert_embedding_inputs.combo_position_ids,
                inputs.bert_embedding_inputs.combo_tokens_type_ids,
            ):
                if (
                    ids is None
                    or ids.numel() != tokens
                    or ids.dtype != torch.int32
                    or not ids.is_contiguous()
                    or ids.device != inputs.input_ids.device
                ):
                    return False
        return True

    def prepare_embedding_attention_inputs(
        self, attn_inputs, host_tokens, host_text_mask
    ):
        if self.use_user_profile_mask:
            from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
                derive_bert_uqi_bounds,
                valid_bert_uqi_bounds,
            )

            # Preserve the native scan and eager schedule. Derive bounds only
            # when the caller did not supply usable host metadata.
            if (
                getattr(attn_inputs, "uqi_b_starts", None) is not None
                and getattr(attn_inputs, "uqi_b_lens", None) is not None
                and valid_bert_uqi_bounds(attn_inputs)
            ):
                return attn_inputs
            attn_inputs.uqi_b_starts, attn_inputs.uqi_b_lens = derive_bert_uqi_bounds(
                host_tokens,
                attn_inputs.input_lengths,
                host_text_mask,
                self.cls_uqi_token_id,
            )
        return attn_inputs

    def validate_embedding_graph_inputs(self, inputs):
        if not super().validate_embedding_graph_inputs(inputs):
            return False
        if not self.use_user_profile_mask:
            return True
        from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
        from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
            valid_bert_uqi_bounds,
        )

        attention = get_attention_inputs_value(inputs)
        groups = attention.values() if isinstance(attention, dict) else [attention]
        return all(valid_bert_uqi_bounds(attn) for attn in groups)

    def prepare_embedding_graph_attention(self, inputs, is_cuda_graph=True):
        return self.prepare_fmha_impl(inputs, is_cuda_graph=True)

    def prepare_embedding_graph_replay(self, attention, graph_inputs, request_inputs):
        if self.use_user_profile_mask:
            from rtp_llm.models_py.model_desc.block_map import (
                get_attention_inputs_value,
            )

            captured = get_attention_inputs_value(graph_inputs)
            current = get_attention_inputs_value(request_inputs)
            captured_groups = (
                captured if isinstance(captured, dict) else {None: captured}
            )
            current_groups = current if isinstance(current, dict) else {None: current}
            for tag, attn in captured_groups.items():
                attn.uqi_b_starts = current_groups[tag].uqi_b_starts
                attn.uqi_b_lens = current_groups[tag].uqi_b_lens
        super().prepare_embedding_graph_replay(attention, graph_inputs, request_inputs)

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> FMHAImplBase:
        # Graphs use the device-specific graph attention implementation; eager
        # retains its configured two-pass/custom-mask implementation.
        if is_cuda_graph:
            if self.kv_cache is not None:
                raise ValueError(
                    "BERT Transformer graph requires an encoder without KV cache"
                )
            if not self.use_user_profile_mask:
                return super().prepare_fmha_impl(inputs, True, "embedding_graph")
            if torch.version.hip is not None:
                from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass import (
                    BertUqiRocmEncoderGraphImpl,
                )

                graph_impl = BertUqiRocmEncoderGraphImpl
            else:
                from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass import (
                    BertUqiEncoderGraphImpl,
                )

                graph_impl = BertUqiEncoderGraphImpl
            return graph_impl(
                self.config.getAttentionConfigs(
                    self.parallelism_config.get_attn_tp_size()
                ),
                inputs.attention_inputs,
                inputs.input_ids.size(0),
                max_sequence_length=self.config.max_seq_len,
            )
        if self.use_user_profile_mask:
            attn_inputs = inputs.attention_inputs
            if attn_inputs.is_prefill:
                if self.use_uqi_two_pass:
                    return self._prepare_uqi_two_pass_impl(inputs, attn_inputs)
                if torch.version.hip is not None:
                    raise ValueError(
                        "ROCm BERT user-profile attention requires VISION_BERT_UQI_TWO_PASS=1; "
                        "the FlashInfer dense-mask implementation is CUDA-only"
                    )
                # Build logical masks on device; variable-size planning uses host scalars.
                token_ids = inputs.input_ids
                text_mask = inputs.embedding_inputs.text_tokens_mask
                if text_mask is not None and text_mask.numel():
                    token_ids = token_ids.masked_fill(
                        ~text_mask.to(device=token_ids.device, dtype=torch.bool), -1
                    )
                cu_seqlens = attn_inputs.cu_seqlens_device
                if cu_seqlens is None or cu_seqlens.numel() == 0:
                    cu_seqlens = attn_inputs.cu_seqlens
                cu_seqlens = cu_seqlens[: attn_inputs.input_lengths.numel() + 1]
                uqi_segment_ids = derive_bert_uqi_segment_ids(
                    token_ids,
                    cu_seqlens,
                    self.cls_uqi_token_id,
                )
                attn_configs = self.config.getAttentionConfigs(
                    self.parallelism_config.get_attn_tp_size()
                )
                from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                    PyFlashinferPrefillImpl,
                )

                custom_mask = None
                if bool(uqi_segment_ids.any()):
                    custom_mask = build_bert_uqi_flashinfer_mask(
                        uqi_segment_ids, cu_seqlens
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
        """Plan native two-pass attention from host metadata, reusing wrappers."""
        if torch.version.hip is not None:
            from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass import (
                BertUqiTwoPassAttnOp,
                BertUqiTwoPassImpl,
            )
        else:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi_two_pass import (
                BertUqiTwoPassAttnOp,
                BertUqiTwoPassImpl,
            )

        batch_size = attn_inputs.input_lengths.size(0)
        cu_host = attn_inputs.cu_seqlens[: batch_size + 1]
        if cu_host.numel() != batch_size + 1:
            lengths = attn_inputs.input_lengths.cpu().to(torch.int32)
            cu_host = torch.zeros(batch_size + 1, dtype=torch.int32)
            cu_host[1:] = lengths.cumsum(0)
        # Prefer the C++ precomputed schedule. Graph-mode eager fallback and
        # callers without a native schedule build it from bounds or text tokens.
        uqi_seg_indptr = getattr(attn_inputs, "uqi_seg_indptr", None)
        uqi_b_lens = getattr(attn_inputs, "uqi_b_lens", None)
        if uqi_seg_indptr is not None and uqi_seg_indptr.numel() == 2 * batch_size + 1:
            if int(attn_inputs.uqi_b_indptr[batch_size]) == 0:
                schedule = BertUqiTwoPassSchedule(
                    has_b=False,
                    perm=None,
                    inv_perm=None,
                    b_rows=None,
                    qo_indptr_p1=cu_host,
                    qo_indptr_p2=None,
                    kv_indptr_p2=None,
                )
            else:
                dev = inputs.input_ids.device
                schedule = BertUqiTwoPassSchedule(
                    has_b=True,
                    perm=attn_inputs.uqi_perm.to(dev, non_blocking=True),
                    inv_perm=attn_inputs.uqi_inv_perm.to(dev, non_blocking=True),
                    b_rows=attn_inputs.uqi_b_rows.to(dev, non_blocking=True),
                    qo_indptr_p1=uqi_seg_indptr,
                    qo_indptr_p2=attn_inputs.uqi_b_indptr,
                    kv_indptr_p2=cu_host,
                )
        elif uqi_b_lens is not None and uqi_b_lens.numel() >= batch_size:
            schedule = build_bert_uqi_two_pass_schedule_from_bounds(
                attn_inputs.uqi_b_starts[:batch_size],
                uqi_b_lens[:batch_size],
                cu_host,
                inputs.input_ids.device,
            )
        else:
            cu_dev = attn_inputs.cu_seqlens_device[: batch_size + 1]
            input_ids = inputs.input_ids[: int(cu_host[-1])]
            text_mask = inputs.embedding_inputs.text_tokens_mask
            if text_mask is not None and text_mask.numel():
                input_ids = input_ids.masked_fill(
                    ~text_mask[: input_ids.numel()].to(
                        device=input_ids.device, dtype=torch.bool
                    ),
                    -1,
                )
            seg_ids = derive_bert_uqi_segment_ids_hostlen(
                input_ids, cu_dev, cu_host, self.cls_uqi_token_id
            )
            schedule = build_bert_uqi_two_pass_schedule(seg_ids, cu_dev, cu_host)
        if self._uqi_two_pass_op is None:
            attn_configs = self.config.getAttentionConfigs(
                self.parallelism_config.get_attn_tp_size()
            )
            self._uqi_two_pass_op = BertUqiTwoPassAttnOp(attn_configs)
        impl = BertUqiTwoPassImpl(self._uqi_two_pass_op, attn_inputs, schedule)
        return impl

    def forward(
        self,
        inputs: PyModelInputs,
        fmha_impl: FMHAImplBase = None,
    ) -> PyModelOutputs:
        # Graph replay supplies the fixed buffer prepared outside capture.
        input_hiddens = getattr(inputs, "input_hiddens", None)
        hidden_states = (
            self.prepare_transformer_inputs(inputs)
            if input_hiddens is None or input_hiddens.numel() == 0
            else input_hiddens
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        if self.config.model_type == "vision_bert" and self.device_type == DeviceType.Cuda:
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.trt import (
                FlashInferTRTLLMFMHAv2PrefillImpl,
            )

            if isinstance(fmha_impl, FlashInferTRTLLMFMHAv2PrefillImpl):
                fmha_impl.fmha_impl.enable_sm120_bert_specialization = True
        quantized_hidden_states = None
        uqi_perm = getattr(fmha_impl, "uqi_perm", None)
        if uqi_perm is not None:
            hidden_states = hidden_states.index_select(0, uqi_perm)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            next_layer_uses_quantized_input = (
                i + 1 < self.layer_num
                and self.layers[
                    i + 1
                ].self_attn.qkv_proj.supports_prequantized_activation
            )
            hidden_states, quantized_hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                quantized_hidden_states=quantized_hidden_states,
                quantize_output=next_layer_uses_quantized_input,
            )
        if uqi_perm is not None:
            hidden_states = hidden_states.index_select(0, fmha_impl.uqi_inv_perm)
        return PyModelOutputs(hidden_states)

    def prepare_transformer_inputs(self, inputs: PyModelInputs) -> torch.Tensor:
        """Request-owned embedding and visual injection, always outside graphs."""
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        # Image slots contain cache hashes, not word-table IDs. Validate the
        # producer contract before the lookup, then skip those word-table rows.
        multimodal_features = inputs.multimodal_inputs.multimodal_features
        multimodal_locs = inputs.multimodal_inputs.mm_features_locs
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        has_mask = text_tokens_mask is not None and text_tokens_mask.numel() != 0
        has_locs = multimodal_locs is not None and multimodal_locs.numel() != 0
        if not (bool(multimodal_features) == has_locs == has_mask):
            raise ValueError(
                "multimodal features, locations, and text_tokens_mask must be "
                "provided together"
            )
        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
            text_tokens_mask if has_mask else None,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)
        # Projected image features are already normalized in decoder input
        # space; never add text position/type embeddings or normalize them twice.
        hidden_states = self.multimodal_embedding_injector(
            hidden_states, multimodal_features, multimodal_locs
        )

        return hidden_states
