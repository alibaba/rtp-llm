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
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    ParamsBase,
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
        use_quantized_mlp_input = self.mlp.up_proj.supports_prequantized_activation
        if use_quantized_mlp_input:
            hidden_states, mlp_input, mlp_input_scales = (
                self.input_layernorm.forward_quantized(
                    hidden_states, residual, attention_bias_tensor
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
        if quantize_output and self.self_attn.qkv_proj.supports_prequantized_activation:
            hidden_states, output_fp8, output_scales = (
                self.post_attention_layernorm.forward_quantized(
                    hidden_states, residual, ffn_bias_tensor
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
        self.cls_uqi_token_id = int(os.environ.get("VISION_BERT_CLS_UQI_TOKEN_ID", "2"))

        self.use_uqi_two_pass = os.environ.get("VISION_BERT_UQI_TWO_PASS", "1") == "1"
        self._uqi_two_pass_op = None

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
            graph_impl = BertEncoderGraphImpl
            if self.use_user_profile_mask:
                if torch.version.hip is not None:
                    from rtp_llm.models_py.modules.factory.attention.rocm_impl.bert_uqi_two_pass import (
                        BertUqiRocmEncoderGraphImpl,
                    )

                    graph_impl = BertUqiRocmEncoderGraphImpl
                else:
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
        # 优先走 C++ 全 schedule 路径: PyWrappedModel 已在 host token 上扫出段
        # 边界并预构建 perm/indptr/b_rows —— Python 侧只剩 flashinfer plan 与
        # permutation H2D。缺少预构建 metadata 时回退到分段推导。
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
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        hidden_states = self.prepare_transformer_inputs(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        return self.forward_transformer(hidden_states, fmha_impl)

    def prepare_transformer_inputs(self, inputs: PyModelInputs) -> torch.Tensor:
        """Request-owned embedding and visual injection, always outside graphs."""
        self._reject_input_embeddings(inputs)
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        # MultimodalProcessor writes arbitrary signed int32 feature hashes into
        # image slots for prefix-cache keys. The image features replace these
        # rows below, but embedding lookup runs first and must only see valid IDs.
        multimodal_features = inputs.multimodal_inputs.multimodal_features
        multimodal_locs = inputs.multimodal_inputs.mm_features_locs
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        has_text_tokens_mask = (
            text_tokens_mask is not None and text_tokens_mask.numel() != 0
        )
        has_multimodal_features = bool(multimodal_features)
        has_multimodal_locs = (
            multimodal_locs is not None and multimodal_locs.numel() != 0
        )
        # This is Bert-specific: image slots hold feature hashes before the
        # word-table lookup, while feature/location pairing belongs to the
        # shared injector below. Keep the check here, before any device lookup.
        if not (has_multimodal_features == has_multimodal_locs == has_text_tokens_mask):
            raise ValueError(
                "multimodal features, locations, and text_tokens_mask must be "
                "provided together: "
                f"features={len(multimodal_features)}, "
                f"locations={multimodal_locs.numel() if has_multimodal_locs else 0}, "
                f"mask_tokens={text_tokens_mask.numel() if has_text_tokens_mask else 0}"
            )

        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
            text_tokens_mask if has_text_tokens_mask else None,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)

        # Producer contract: every multimodal feature is already projected to
        # hidden_size and normalized in the first decoder layer's input space.
        # Replace image rows after the text pre-LN so producer-owned features do
        # not inherit text position/type embeddings or get normalized twice.
        hidden_states = self.multimodal_embedding_injector(
            hidden_states,
            multimodal_features,
            multimodal_locs,
        )

        return hidden_states

    def forward_transformer(
        self, hidden_states: torch.Tensor, fmha_impl: FMHAImplBase
    ) -> PyModelOutputs:
        """Transformer-only entry point shared by eager and graph execution."""
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


class BertEncoderGraphParams(ParamsBase):
    def check_recycle(self):
        return True


class BertEncoderGraphImpl(FMHAImplBase):
    def __init__(
        self,
        configs,
        attn_inputs,
        token_capacity,
        max_sequence_length=None,
        custom_mask=False,
        allocate_indptr=True,
    ):
        if configs.is_causal:
            raise ValueError("BERT encoder graph requires noncausal attention")
        self.heads = configs.head_num
        self.kv_heads = configs.kv_head_num
        self.dim = configs.size_per_head
        self.dtype = configs.dtype
        self.token_capacity = token_capacity
        self.max_sequence_length = max_sequence_length
        self.request_capacity = attn_inputs.input_lengths.numel()
        self.fmha_params = BertEncoderGraphParams()
        self.cu_seqlens_host = None
        self.cu_seqlens_device = None
        if allocate_indptr:
            self.cu_seqlens_host = torch.empty(
                self.request_capacity + 2, dtype=torch.int32, pin_memory=True
            )
            self.cu_seqlens_device = torch.empty_like(
                self.cu_seqlens_host, device="cuda"
            )
        self.wrapper = None
        if torch.version.hip is None:
            from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

            # Each graph owns its planner workspace and indptr storage. Replanning
            # another bucket must not overwrite this graph's execution plan.
            self.workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self.kv_cu_seqlens_device = torch.empty_like(self.cu_seqlens_device)
            mask_buffers = {}
            if custom_mask:
                # Each sequence is packed separately and may round up one byte.
                if token_capacity * token_capacity > torch.iinfo(torch.int32).max:
                    raise ValueError("UQI mask exceeds int32 mask-indptr capacity")
                mask_bytes = (
                    (token_capacity * token_capacity + 7) // 8
                    + self.request_capacity
                    + 1
                )
                self.custom_mask_buffer = torch.empty(
                    mask_bytes, dtype=torch.uint8, device="cuda"
                )
                self.mask_indptr_buffer = torch.empty_like(self.cu_seqlens_device)
                mask_buffers = dict(
                    custom_mask_buf=self.custom_mask_buffer,
                    mask_indptr_buf=self.mask_indptr_buffer,
                )
            self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace,
                use_cuda_graph=True,
                qo_indptr_buf=self.cu_seqlens_device,
                kv_indptr_buf=self.kv_cu_seqlens_device,
                backend="fa2",
                **mask_buffers,
            )
        # Capture uses one synthetic sequence as long as the aggregate bucket.
        # Only real replay sequences are constrained by position-table length.
        self._prepare(attn_inputs, for_capture=True)

    def prepare_cuda_graph(self, attn_inputs):
        self._prepare(attn_inputs, for_capture=False)

    def _prepare(self, attn_inputs, for_capture):
        """Plan synthetic capture inputs or validate and plan a real replay batch."""
        lengths = attn_inputs.input_lengths
        if lengths.is_cuda:
            raise ValueError("BERT graph planning requires host input lengths")
        if lengths.numel() != self.request_capacity:
            raise ValueError("BERT graph request capacity changed")
        prefix = attn_inputs.prefix_lengths
        if prefix is not None and prefix.numel() and bool(prefix.any()):
            raise ValueError("BERT encoder graph does not support prefix KV")
        if bool((lengths < 0).any()):
            raise ValueError("BERT graph sequence lengths must be nonnegative")
        if (
            not for_capture
            and self.max_sequence_length is not None
            and bool((lengths > self.max_sequence_length).any())
        ):
            raise ValueError("BERT graph sequence exceeds position-table capacity")
        real_tokens = int(lengths.sum())
        if real_tokens > self.token_capacity:
            raise ValueError("BERT graph token capacity exceeded")
        self.cu_seqlens_host[0] = 0
        torch.cumsum(lengths, 0, out=self.cu_seqlens_host[1:-1])
        self.cu_seqlens_host[-1] = self.token_capacity
        if self.wrapper is None:
            self.cu_seqlens_device.copy_(self.cu_seqlens_host, non_blocking=True)
        else:
            self.wrapper.plan(
                self.cu_seqlens_host,
                self.cu_seqlens_host,
                self.heads,
                self.kv_heads,
                self.dim,
                self.dim,
                causal=False,
                q_data_type=self.dtype,
                kv_data_type=self.dtype,
                o_data_type=self.dtype,
                disable_split_kv=True,
                **self._mask_plan_args(attn_inputs, for_capture),
            )

    def _mask_plan_args(self, attn_inputs, for_capture):
        return {}

    def forward(self, qkv, kv_cache, layer_idx=0):
        if kv_cache is not None:
            raise ValueError("BERT encoder graph cannot use KV cache")
        q, k, v = torch.split(
            qkv.reshape(self.token_capacity, -1),
            [self.heads * self.dim, self.kv_heads * self.dim, self.kv_heads * self.dim],
            dim=-1,
        )
        q = q.reshape(self.token_capacity, self.heads, self.dim)
        k = k.reshape(self.token_capacity, self.kv_heads, self.dim)
        v = v.reshape(self.token_capacity, self.kv_heads, self.dim)
        if self.wrapper is not None:
            output = self.wrapper.run(q, k, v)
        else:
            from aiter import flash_attn_varlen_func

            output = flash_attn_varlen_func(
                q,
                k,
                v,
                self.cu_seqlens_device,
                self.cu_seqlens_device,
                self.token_capacity,
                self.token_capacity,
                dropout_p=0.0,
                causal=False,
            )
        return output.reshape(self.token_capacity, self.heads * self.dim)

    @staticmethod
    def support(configs, attn_inputs):
        return attn_inputs.is_prefill and not configs.is_causal

    def support_cuda_graph(self):
        return True


class BertUqiEncoderGraphImpl(BertEncoderGraphImpl):
    """CUDA UQI graphs with one FlashInfer masked attention per layer."""

    def __init__(self, configs, attn_inputs, token_capacity, max_sequence_length=None):
        if torch.version.hip is not None:
            raise ValueError("UQI encoder graph currently supports CUDA only")
        super().__init__(
            configs, attn_inputs, token_capacity, max_sequence_length, custom_mask=True
        )

    def _mask_plan_args(self, attn_inputs, for_capture):
        from rtp_llm.models_py.modules.factory.attention.bert_uqi_schedule import (
            build_uqi_graph_layout,
        )

        lengths = attn_inputs.input_lengths.tolist()
        if not for_capture:
            starts = attn_inputs.uqi_b_starts
            b_lengths = attn_inputs.uqi_b_lens
            if (
                starts is None
                or b_lengths is None
                or starts.is_cuda
                or b_lengths.is_cuda
            ):
                raise ValueError("UQI graph requires host segment bounds")
            if (
                starts.numel() > self.request_capacity
                or starts.numel() != b_lengths.numel()
            ):
                raise ValueError("invalid UQI graph request metadata")
            active = starts.numel()
            if any(lengths[active:]):
                raise ValueError("missing UQI graph segment bounds")
            starts = starts.tolist() + [-1] * (len(lengths) - active)
            b_lengths = b_lengths.tolist() + [0] * (len(lengths) - active)
        else:
            # Synthetic startup inputs contain no user segment. Always pass a
            # mask, even here, so capture never specializes to an unmasked path.
            starts, b_lengths = [-1] * len(lengths), [0] * len(lengths)
        layout = build_uqi_graph_layout(lengths, starts, b_lengths, self.token_capacity)
        masks = []
        for n, start, count in layout:
            pos = torch.arange(n, device="cuda")
            is_b = (pos >= start) & (pos < start + count)
            # A -> A; B -> A+B. Separate ragged sequences isolate padding.
            masks.append((is_b[:, None] | ~is_b[None, :]).reshape(-1))
        return {"custom_mask": torch.cat(masks)}
