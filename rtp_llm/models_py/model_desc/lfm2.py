"""LiquidAI LFM2-VL model description (d1 series backbone).

Hybrid architecture (text backbone):
  - 22 gated short-conv layers (``Lfm2ShortConv``): in_proj -> (B, C, x)
    chunks, B*x gated depthwise causal conv (kernel 3, no activation),
    C output gate, out_proj; conv state = last 2 gated inputs per sequence.
  - 8 full-attention layers: GQA 32q/8kv, per-head q/k RMSNorm, RoPE.
  - All 30 layers share a SwiGLU MLP (sequential residual, HF-parity).

The conv state lives in the LINEAR hybrid cache group via the same
block-map mechanism kimi_linear uses; the layout contract is documented
in ``rtp_llm/models/lfm2/lfm2.py``.
"""

from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MultimodalEmbeddingInjector,
    RMSResNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.gated_causal_conv1d import (
    gated_causal_conv1d_fn,
    gated_causal_conv1d_update,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype


class Lfm2Metadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta


class Lfm2ShortConv(nn.Module):
    """Gated depthwise causal short conv (HF ``Lfm2ShortConv``).

    forward: out = out_proj(C ⊙ dwconv_k3(B ⊙ x)) with B, C, x the three
    chunks of ``in_proj``. Both gate multiplies and the depthwise conv run
    in the fused ``gated_causal_conv1d`` Triton kernel, which reads the
    ``in_proj`` output directly and reproduces the decomposed
    mul→conv→mul path bit-for-bit (B*X is rounded to bf16 before the conv
    taps, the conv result is rounded before the C multiply). The conv state
    still stores the *gated* input ``B ⊙ x`` of the last ``kernel - 1``
    tokens, so state layout and continuation semantics are unchanged.
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_hidden = config.hidden_size // self.attn_tp_size
        self.kernel = config.linear_attention_config.linear_conv_kernel_dim

        self.in_proj = Lfm2ShortConv._create_linear(
            weights, W.conv_in_proj_w, quant_config, hw_kernel_config
        )
        self.out_proj = Lfm2ShortConv._create_linear(
            weights, W.conv_out_proj_w, quant_config, hw_kernel_config
        )
        # [local_hidden, kernel] depthwise taps
        self.conv_weights = weights[W.conv_w]
        assert self.conv_weights.shape == (
            self.local_hidden,
            self.kernel,
        ), f"conv weight shape {tuple(self.conv_weights.shape)} != ({self.local_hidden}, {self.kernel})"

        # LINEAR cache-group layout contract: the (unused) ssm head occupies
        # the first ssm_elems slots of every block row; conv state follows.
        linear_config = config.linear_attention_config
        local_v_heads = linear_config.linear_num_value_heads // self.attn_tp_size
        # conv state width == hidden size by the lfm2.py layout contract
        qkv_size = (
            linear_config.linear_key_head_dim
            * (linear_config.linear_num_key_heads // self.attn_tp_size)
            * 2
            + linear_config.linear_value_head_dim * local_v_heads
        )
        assert qkv_size == self.local_hidden, (
            f"LINEAR cache qkv width {qkv_size} != local hidden {self.local_hidden};"
            " update the layout contract in rtp_llm/models/lfm2/lfm2.py"
        )
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=local_v_heads,
            head_v_dim=linear_config.linear_value_head_dim,
            head_k_dim=linear_config.linear_key_head_dim,
            ssm_state_dtype=to_torch_dtype(linear_config.ssm_state_dtype),
            linear_conv_kernel_dim=linear_config.linear_conv_kernel_dim,
            qkv_size=qkv_size,
            conv_state_dtype=to_torch_dtype(linear_config.conv_state_dtype),
        )

    @staticmethod
    def _create_linear(weights, weight_key, quant_config, hw_kernel_config):
        return LinearFactory.create_linear_from_weights(
            weights,
            weight_key,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        # [blocks, kernel - 1, qkv] view of the LINEAR block rows
        return self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)

    def _split_gates(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``in_proj`` output [T, 3H]; gate chunks are (B, C, X) on the last dim."""
        return self.in_proj(hidden_states)

    def _conv1d_prefill(
        self,
        projected: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata],
    ) -> torch.Tensor:
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)
            if kv_cache_tensor is not None
            else None
        )
        # reads projected [T, 3H] and applies B*X and C gating in-register
        return gated_causal_conv1d_fn(
            projected=projected,
            weight=self.conv_weights,
            conv_states=conv_states,
            query_start_loc=attn_inputs.cu_seqlens_device,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            prefix_lengths=attn_inputs.prefix_lengths_device,
            seq_size_per_block=seq_size_per_block,
            activation=None,
            metadata=metadata,
        )

    def _conv1d_decode(
        self,
        projected: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        batch, seq = self._get_bs_from_attention_input(
            projected, attn_inputs, is_target_verify
        )

        tokens = projected.shape[0]
        proj = projected.reshape(batch, seq, -1).transpose(1, 2)
        out = gated_causal_conv1d_update(
            proj,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation=None,
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
        )
        return out.transpose(1, 2).reshape(tokens, -1)

    @staticmethod
    def _get_bs_from_attention_input(
        projected: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = projected.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), "empty prefix_lengths in target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), "token count not divisible by target-verify batch"
        b = attention_inputs.prefix_lengths.size(0)
        s = token // b
        return b, s

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_inputs: Optional[PyAttentionInputs],
        kv_cache: Optional[LayerKVCache],
        attn_meta: Lfm2Metadata,
    ) -> torch.Tensor:
        assert attn_inputs is not None, "attention_inputs is required for conv layers"

        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        projected = self._split_gates(hidden_states)
        if attn_inputs.is_prefill and not attn_meta.is_target_verify:
            y = self._conv1d_prefill(
                projected,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                metadata=attn_meta.get_prefill_conv1d_meta(),
            )
        else:
            assert kv_cache_tensor is not None, "kv_cache is required for conv decode"
            y = self._conv1d_decode(
                projected,
                kv_cache_tensor,
                seq_size_per_block,
                attn_inputs,
                attn_meta.is_target_verify,
            )
        output = self.out_proj(y)
        if self.attn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


class Lfm2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]

        if self.layer_type == HybridAttentionType.LINEAR:
            self.conv = Lfm2ShortConv(
                config,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: Optional[FMHAImplBase],
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Lfm2Metadata = Lfm2Metadata(),
    ):
        # LFM2 residual stream: h = x + op(operator_norm(x));
        # out = h + MLP(ffn_norm(h)) -- RMSResNorm reproduces both adds.
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.conv(
                hidden_states=hidden_states,
                attn_inputs=attention_inputs,
                kv_cache=kv_cache,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Lfm2VlModel(GptModelBase):
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
        self.quant_config = quant_config
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.layers = nn.ModuleList(
            [
                Lfm2DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual capture requires a residual tensor")
        return hidden_states + residual

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual finalization requires a residual tensor")
        return self.norm(hidden_states, residual)[0]

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        full_attention_layers = (
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.layer_type != HybridAttentionType.LINEAR
        )
        return get_group_tags_for_layers(self.kv_cache, full_attention_layers)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        multimodal_features = None
        mm_feature_locs = None
        text_tokens_mask = None
        if inputs.multimodal_inputs is not None:
            multimodal_inputs = inputs.multimodal_inputs
            multimodal_features = multimodal_inputs.multimodal_features
            mm_feature_locs = multimodal_inputs.mm_features_locs
        if inputs.embedding_inputs is not None:
            text_tokens_mask = inputs.embedding_inputs.text_tokens_mask

        # image slots carry feature-hash ids with text_tokens_mask=0; the
        # embedding lookup skips them and the injector writes the real
        # features in their place
        inputs_embeds = self.embed_tokens(input_ids, text_tokens_mask=text_tokens_mask)
        if multimodal_features:
            inputs_embeds = self.multimodal_embedding_injector(
                inputs_embeds, multimodal_features, mm_feature_locs
            )
        hidden_states = inputs_embeds

        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=attention_inputs.cu_seqlens_device,
                device=hidden_states.device,
            )
        attn_meta = Lfm2Metadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        capture = self.capture_context(inputs.capture_hidden_states)
        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            layer_attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, i
            )
            layer_fmha_impl = (
                None
                if decoder_layer.layer_type == HybridAttentionType.LINEAR
                else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            )
            hidden_states, residual = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )
            capture.capture_layer(i, hidden_states, residual)

        return capture.finalize(hidden_states, residual)


__all__ = [
    "Lfm2VlModel",
    "Lfm2DecoderLayer",
    "Lfm2ShortConv",
    "Lfm2Metadata",
]
