"""MiniMax-M3.1 text runtime on main's model and tagged cache contracts."""

from collections.abc import Mapping

import torch
from torch import nn

from rtp_llm.models.minimax_m31_kv_cache import INDEXER_KV_TAG, MAIN_KV_TAG
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    get_layer_caches_for_groups,
)
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeModel
from rtp_llm.models_py.model_desc.minimax_m31_attention import M31Attention
from rtp_llm.models_py.model_desc.minimax_m31_mlp import M31DenseMLP
from rtp_llm.models_py.model_desc.minimax_m31_moe import (
    _PREFILL_CHUNK_PLAN,
    M31MoeLayer,
    prepare_m31_prefill_chunks,
)
from rtp_llm.models_py.modules import RMSResNorm
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class M31DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        parallelism_config,
        weights,
        global_weights,
        layer_idx,
        moe_config,
        max_generate_batch_size=0,
        enable_cuda_graph=False,
        hw_kernel_config=None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = M31Attention(
            config,
            parallelism_config,
            weights,
            layer_idx,
            hw_kernel_config,
        )
        if layer_idx in config.moe_layer_index:
            self.mlp = M31MoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            self.mlp = M31DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                alpha=config.swiglu_alpha,
                limit=config.swiglu_limit,
            )
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(self, hidden_states, residual, attention_inputs, kv_cache):
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(hidden_states, attention_inputs, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        return DecodeLayerOutput(self.mlp(hidden_states), residual)


class _M31NativeAttentionContext:
    fmha_params = None

    def prepare_cuda_graph(self, attention_inputs):
        return None

    def support_cuda_graph(self) -> bool:
        return True


class MiniMaxM31Model(GenericMoeModel):
    decoder_layer_cls = M31DecoderLayer

    def has_mtp_hidden_buffer(self) -> bool:
        return bool(self._mtp_aux_capture_layer_ids)

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode=None,
    ):
        return _M31NativeAttentionContext()

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        del fmha_impl
        attention_inputs = get_attention_inputs_value(inputs)
        required = {MAIN_KV_TAG, INDEXER_KV_TAG}
        if (
            not isinstance(attention_inputs, Mapping)
            or set(attention_inputs) != required
        ):
            raise ValueError("M3.1 requires separate main and indexer attention inputs")
        if self.kv_cache is None:
            raise ValueError("M3.1 requires initialized tagged KV cache")
        hidden_states = self.embedding(inputs)
        residual = torch.zeros_like(hidden_states)
        capture = self.capture_context(inputs.capture_hidden_states)
        main_inputs = attention_inputs[MAIN_KV_TAG]
        if (
            main_inputs.is_prefill
            and not main_inputs.is_target_verify
            and not (hidden_states.is_cuda and torch.cuda.is_current_stream_capturing())
        ):
            plan = prepare_m31_prefill_chunks(
                self.layers[: self.layer_num],
                hidden_states.shape[0],
                hidden_states.device,
            )
        else:
            plan = None
        token = _PREFILL_CHUNK_PLAN.set(plan)
        capture_aux_hidden = bool(self._mtp_aux_capture_layer_ids)
        if capture_aux_hidden:
            self.begin_aux_hidden_capture(
                hidden_states, attention_inputs[MAIN_KV_TAG].is_target_verify
            )
        try:
            for i, layer in enumerate(self.layers[: self.layer_num]):
                cache = get_layer_caches_for_groups(
                    self.kv_cache, i, (MAIN_KV_TAG, INDEXER_KV_TAG)
                )
                output = layer(hidden_states, residual, attention_inputs, cache)
                hidden_states, residual = output.hidden_states, output.residual
                if i in self._mtp_aux_capture_layer_id_set:
                    self.capture_aux_hidden(i, hidden_states, residual)
                capture.capture_layer(i, hidden_states, residual)
            if capture_aux_hidden:
                self.finish_aux_hidden_capture()
            outputs = capture.finalize(hidden_states, residual)
            if capture_aux_hidden:
                outputs.mtp_target_hidden_states = self._mtp_target_hidden_states
            return outputs
        finally:
            _PREFILL_CHUNK_PLAN.reset(token)
