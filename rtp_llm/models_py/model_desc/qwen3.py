from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_primary_attention_inputs,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    RMSNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
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
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
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
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class AngelSlimQwen3Eagle3DecoderLayer(nn.Module):
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
            0,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.hidden_norm = RMSNorm(
            weights[W.eagle3_fc_norm_gamma], eps=config.layernorm_eps
        )
        self.input_layernorm = RMSNorm(
            weights[W.eagle3_input_norm_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = torch.cat(
            [self.input_layernorm(input_embeds), self.hidden_norm(hidden_states)],
            dim=-1,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class AngelSlimQwen3Eagle3Model(GptModelBase):
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
        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.weights[0],
            W.eagle3_fc_proj,
            quant_config=quant_config,
            hw_kernel_config=py_hw_kernel_config,
        )
        self.layer = AngelSlimQwen3Eagle3DecoderLayer(
            config,
            parallelism_config,
            weights.weights[0],
            quant_config,
            py_hw_kernel_config,
        )
        final_norm_weight = weights.get_global_weight(W.final_ln_gamma)
        self.norm = RMSNorm(final_norm_weight, eps=config.layernorm_eps)
        self.input_hidden_size = 3 * config.hidden_size
        # Recurrent draft hidden used as the next step's input. Two storages hold
        # the same value for different execution phases:
        #   * graph buffer: fixed address/capacity for CUDA-graph decode replay,
        #     which cannot allocate or rebind Python tensors during replay.
        #   * eager tensor: exact-size, allocated per non-graph prefill forward,
        #     whose row count can exceed the decode graph capacity.
        # Capacity covers the largest decode/verify batch: max_batch * (gamma + 1).
        self._mtp_hidden_graph_capacity = max_generate_batch_size * max(
            int(config.gen_num_per_cycle) + 1, 1
        )
        self.register_buffer(
            "_mtp_hidden_graph_buffer",
            torch.empty(
                (self._mtp_hidden_graph_capacity, config.hidden_size),
                dtype=final_norm_weight.dtype,
                device=final_norm_weight.device,
            ),
            persistent=False,
        )
        self._mtp_hidden_eager_tensor: Optional[torch.Tensor] = None
        self._mtp_hidden_eager_valid_tokens = 0

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        input_hiddens: torch.Tensor = inputs.input_hiddens
        if input_hiddens is None or input_hiddens.numel() == 0:
            raise RuntimeError("Qwen3 Eagle3 requires target hidden states")
        if input_hiddens.size(0) != input_ids.numel():
            raise ValueError(
                "Qwen3 Eagle3 token/hidden row mismatch: "
                f"tokens={input_ids.numel()}, hidden_rows={input_hiddens.size(0)}"
            )

        input_embeds = self.embed_tokens(input_ids)
        if input_hiddens.size(-1) != self.config.hidden_size:
            expected_width = 3 * self.config.hidden_size
            if input_hiddens.size(-1) != expected_width:
                raise ValueError(
                    "Qwen3 Eagle3 hidden width mismatch: "
                    f"got={input_hiddens.size(-1)}, expected={expected_width}"
                )
            input_hiddens = self.fc(input_hiddens)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        hidden_states = self.layer(
            input_embeds,
            input_hiddens,
            select_fmha_impl_for_layer(fmha_impl, self.kv_cache, 0),
            kv_cache=self.kv_cache.get_layer_cache(0) if self.kv_cache else None,
        )
        rows = hidden_states.size(0)
        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        is_cuda_graph = bool(attention_inputs.is_cuda_graph) or (
            hidden_states.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        if rows <= self._mtp_hidden_graph_buffer.size(0):
            self._mtp_hidden_graph_buffer[:rows].copy_(hidden_states)
        elif is_cuda_graph:
            raise ValueError(
                "AngelSlim Qwen3 Eagle3 recurrent hidden states exceed Graph capacity: "
                f"rows={rows}, capacity={self._mtp_hidden_graph_buffer.size(0)}"
            )
        if not is_cuda_graph:
            self._mtp_hidden_eager_tensor = hidden_states
            self._mtp_hidden_eager_valid_tokens = rows
        return PyModelOutputs(self.norm(hidden_states))

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        # num_tokens < 0: eager prefill reads the producer's exact row count.
        # num_tokens >= 0: graph decode/verify reads an explicit row count,
        # since graph replay does not update the Python-side valid-token counter.
        if num_tokens < 0:
            buffer = self._mtp_hidden_eager_tensor
            requested = self._mtp_hidden_eager_valid_tokens
        else:
            buffer = self._mtp_hidden_graph_buffer
            requested = int(num_tokens)
        if buffer is None or requested <= 0:
            return None
        if requested > buffer.size(0):
            raise ValueError(
                "AngelSlim Qwen3 Eagle3 recurrent hidden-state request exceeds capacity: "
                f"requested={requested}, capacity={buffer.size(0)}"
            )
        return buffer[:requested]


class Qwen3Model(GptModelBase):
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

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
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
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )
        self._capture_aux_hidden_layer_ids = tuple(
            int(layer_id) for layer_id in (config.capture_aux_hidden_layer_ids or ())
        )
        self._capture_aux_hidden_layer_id_set = set(self._capture_aux_hidden_layer_ids)
        capture_width = len(self._capture_aux_hidden_layer_ids) * config.hidden_size
        final_norm_weight = weights.get_global_weight(W.final_ln_gamma)
        # Captured target auxiliary hidden exported to the Eagle3 draft. Two
        # storages hold it for different execution phases:
        #   * graph buffer: fixed address/capacity for CUDA-graph target verify
        #     replay, which cannot allocate or rebind Python tensors during replay.
        #   * eager tensor: exact-size, allocated per non-graph prefill forward,
        #     whose row count can exceed the verify graph capacity.
        # Capacity covers the largest verify batch: max_batch * (gamma + 1).
        self._mtp_target_hidden_graph_capacity = max_generate_batch_size * max(
            int(config.gen_num_per_cycle) + 1, 1
        )
        self.register_buffer(
            "_mtp_target_hidden_graph_buffer",
            (
                torch.empty(
                    (self._mtp_target_hidden_graph_capacity, capture_width),
                    dtype=final_norm_weight.dtype,
                    device=final_norm_weight.device,
                )
                if capture_width > 0
                else None
            ),
            persistent=False,
        )
        self._mtp_target_hidden_eager_tensor: Optional[torch.Tensor] = None
        self._mtp_target_hidden_eager_valid_tokens = 0

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        captured_count = 0
        captured_rows = hidden_states.size(0)
        capture_buffer: Optional[torch.Tensor] = None
        is_target_verify = False
        if self._capture_aux_hidden_layer_ids:
            attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
            is_target_verify = bool(attention_inputs.is_target_verify)
            if is_target_verify:
                capture_buffer = self._mtp_target_hidden_graph_buffer
                if capture_buffer is None:
                    raise RuntimeError("Qwen3 Eagle3 target verify buffer is missing")
                if captured_rows > capture_buffer.size(0):
                    raise ValueError(
                        "Qwen3 Eagle3 target hidden states exceed Graph capacity: "
                        f"rows={captured_rows}, capacity={capture_buffer.size(0)}"
                    )
            else:
                capture_buffer = hidden_states.new_empty(
                    captured_rows,
                    len(self._capture_aux_hidden_layer_ids) * self.config.hidden_size,
                )
                self._mtp_target_hidden_eager_tensor = capture_buffer
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            if i in self._capture_aux_hidden_layer_id_set:
                if capture_buffer is None:
                    raise RuntimeError("Qwen3 Eagle3 target hidden buffer is missing")
                start = captured_count * self.config.hidden_size
                end = start + self.config.hidden_size
                capture_buffer[:captured_rows, start:end].copy_(hidden_states)
                captured_count += 1
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            hidden_states = decoder_layer(
                hidden_states,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        if self._capture_aux_hidden_layer_ids:
            if captured_count != len(self._capture_aux_hidden_layer_ids):
                raise RuntimeError(
                    "Qwen3 Eagle3 did not capture every configured hidden state"
                )
            if not is_target_verify:
                self._mtp_target_hidden_eager_valid_tokens = captured_rows
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states)

    def get_mtp_target_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        # num_tokens < 0: eager prefill reads the producer's exact row count.
        # num_tokens >= 0: graph target verify reads an explicit row count,
        # since graph replay does not update the Python-side valid-token counter.
        if num_tokens < 0:
            buffer = self._mtp_target_hidden_eager_tensor
            requested = self._mtp_target_hidden_eager_valid_tokens
        else:
            buffer = self._mtp_target_hidden_graph_buffer
            requested = int(num_tokens)
        if buffer is None or requested <= 0:
            return None
        if requested > buffer.size(0):
            raise ValueError(
                "Qwen3 Eagle3 hidden-state request exceeds buffer capacity: "
                f"requested={requested}, capacity={buffer.size(0)}"
            )
        return buffer[:requested]

    def has_mtp_hidden_buffer(self) -> bool:
        return bool(self._capture_aux_hidden_layer_ids)


__all__ = [
    "AngelSlimQwen3Eagle3Model",
    "Qwen3Model",
]
