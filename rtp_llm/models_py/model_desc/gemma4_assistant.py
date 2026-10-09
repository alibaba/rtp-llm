from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.gemma4_assistant_weight import (
    GEMMA4_ASSISTANT_POST_PROJECTION_W,
    GEMMA4_ASSISTANT_PRE_PROJECTION_W,
    GEMMA4_ASSISTANT_Q_PROJ_W,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_gather, all_reduce
from rtp_llm.models_py.model_desc.block_map import get_attention_inputs_value
from rtp_llm.models_py.model_desc.gemma4 import (
    GEMMA4_TAG_FULL,
    GEMMA4_TAG_SWA,
    Gemma4RMSNorm,
    Gemma4RopeTable,
    Gemma4TorchFMHAImpl,
    apply_gemma4_rope,
    build_gemma4_layer_geometry,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Gemma4AssistantOutput:
    draft_hidden_states: torch.Tensor
    backbone_hidden_states: torch.Tensor
    logits: torch.Tensor
    hidden_states: Tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Gemma4AssistantLayerConfig:
    layer_type: str
    hidden_size: int
    intermediate_size: int
    head_num: int
    kv_head_num: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    sliding_window: int
    rms_norm_eps: float
    attn_tp_size: int = 1
    ffn_tp_size: int = 1


class Gemma4AssistantAttention(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        config: Gemma4AssistantLayerConfig,
    ):
        super().__init__()
        self.config = config
        self.q_proj_weight = nn.Parameter(
            weights["self_attn.q_proj.weight"], requires_grad=False
        )
        self.o_proj_weight = nn.Parameter(
            weights["self_attn.o_proj.weight"], requires_grad=False
        )
        self.q_norm = Gemma4RMSNorm(
            weights["self_attn.q_norm.weight"], config.rms_norm_eps
        )
        self.rope = Gemma4RopeTable(
            config.head_dim,
            config.rope_theta,
            config.partial_rotary_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, tokens, _ = hidden_states.shape
        query = F.linear(hidden_states, self.q_proj_weight).reshape(
            batch * tokens, self.config.head_num, self.config.head_dim
        )
        query = self.q_norm(query)
        cos, sin = self.rope.cos_sin(position_ids.reshape(-1))
        query = apply_gemma4_rope(query, cos, sin)
        query = query.reshape(
            batch, tokens, self.config.head_num, self.config.head_dim
        ).transpose(1, 2)

        key, value = shared_kv
        key = key.to(device=query.device, dtype=query.dtype)
        value = value.to(device=query.device, dtype=query.dtype)
        if self.config.head_num != self.config.kv_head_num:
            repeats = self.config.head_num // self.config.kv_head_num
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        attention_weights = torch.matmul(query, key.transpose(-1, -2))
        if attention_mask is not None:
            attention_weights = attention_weights + attention_mask
        probabilities = F.softmax(attention_weights, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(probabilities, value)
        output = output.transpose(1, 2).reshape(
            batch, tokens, self.config.head_num * self.config.head_dim
        )
        output = F.linear(output, self.o_proj_weight)
        if self.config.attn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        fmha_impl: Gemma4TorchFMHAImpl,
        kv_cache: LayerKVCache,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]
        query = F.linear(hidden_states, self.q_proj_weight).reshape(
            total_tokens, self.config.head_num, self.config.head_dim
        )
        query = self.q_norm(query)
        positions = position_ids.reshape(-1).to(device=query.device, dtype=torch.long)
        if positions.numel() != total_tokens:
            raise RuntimeError(
                f"assistant position rows {positions.numel()} do not match query rows {total_tokens}"
            )
        cos, sin = self.rope.cos_sin(positions)
        query = apply_gemma4_rope(query, cos, sin)

        lengths, starts = fmha_impl._request_layout()
        if starts[-1] != total_tokens:
            raise RuntimeError(
                f"assistant query rows {total_tokens} do not match attention layout {starts[-1]}"
            )
        k_cache, v_cache = fmha_impl._paged_views(kv_cache)
        page_indptr = fmha_impl.fmha_params.decode_page_indptr_d.cpu().tolist()
        last_page_len = fmha_impl.fmha_params.paged_kv_last_page_len_d.cpu().tolist()

        outputs = []
        for request_idx, _ in enumerate(lengths):
            start, end = starts[request_idx], starts[request_idx + 1]
            keys, values, valid = fmha_impl._gather_request(
                request_idx, page_indptr, last_page_len, k_cache, v_cache
            )
            request_query = query[start:end].transpose(0, 1)
            request_keys = keys.transpose(0, 1)
            request_values = values.transpose(0, 1)
            if self.config.head_num != self.config.kv_head_num:
                repeats = self.config.head_num // self.config.kv_head_num
                request_keys = request_keys.repeat_interleave(repeats, dim=0)
                request_values = request_values.repeat_interleave(repeats, dim=0)

            query_positions = positions[start:end]
            key_positions = torch.arange(keys.shape[0], device=query.device)
            # KV visibility follows logical cache slots, not multimodal RoPE coordinates.
            cache_query_positions = torch.full_like(query_positions, keys.shape[0] - 1)
            allowed = valid.unsqueeze(0) & (
                key_positions.unsqueeze(0) <= cache_query_positions.unsqueeze(1)
            )
            if self.config.layer_type == "sliding_attention":
                allowed = allowed & (
                    key_positions.unsqueeze(0)
                    > cache_query_positions.unsqueeze(1) - self.config.sliding_window
                )
            if not bool(allowed.any(dim=1).all()):
                raise RuntimeError(
                    f"assistant request {request_idx} has a query without visible target KV"
                )

            scores = torch.matmul(request_query, request_keys.transpose(-1, -2))
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(
                query.dtype
            )
            request_output = torch.matmul(probabilities, request_values)
            outputs.append(
                request_output.transpose(0, 1).reshape(
                    end - start, self.config.head_num * self.config.head_dim
                )
            )

        output = F.linear(torch.cat(outputs, dim=0), self.o_proj_weight)
        if self.config.attn_tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output


class Gemma4AssistantDecoderLayer(nn.Module):
    def __init__(
        self,
        weights: Mapping[str, torch.Tensor],
        config: Gemma4AssistantLayerConfig,
    ):
        super().__init__()
        self.config = config
        self.input_layernorm = Gemma4RMSNorm(
            weights["input_layernorm.weight"], config.rms_norm_eps
        )
        self.attention = Gemma4AssistantAttention(weights, config)
        self.post_attention_layernorm = Gemma4RMSNorm(
            weights["post_attention_layernorm.weight"], config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            weights["pre_feedforward_layernorm.weight"], config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            weights["post_feedforward_layernorm.weight"], config.rms_norm_eps
        )
        self.gate_weight = nn.Parameter(
            weights["mlp.gate_proj.weight"], requires_grad=False
        )
        self.up_weight = nn.Parameter(
            weights["mlp.up_proj.weight"], requires_grad=False
        )
        self.down_weight = nn.Parameter(
            weights["mlp.down_proj.weight"], requires_grad=False
        )
        self.layer_scalar = nn.Parameter(weights["layer_scalar"], requires_grad=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states, position_ids, shared_kv, attention_mask
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        gate = F.linear(hidden_states, self.gate_weight)
        up = F.linear(hidden_states, self.up_weight)
        hidden_states = F.gelu(gate, approximate="tanh") * up
        hidden_states = F.linear(hidden_states, self.down_weight)
        if self.config.ffn_tp_size > 1:
            hidden_states = all_reduce(hidden_states, group=Group.TP)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return (residual + hidden_states) * self.layer_scalar

    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        fmha_impl: Gemma4TorchFMHAImpl,
        kv_cache: LayerKVCache,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention.forward_paged(
            hidden_states, position_ids, fmha_impl, kv_cache
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        gate = F.linear(hidden_states, self.gate_weight)
        up = F.linear(hidden_states, self.up_weight)
        hidden_states = F.gelu(gate, approximate="tanh") * up
        hidden_states = F.linear(hidden_states, self.down_weight)
        if self.config.ffn_tp_size > 1:
            hidden_states = all_reduce(hidden_states, group=Group.TP)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return (residual + hidden_states) * self.layer_scalar


class Gemma4AssistantModel(nn.Module):
    def __init__(self, config: Mapping, weights: Mapping[str, torch.Tensor]):
        super().__init__()
        text_config = config["text_config"]
        hidden_size = int(text_config["hidden_size"])
        backbone_hidden_size = int(config["backbone_hidden_size"])
        self.pre_projection_weight = nn.Parameter(
            weights["pre_projection.weight"], requires_grad=False
        )
        self.post_projection_weight = nn.Parameter(
            weights["post_projection.weight"], requires_grad=False
        )
        self.embedding_weight = nn.Parameter(
            weights["model.embed_tokens.weight"], requires_grad=False
        )
        self.final_norm = Gemma4RMSNorm(
            weights["model.norm.weight"], float(text_config["rms_norm_eps"])
        )
        self.layers = nn.ModuleList()
        layer_types = list(text_config["layer_types"])
        for layer_idx, layer_type in enumerate(layer_types):
            full_attention = layer_type == "full_attention"
            prefix = f"model.layers.{layer_idx}."
            layer_weights = {
                key[len(prefix) :]: value
                for key, value in weights.items()
                if key.startswith(prefix)
            }
            self.layers.append(
                Gemma4AssistantDecoderLayer(
                    layer_weights,
                    Gemma4AssistantLayerConfig(
                        layer_type=layer_type,
                        hidden_size=hidden_size,
                        intermediate_size=int(text_config["intermediate_size"]),
                        head_num=int(text_config["num_attention_heads"]),
                        kv_head_num=int(
                            text_config["num_global_key_value_heads"]
                            if full_attention
                            else text_config["num_key_value_heads"]
                        ),
                        head_dim=int(
                            text_config["global_head_dim"]
                            if full_attention
                            else text_config["head_dim"]
                        ),
                        rope_theta=float(
                            text_config["rope_parameters"][layer_type]["rope_theta"]
                        ),
                        partial_rotary_factor=float(
                            text_config["rope_parameters"][layer_type].get(
                                "partial_rotary_factor", 1.0
                            )
                        ),
                        sliding_window=int(text_config["sliding_window"]),
                        rms_norm_eps=float(text_config["rms_norm_eps"]),
                    ),
                )
            )
        if len(self.layers) != int(text_config["num_hidden_layers"]):
            raise ValueError("assistant layer count does not match config")
        if int(text_config["num_kv_shared_layers"]) != len(self.layers):
            raise ValueError("all Gemma4 assistant layers must share target KV")
        if self.pre_projection_weight.shape != (
            hidden_size,
            2 * backbone_hidden_size,
        ):
            raise ValueError("assistant pre_projection shape does not match config")
        if self.post_projection_weight.shape != (
            backbone_hidden_size,
            hidden_size,
        ):
            raise ValueError("assistant post_projection shape does not match config")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv_states: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
        attention_masks: Optional[Mapping[str, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None or shared_kv_states is None:
            raise ValueError("inputs_embeds and shared_kv_states cannot be None")
        hidden_states = F.linear(inputs_embeds, self.pre_projection_weight)
        layer_hidden_states = [hidden_states]
        for layer in self.layers:
            layer_type = layer.config.layer_type
            if layer_type not in shared_kv_states:
                raise KeyError(f"missing shared KV for {layer_type}")
            attention_mask = (
                attention_masks.get(layer_type) if attention_masks is not None else None
            )
            hidden_states = layer(
                hidden_states,
                position_ids,
                shared_kv_states[layer_type],
                attention_mask,
            )
            layer_hidden_states.append(hidden_states)
        draft_hidden_states = self.final_norm(hidden_states)
        layer_hidden_states[-1] = draft_hidden_states
        backbone_hidden_states = F.linear(
            draft_hidden_states, self.post_projection_weight
        )
        logits = F.linear(draft_hidden_states, self.embedding_weight)
        return Gemma4AssistantOutput(
            draft_hidden_states=draft_hidden_states,
            backbone_hidden_states=backbone_hidden_states,
            logits=logits,
            hidden_states=tuple(layer_hidden_states),
        )


class Gemma4AssistantRuntimeModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weights: ModelWeights,
        max_generate_batch_size: int,
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
        self.attn_tp_size = parallelism_config.get_attn_tp_size()
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        metadata = config.mm_related_params.config
        self.backbone_hidden_size = int(metadata["assistant_backbone_hidden_size"])
        self.pre_projection_weight = nn.Parameter(
            weights.get_global_weight(GEMMA4_ASSISTANT_PRE_PROJECTION_W),
            requires_grad=False,
        )
        self.post_projection_weight = nn.Parameter(
            weights.get_global_weight(GEMMA4_ASSISTANT_POST_PROJECTION_W),
            requires_grad=False,
        )
        self.final_norm = Gemma4RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), config.layernorm_eps
        )
        self.geometries = [
            build_gemma4_layer_geometry(config, parallelism_config, layer_idx)
            for layer_idx in range(config.num_layers)
        ]
        self.layers = nn.ModuleList()
        for layer_idx, geometry in enumerate(self.geometries):
            layer_weights = weights.weights[layer_idx]
            self.layers.append(
                Gemma4AssistantDecoderLayer(
                    {
                        "self_attn.q_proj.weight": layer_weights[
                            GEMMA4_ASSISTANT_Q_PROJ_W
                        ].transpose(0, 1),
                        "self_attn.o_proj.weight": layer_weights[W.attn_o_w].transpose(
                            0, 1
                        ),
                        "self_attn.q_norm.weight": layer_weights[W.q_ln_gamma],
                        "input_layernorm.weight": layer_weights[W.pre_ln_gamma],
                        "post_attention_layernorm.weight": layer_weights[
                            W.post_ln_gamma
                        ],
                        "pre_feedforward_layernorm.weight": layer_weights[
                            W.pre_ffn_ln_gamma
                        ],
                        "post_feedforward_layernorm.weight": layer_weights[
                            W.post_ffn_ln_gamma
                        ],
                        "mlp.gate_proj.weight": layer_weights[W.ffn_w1].transpose(0, 1),
                        "mlp.up_proj.weight": layer_weights[W.ffn_w3].transpose(0, 1),
                        "mlp.down_proj.weight": layer_weights[W.ffn_w2].transpose(0, 1),
                        "layer_scalar": layer_weights[W.layer_scalar],
                    },
                    Gemma4AssistantLayerConfig(
                        layer_type=(
                            "full_attention"
                            if geometry.tag == GEMMA4_TAG_FULL
                            else "sliding_attention"
                        ),
                        hidden_size=config.hidden_size,
                        intermediate_size=config.inter_size,
                        head_num=geometry.head_num,
                        kv_head_num=geometry.kv_head_num,
                        head_dim=geometry.head_dim,
                        rope_theta=geometry.rope_theta,
                        partial_rotary_factor=geometry.rope_partial_rotary_factor,
                        sliding_window=geometry.sliding_window,
                        rms_norm_eps=config.layernorm_eps,
                        attn_tp_size=self.attn_tp_size,
                        ffn_tp_size=self.ffn_tp_size,
                    ),
                )
            )
        self.target_embedding: Optional[torch.Tensor] = None
        self.target_embedding_scalar = 1.0

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        super().initialize(init_resource)
        target_embedding = init_resource.speculative_target_embedding
        if target_embedding is None or target_embedding.numel() == 0:
            raise RuntimeError("Gemma4 Assistant requires the target embedding")
        if self.backbone_hidden_size % self.attn_tp_size != 0:
            raise RuntimeError(
                "Gemma4 Assistant target hidden size must be divisible by attention TP"
            )
        local_backbone_hidden_size = self.backbone_hidden_size // self.attn_tp_size
        if tuple(target_embedding.shape) != (
            self.config.vocab_size,
            local_backbone_hidden_size,
        ):
            raise RuntimeError(
                "Gemma4 Assistant target embedding shape mismatch: "
                f"got {tuple(target_embedding.shape)}, expected "
                f"({self.config.vocab_size}, {local_backbone_hidden_size})"
            )
        if self.kv_cache is None or self.kv_cache.layer_count != self.layer_num:
            raise RuntimeError(
                "Gemma4 Assistant requires four projected target KV layers"
            )
        self.target_embedding = target_embedding
        self.target_embedding_scalar = float(
            torch.tensor(
                init_resource.speculative_target_embedding_scalar,
                dtype=target_embedding.dtype,
            )
        )
        return True

    def _layer_kv_cache(self, layer_idx: int, tag: str) -> LayerKVCache:
        if self.kv_cache is None:
            raise RuntimeError("Gemma4 Assistant target KV is not initialized")
        groups = self.kv_cache.get_layer_cache_groups(layer_idx)
        for cache in groups:
            if str(cache.tag) == tag:
                return cache
        raise RuntimeError(
            f"assistant layer {layer_idx} has no projected KV tag {tag!r}"
        )

    def _page_size_for_tag(self, tag: str) -> int:
        if self.kv_cache is None:
            raise RuntimeError("Gemma4 Assistant target KV is not initialized")
        return int(self.kv_cache.get_kernel_seq_size_per_block(tag))

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Dict[str, Gemma4TorchFMHAImpl]:
        del cuda_graph_selection_mode
        if is_cuda_graph:
            raise RuntimeError("Gemma4 Assistant shared-KV graph is not implemented")
        attention_inputs = get_attention_inputs_value(inputs)
        if not isinstance(attention_inputs, Mapping):
            raise RuntimeError(
                "Gemma4 Assistant requires tagged swa/full attention inputs"
            )
        implementations: Dict[str, Gemma4TorchFMHAImpl] = {}
        for tag in (GEMMA4_TAG_SWA, GEMMA4_TAG_FULL):
            if tag not in attention_inputs:
                raise RuntimeError(
                    f"Gemma4 Assistant attention input is missing {tag!r}"
                )
            geometry = next(item for item in self.geometries if item.tag == tag)
            implementations[tag] = Gemma4TorchFMHAImpl(
                self.config,
                geometry,
                attention_inputs[tag],
                self._page_size_for_tag(tag),
                parallelism_config=self.parallelism_config,
            )
        return implementations

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if self.target_embedding is None:
            raise RuntimeError("Gemma4 Assistant target embedding is not initialized")
        if inputs.input_hiddens is None or inputs.input_hiddens.numel() == 0:
            raise RuntimeError("Gemma4 Assistant requires target hidden states")
        if inputs.input_ids.numel() != inputs.input_hiddens.shape[0]:
            raise RuntimeError(
                "Gemma4 Assistant token/hidden row mismatch: "
                f"{inputs.input_ids.numel()} vs {inputs.input_hiddens.shape[0]}"
            )
        if inputs.input_hiddens.shape[-1] != self.backbone_hidden_size:
            raise RuntimeError(
                "Gemma4 Assistant target hidden width mismatch: "
                f"{inputs.input_hiddens.shape[-1]} vs {self.backbone_hidden_size}"
            )
        target_embeddings = F.embedding(
            inputs.input_ids.to(torch.long), self.target_embedding
        )
        if self.attn_tp_size > 1:
            tokens, local_hidden_size = target_embeddings.shape
            target_embeddings = all_gather(target_embeddings, group=Group.TP)
            target_embeddings = (
                target_embeddings.reshape(self.attn_tp_size, tokens, local_hidden_size)
                .transpose(0, 1)
                .reshape(tokens, self.backbone_hidden_size)
            )
        target_embeddings = target_embeddings * self.target_embedding_scalar
        hidden_states = torch.cat(
            [target_embeddings, inputs.input_hiddens.to(target_embeddings.dtype)],
            dim=-1,
        )
        hidden_states = torch.matmul(hidden_states, self.pre_projection_weight)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        if not isinstance(fmha_impl, Mapping):
            raise RuntimeError(
                "Gemma4 Assistant requires tagged attention implementations"
            )
        if inputs.combo_position_ids is None or inputs.combo_position_ids.numel() == 0:
            position_ids = next(iter(fmha_impl.values())).positions(
                inputs.input_ids.numel()
            )
        else:
            position_ids = inputs.combo_position_ids

        for layer_idx, layer in enumerate(self.layers):
            tag = self.geometries[layer_idx].tag
            hidden_states = layer.forward_paged(
                hidden_states,
                position_ids,
                fmha_impl[tag],
                self._layer_kv_cache(layer_idx, tag),
            )
        draft_hidden_states = self.final_norm(hidden_states)
        backbone_hidden_states = torch.matmul(
            draft_hidden_states, self.post_projection_weight
        )
        return PyModelOutputs(draft_hidden_states, backbone_hidden_states)


__all__ = [
    "Gemma4AssistantAttention",
    "Gemma4AssistantDecoderLayer",
    "Gemma4AssistantLayerConfig",
    "Gemma4AssistantModel",
    "Gemma4AssistantOutput",
    "Gemma4AssistantRuntimeModel",
]
