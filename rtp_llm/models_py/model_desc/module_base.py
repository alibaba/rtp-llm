import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import Tensor, nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttentionImpl
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class GptModelBase(nn.Module):

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num: int = config.num_layers
        self.vocab_size: int = config.vocab_size

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        if self.kv_cache is not None:
            num_layers = self.kv_cache.layer_count
            layer0_caches = (
                self.kv_cache.get_layer_cache_groups(0) if num_layers > 0 else []
            )
            layer0_shapes = [cache.kv_cache_base.shape for cache in layer0_caches]
            layer0_scale_count = sum(
                cache.kv_scale_base is not None and cache.kv_scale_base.numel() > 0
                for cache in layer0_caches
            )
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shapes={layer0_shapes}, "
                f"layer0_scale_groups={layer0_scale_count}, "
            )
        return True

    def embedding_cuda_graph_unsupported_reason(self) -> str:
        """Capability query, not a Graph enable switch. Empty means supported."""
        if (
            type(self).prepare_transformer_inputs
            is GptModelBase.prepare_transformer_inputs
            or type(self).forward_transformer is GptModelBase.forward_transformer
        ):
            return "model has no split embedding/Transformer implementation"
        if self.kv_cache is not None:
            return "embedding graphs require a cache-free model"
        if getattr(self.config, "use_attention_linear_bias", False):
            return "embedding graph attention linear bias requires an adapter"
        if getattr(self.config, "headwise_config", None) is not None:
            return "embedding graph headwise attention requires an adapter"
        if self.parallelism_config.prefill_cp_config.is_enabled():
            return "embedding graph context parallelism is not implemented"
        from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import (
            CacheFreePrefillGraphImpl,
        )

        configs = self.config.getAttentionConfigs(
            self.parallelism_config.get_attn_tp_size()
        )
        return CacheFreePrefillGraphImpl.unsupported_reason(configs)

    def prepare_transformer_inputs(self, inputs: PyModelInputs) -> Tensor:
        """Compute embeddings outside capture. Return [real_tokens, hidden_size]."""
        raise NotImplementedError("model must implement embedding input preparation")

    def forward_transformer(
        self, hidden_states: Tensor, fmha_impl: Any
    ) -> PyModelOutputs:
        """Run the backbone, including model final norm, without pooling/heads."""
        raise NotImplementedError("model must implement Transformer-only forward")

    def prepare_embedding_attention_inputs(
        self, attn_inputs, host_tokens, host_text_mask
    ):
        """Prepare model metadata from host views; never read pending H2D copies."""
        return attn_inputs

    def validate_embedding_graph_inputs(self, inputs: PyModelInputs) -> bool:
        """Side-effect-free request eligibility, before backend planning."""
        attention = get_attention_inputs_value(inputs)
        groups = attention.values() if isinstance(attention, Mapping) else [attention]
        for attn in groups:
            lengths = attn.input_lengths
            prefix = attn.prefix_lengths
            if (
                not attn.is_prefill
                or attn.is_target_verify
                or lengths is None
                or lengths.is_cuda
                or lengths.dtype != torch.int32
                or lengths.ndim != 1
                or not lengths.is_contiguous()
                or lengths.numel() == 0
                or bool((lengths < 0).any())
                or bool((lengths > self.config.max_seq_len).any())
            ):
                return False
            tokens = getattr(inputs, "input_ids", None)
            if tokens is not None and tokens.numel():
                if (
                    tokens.ndim != 1
                    or tokens.dtype != torch.int32
                    or not tokens.is_cuda
                    or not tokens.is_contiguous()
                    or tokens.numel() != int(lengths.sum())
                ):
                    return False
            if (
                prefix is None
                or prefix.is_cuda
                or prefix.dtype != torch.int32
                or prefix.ndim != 1
            ):
                return False
            if (
                prefix.numel() != lengths.numel()
                or not prefix.is_contiguous()
                or bool(prefix.any())
            ):
                return False
        return self.validate_embedding_input_features(inputs)

    def validate_embedding_input_features(self, inputs):
        # The default adapter reconstructs positions from zero for each sequence.
        # Models with explicit positions must provide their own replay adapter.
        attention = get_attention_inputs_value(inputs)
        groups = attention.values() if isinstance(attention, Mapping) else [attention]
        for source in [inputs, *groups]:
            positions = getattr(source, "combo_position_ids", None)
            if positions is not None and positions.numel():
                return False
        mm = inputs.multimodal_inputs
        mask = inputs.embedding_inputs.text_tokens_mask
        return (
            not mm.multimodal_features
            and (mm.mm_features_locs is None or mm.mm_features_locs.numel() == 0)
            and (mask is None or mask.numel() == 0)
        )

    def prepare_embedding_graph_attention(
        self, inputs: PyModelInputs, is_cuda_graph=True
    ):
        return self.prepare_fmha_impl(inputs, True, "embedding_graph")

    def prepare_embedding_graph_replay(self, attention, graph_inputs, request_inputs):
        """Plan against fixed Graph buffers, retaining request addresses outside capture."""
        if attention is None:
            return
        graph_attention = get_attention_inputs_value(graph_inputs)
        if isinstance(attention, Mapping):
            for tag, impl in attention.items():
                impl.prepare_cuda_graph(
                    select_attention_inputs_for_tag(graph_attention, tag)
                )
        else:
            attention.prepare_cuda_graph(graph_attention)

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> AttentionImpl | dict[str, AttentionImpl]:
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            fmha_group_tags = self._get_fmha_group_tags()
            selected_group_inputs = (
                attention_inputs.items()
                if fmha_group_tags is None
                else (
                    (tag, select_attention_inputs_for_tag(attention_inputs, tag))
                    for tag in fmha_group_tags
                )
            )
            return {
                tag: AttnImplFactory.get_fmha_impl(
                    self.config,
                    self.parallelism_config,
                    self.weight,
                    group_inputs,
                    self.fmha_config,
                    is_cuda_graph,
                    cuda_graph_selection_mode,
                )
                for tag, group_inputs in selected_group_inputs
            }
        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
            is_cuda_graph,
            cuda_graph_selection_mode,
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def get_inputs_embeds(self, input_ids: Tensor, inputs: PyModelInputs) -> Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return self.apply_input_embeddings(inputs_embeds, inputs)

    def apply_input_embeddings(
        self, inputs_embeds: Tensor, inputs: PyModelInputs
    ) -> Tensor:
        # Unit-level callers and external model adapters may provide the
        # PyModelInputs protocol through a lightweight namespace. Keep those
        # adapters source-compatible with the graph-only optional fields.
        graph_overrides = getattr(inputs, "cuda_graph_input_embedding_overrides", None)
        graph_mask = getattr(inputs, "cuda_graph_input_embedding_mask", None)
        if graph_overrides is not None and graph_overrides.numel() > 0:
            if graph_mask is None or graph_mask.numel() != inputs_embeds.size(0):
                raise ValueError(
                    "cuda_graph_input_embedding_mask must contain one entry per token"
                )
            if graph_overrides.shape != inputs_embeds.shape:
                raise ValueError(
                    "cuda_graph_input_embedding_overrides must match inputs_embeds"
                )
            return torch.where(graph_mask.unsqueeze(1), graph_overrides, inputs_embeds)

        if inputs.input_embeddings is not None and len(inputs.input_embeddings) > 0:
            locs = inputs.input_embeddings_locs
            if locs is None:
                raise ValueError("input_embeddings_locs must be set")
            if inputs_embeds.dim() != 2:
                raise ValueError(
                    "inputs_embeds must be a 2D tensor of shape [tokens, hidden_size]"
                )
            if locs.device.type == "cpu" and locs.dtype in (
                torch.int32,
                torch.int64,
            ):
                loc_values = locs.view(-1).tolist()
            else:
                loc_values = locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            if len(inputs.input_embeddings) != len(loc_values):
                raise ValueError(
                    f"input_embeddings count ({len(inputs.input_embeddings)}) "
                    f"!= input_embeddings_locs count ({len(loc_values)})"
                )
            token_num = inputs_embeds.size(0)
            hidden_size = inputs_embeds.size(1)
            normalized_embeddings = []
            previous_end = 0
            for i, (emb, loc) in enumerate(zip(inputs.input_embeddings, loc_values)):
                if loc < 0:
                    raise ValueError(f"input_embeddings_locs[{i}]={loc} must be >= 0")
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                if emb.dim() != 2:
                    raise ValueError(
                        f"input_embeddings[{i}] must be 1D or 2D, got dim={emb.dim()}"
                    )
                if not emb.is_floating_point():
                    raise ValueError(
                        f"input_embeddings[{i}] must be floating point, got dtype={emb.dtype}"
                    )
                if emb.size(0) <= 0:
                    raise ValueError(f"input_embeddings[{i}] must not be empty")
                if emb.size(1) != hidden_size:
                    raise ValueError(
                        f"input_embeddings[{i}] hidden size {emb.size(1)} "
                        f"!= model hidden size {hidden_size}"
                    )
                emb_len = emb.size(0)
                if loc + emb_len > token_num:
                    raise ValueError(
                        f"input_embeddings[{i}] at loc {loc} with length {emb_len} "
                        f"exceeds token count {token_num}"
                    )
                if loc < previous_end:
                    raise ValueError(
                        f"input_embeddings_locs[{i}]={loc} overlaps or is out of order; "
                        f"previous interval ends at {previous_end}"
                    )
                normalized_embeddings.append((loc, emb))
                previous_end = loc + emb_len
            for loc, emb in normalized_embeddings:
                emb_len = emb.size(0)
                target = inputs_embeds.narrow(0, loc, emb_len)
                if emb.device == target.device:
                    target.copy_(emb)
                else:
                    target.copy_(
                        emb.to(
                            device=target.device,
                            dtype=target.dtype,
                        )
                    )
        return inputs_embeds

    @staticmethod
    def _has_input_embeddings(inputs: PyModelInputs) -> bool:
        return inputs.input_embeddings is not None and len(inputs.input_embeddings) > 0

    def _reject_input_embeddings(self, inputs: PyModelInputs) -> None:
        if self._has_input_embeddings(inputs):
            raise RuntimeError(
                f"{type(self).__name__} does not support input_embeddings."
            )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
