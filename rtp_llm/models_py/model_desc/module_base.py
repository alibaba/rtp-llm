import logging
from collections.abc import Mapping
from typing import Any, Optional

from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.pp_layout import (
    derive_pp_rank,
    stage_has_embedding,
    stage_has_lm_head,
    stage_layer_range,
)
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

    # Pipeline-parallel stage view; layout delegates to config/pp_layout.py.
    @property
    def pp_size(self) -> int:
        return max(int(getattr(self.parallelism_config, "pp_size", 1) or 1), 1)

    @property
    def pp_rank(self) -> int:
        rank = getattr(self.parallelism_config, "pp_rank", None)
        if rank is None:
            rank = derive_pp_rank(
                getattr(self.parallelism_config, "world_rank", 0),
                getattr(self.parallelism_config, "dp_size", 1),
                getattr(self.parallelism_config, "tp_size", 1),
            )
        return int(rank)

    @property
    def pp_has_embedding(self) -> bool:
        """First stage owns the token/positional embedding."""
        return stage_has_embedding(self.pp_rank)

    @property
    def pp_has_lm_head(self) -> bool:
        """Last stage owns lm_head + final_layernorm."""
        return stage_has_lm_head(self.pp_rank, self.pp_size)

    def pp_layer_ids(self) -> list[int]:
        """Global layer ids owned by this stage.

        Lookup over the materialized partition on ParallelismConfig
        (decided once at startup), so model construction stays in sync
        with weight loading; pp_size=1 is trivially all layers.
        """
        counts = getattr(self.parallelism_config, "pp_stage_layer_counts", None)
        return list(
            stage_layer_range(self.layer_num, self.pp_size, self.pp_rank, counts)
        )

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

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
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
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
