import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttentionImpl
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    from rtp_llm.model_loader.model_weight_info import ModelWeights


def required_config_value(config: Any, *names: str) -> Any:
    for name in names:
        value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if value is not None:
            return value
    raise ValueError(f"Model config requires one of {names}")


class GptModelBase(RtpModule):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: Optional["ModelWeights"] = None,
        max_generate_batch_size: int = 0,
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
        self.layer_num = required_config_value(
            config, "num_layers", "num_hidden_layers"
        )
        self.vocab_size = required_config_value(config, "vocab_size")

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()
        self._mtp_aux_capture_layer_ids = tuple(
            config.capture_aux_hidden_layer_ids or ()
        )
        self._mtp_aux_capture_layer_id_set = frozenset(
            self._mtp_aux_capture_layer_ids
        )
        self._mtp_target_hidden_states: Optional[torch.Tensor] = None
        self._mtp_target_graph_buffer: Optional[torch.Tensor] = None
        self._mtp_target_prompt_buffer: Optional[torch.Tensor] = None
        self._mtp_aux_capture_buffer: Optional[torch.Tensor] = None
        self._mtp_aux_capture_rows = 0
        self._mtp_aux_capture_index = 0

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
                    model_config=self.config,
                    parallelism_config=self.parallelism_config,
                    weight=self.weight,
                    attn_inputs=group_inputs,
                    fmha_config=self.fmha_config,
                    is_cuda_graph=is_cuda_graph,
                    cuda_graph_selection_mode=cuda_graph_selection_mode,
                )
                for tag, group_inputs in selected_group_inputs
            }
        return AttnImplFactory.get_fmha_impl(
            model_config=self.config,
            parallelism_config=self.parallelism_config,
            weight=self.weight,
            attn_inputs=attention_inputs,
            fmha_config=self.fmha_config,
            is_cuda_graph=is_cuda_graph,
            cuda_graph_selection_mode=cuda_graph_selection_mode,
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def begin_aux_hidden_capture(
        self, hidden_states: torch.Tensor, is_target_verify: bool
    ) -> None:
        """Start one target auxiliary-hidden capture.

        Decode graphs keep one grow-only buffer so every captured batch bucket
        writes to a stable address. Prompt prefill owns a separate grow-only
        buffer because its token count can exceed the decode graph capacity.
        """
        self._mtp_target_hidden_states = None
        self._mtp_aux_capture_buffer = None
        self._mtp_aux_capture_rows = int(hidden_states.shape[0])
        self._mtp_aux_capture_index = 0

        layer_ids = self._mtp_aux_capture_layer_ids
        if not layer_ids:
            return

        width = len(layer_ids) * int(hidden_states.shape[-1])
        buffer = (
            self._mtp_target_graph_buffer
            if is_target_verify
            else self._mtp_target_prompt_buffer
        )
        if (
            buffer is None
            or int(buffer.shape[0]) < self._mtp_aux_capture_rows
            or int(buffer.shape[1]) != width
            or buffer.dtype != hidden_states.dtype
            or buffer.device != hidden_states.device
        ):
            buffer = torch.empty(
                (self._mtp_aux_capture_rows, width),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            if is_target_verify:
                self._mtp_target_graph_buffer = buffer
            else:
                self._mtp_target_prompt_buffer = buffer
        self._mtp_aux_capture_buffer = buffer

    def capture_aux_hidden(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> None:
        """Materialize one configured decoder-layer boundary into the capture."""
        if layer_id not in self._mtp_aux_capture_layer_id_set:
            return

        buffer = self._mtp_aux_capture_buffer
        if buffer is None:
            raise RuntimeError(
                "auxiliary hidden capture was not initialized for this forward"
            )
        hidden_width = int(hidden_states.shape[-1])
        start = self._mtp_aux_capture_index * hidden_width
        target = buffer[: self._mtp_aux_capture_rows, start : start + hidden_width]
        if residual is None:
            target.copy_(hidden_states)
        else:
            # Qwen3Next keeps the layer-boundary value split between the fused
            # residual and hidden tensors. DSpARK was trained on their sum.
            torch.add(hidden_states, residual, out=target)
        self._mtp_aux_capture_index += 1

    def finish_aux_hidden_capture(self) -> None:
        expected_parts = len(self._mtp_aux_capture_layer_ids)
        if expected_parts == 0:
            return
        if self._mtp_aux_capture_index != expected_parts:
            raise RuntimeError(
                "auxiliary hidden capture missed configured layers: "
                f"captured={self._mtp_aux_capture_index}, expected={expected_parts}"
            )
        if self._mtp_aux_capture_buffer is None:
            raise RuntimeError("auxiliary hidden capture buffer is unavailable")
        self._mtp_target_hidden_states = self._mtp_aux_capture_buffer[
            : self._mtp_aux_capture_rows
        ]

    def get_mtp_target_hidden_states(
        self, num_tokens: int
    ) -> Optional[torch.Tensor]:
        hidden = self._mtp_target_hidden_states
        if hidden is None:
            return None
        if num_tokens < 0:
            return hidden
        if num_tokens > hidden.shape[0]:
            raise RuntimeError(
                f"requested {num_tokens} MTP target rows from "
                f"{hidden.shape[0]} captured rows"
            )
        return hidden[:num_tokens]

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
