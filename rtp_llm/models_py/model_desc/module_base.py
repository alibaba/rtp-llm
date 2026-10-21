import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.pp_layout import (
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
from rtp_llm.models_py.model_desc.hidden_state_capture import (
    CanonicalHiddenHook,
    CaptureContext,
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
        apply_pp_partition: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self._apply_pp_partition = apply_pp_partition
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
        self._mtp_aux_capture_expected_parts = 0

        self._capture_context = CaptureContext.unsupported()

    def _init_capture_context(
        self,
        canonical_layer: CanonicalHiddenHook,
        canonical_final: CanonicalHiddenHook,
    ) -> None:
        self._capture_context = CaptureContext.configured(
            self.config.hidden_state_capture_layer_ids,
            canonical_layer,
            canonical_final,
        )

    def _init_capture_passthrough(self) -> None:
        self._capture_context = CaptureContext.passthrough()

    def _disable_capture_context(self) -> None:
        self._capture_context = CaptureContext.unsupported()

    @property
    def supports_hidden_state_capture(self) -> bool:
        return self._capture_context.supported

    def capture_context(self, requested: bool) -> CaptureContext:
        return self._capture_context.for_forward(requested)

    # Pipeline-parallel stage view; layout delegates to config/pp_layout.py.
    @property
    def pp_size(self) -> int:
        return self.parallelism_config.pp_size

    @property
    def pp_rank(self) -> int:
        return self.parallelism_config.pp_rank

    @property
    def pp_has_embedding(self) -> bool:
        """The first target stage or a complete draft owns its embedding."""
        return not self._apply_pp_partition or stage_has_embedding(self.pp_rank)

    @property
    def pp_has_lm_head(self) -> bool:
        """Only the last stage owns lm_head and final layernorm.

        Draft placement is already restricted to the last stage, so model
        construction scope does not change output-module ownership.
        """
        return stage_has_lm_head(self.pp_rank, self.pp_size)

    def pp_layer_ids(self) -> list[int]:
        """Global layer ids owned by this stage.

        Lookup over the materialized partition on ParallelismConfig
        (decided once at startup), so model construction stays in sync
        with weight loading; pp_size=1 is trivially all layers.
        A draft uses all of its own layers while retaining the physical
        ParallelismConfig for TP/EP communication on its owning stage.
        """
        if not self._apply_pp_partition:
            return list(range(self.layer_num))
        counts = getattr(self.parallelism_config, "pp_stage_layer_counts", None)
        return list(
            stage_layer_range(self.layer_num, self.pp_size, self.pp_rank, counts)
        )

    def make_empty_intermediate_tensors(self, hidden_template: Any) -> dict[str, Any]:
        """Build this model's stage-boundary inputs for stage-local PP warmup."""
        return {"hidden_states": hidden_template}

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

    def _aux_hidden_capture_parts(self) -> tuple[int, int]:
        """Count configured boundaries before and through this model's layers."""
        if not self._mtp_aux_capture_layer_ids:
            return 0, 0
        layer_ids = self.pp_layer_ids()
        previous_parts = sum(
            layer_id < layer_ids[0] for layer_id in self._mtp_aux_capture_layer_ids
        )
        expected_parts = sum(
            layer_id <= layer_ids[-1] for layer_id in self._mtp_aux_capture_layer_ids
        )
        return previous_parts, expected_parts

    def begin_aux_hidden_capture(
        self,
        hidden_states: torch.Tensor,
        is_target_verify: bool,
        upstream_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        """Capture this model's configured layers after any upstream prefix.

        Prompt and verify use separate grow-only buffers so prompt growth does
        not change the addresses used by decode graphs. The executor snapshots
        PP outputs before another forward can reuse these buffers.
        """
        self._mtp_target_hidden_states = None
        self._mtp_aux_capture_buffer = None
        self._mtp_aux_capture_rows = int(hidden_states.shape[0])
        previous_parts, expected_parts = self._aux_hidden_capture_parts()
        # Local captures append after the slots supplied by upstream stages.
        self._mtp_aux_capture_index = previous_parts
        self._mtp_aux_capture_expected_parts = expected_parts

        if expected_parts == 0:
            return

        hidden_width = int(hidden_states.shape[-1])
        width = expected_parts * hidden_width
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
        self._mtp_aux_capture_buffer = buffer[: self._mtp_aux_capture_rows]
        if previous_parts:
            prefix_width = previous_parts * hidden_width
            self._mtp_aux_capture_buffer[:, :prefix_width].copy_(upstream_hidden)

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
        """Validate this stage's prefix; only a complete target publishes it."""
        expected_parts = self._mtp_aux_capture_expected_parts
        if expected_parts == 0:
            return
        if self._mtp_aux_capture_index != expected_parts:
            raise RuntimeError(
                "auxiliary hidden capture missed configured layers: "
                f"captured={self._mtp_aux_capture_index}, expected={expected_parts}"
            )
        if self._mtp_aux_capture_buffer is None:
            raise RuntimeError("auxiliary hidden capture buffer is unavailable")
        if not self._apply_pp_partition or self.pp_has_lm_head:
            self._mtp_target_hidden_states = self._mtp_aux_capture_buffer

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
