from collections.abc import Mapping

import torch

from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.hidden_state_capture import CaptureContext
from rtp_llm.models_py.model_desc.module_base import GptModelBase, required_config_value


class NewLoaderModelBase(GptModelBase):
    """Generation runtime with the engine's packed hidden-state output contract.

    Each forward records post-layer, pre-final-norm activations and performs its
    normal final normalization before calling capture.finalize(). This is
    independent of DSpARK's auxiliary target-hidden buffer.
    """

    def __init__(
        self,
        config,
        parallelism_config,
        weight=None,
        max_generate_batch_size=0,
        fmha_config=None,
        py_hw_kernel_config=None,
        *,
        load_config=None,
    ):
        if load_config is not None:
            max_generate_batch_size = load_config.max_generate_batch_size
            quant_config = load_config.quant_config
            py_hw_kernel_config = getattr(quant_config, "hw_kernel_config", None)
        super().__init__(
            config,
            parallelism_config,
            weight,
            max_generate_batch_size,
            fmha_config,
            py_hw_kernel_config,
        )
        layer_ids = (
            config.get("hidden_state_capture_layer_ids")
            if isinstance(config, Mapping)
            else getattr(config, "hidden_state_capture_layer_ids", None)
        )
        self._capture_context = CaptureContext.configured(
            layer_ids, self._capture_layer_value, self._capture_final_value
        )

    @staticmethod
    def _capture_layer_value(
        hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        # Residual-add norms can overwrite their inputs in the next layer. Own
        # the captured boundary, never retain an alias to a mutable activation.
        return hidden_states.clone() if residual is None else hidden_states + residual

    @staticmethod
    def _capture_final_value(
        hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        return hidden_states


__all__ = ["NewLoaderModelBase", "required_config_value", "select_fmha_impl_for_layer"]
