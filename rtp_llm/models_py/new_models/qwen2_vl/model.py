from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

from rtp_llm.models_py.module_base import rtp_module
from rtp_llm.models_py.new_models.qwen2_vl.language import Qwen2ForCausalLM
from rtp_llm.models_py.new_models.qwen2_vl.vision import Qwen2VisionTransformer
from rtp_llm.models_py.weight_mapper import WeightsMapper


@rtp_module
class Qwen2VLForConditionalGeneration(nn.Module):

    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "visual.": "visual.",
            "model.": "language_model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config
        vit_config = self._get_vit_config(model_config)

        self.visual = Qwen2VisionTransformer(
            vit_config=vit_config, load_config=load_config
        )
        self.language_model = Qwen2ForCausalLM(
            model_config=model_config, load_config=load_config
        )

    def initialize(self, init_resource) -> bool:
        return self.language_model.initialize(init_resource)

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        return self.language_model.prepare_fmha_impl(inputs, is_cuda_graph)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        grouped = self._groupby_prefix(mapped_iter)

        for prefix, sub_weights in grouped.items():
            child = self._get_child_module(prefix)
            if child is not None and hasattr(child, "load_weights"):
                child.load_weights(sub_weights)

    def _get_vit_config(self, model_config) -> dict:
        if hasattr(model_config, "vision_config"):
            return model_config.vision_config
        elif isinstance(model_config, dict):
            return model_config.get("vision_config", {})
        return {
            "hidden_size": 1280,
            "num_heads": 16,
            "num_layers": 32,
            "intermediate_size": 5120,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
        }

    def forward(self, inputs, fmha_impl: Any = None):
        return self.language_model.forward(inputs, fmha_impl=fmha_impl)
