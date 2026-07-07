"""Qwen3-VL-MoE top-level container for the new loader.

结构与 ``new_models/qwen3_vl/model.py`` 完全平行,只把 language_model 换成 MoE 版;
视觉塔直接复用 qwen3_vl 的 ``Qwen3VLVisionTransformer``(HF Qwen3VL(Moe) 视觉塔)。
"""

from typing import Any

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLVisionTransformer
from rtp_llm.models_py.new_models.qwen3_vl_moe.language import Qwen3VLMoeForCausalLM
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen3VLMoeForConditionalGeneration(RtpModule):

    # 与 qwen3_vl 相同:HF ckpt 把语言嵌在 model.language_model.、视觉在 model.visual. 下。
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config

        self.visual = Qwen3VLVisionTransformer(
            model_config=model_config, load_config=load_config
        )
        self.language_model = Qwen3VLMoeForCausalLM(
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

    def forward(self, inputs, fmha_impl: Any = None):
        return self.language_model.forward(inputs, fmha_impl=fmha_impl)
