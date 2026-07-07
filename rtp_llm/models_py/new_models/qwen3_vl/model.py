"""Qwen3-VL top-level container for the new loader.

结构照抄 ``new_models/qwen2_vl/model.py``：``visual`` + ``language_model`` 两个子模块，
顶层 ``WEIGHTS_MAPPER`` 把 ckpt 的 ``visual.*`` / ``model.*`` / ``lm_head.*`` 前缀
分别路由到 ``visual`` 与 ``language_model``，``forward`` 委托给 language_model
（多模态注入在 language_model 内部完成）。
"""

from typing import Any

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_vl.language import Qwen3VLForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLVisionTransformer
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen3VLForConditionalGeneration(RtpModule):

    # Qwen3-VL 的 HF ckpt 把语言模型嵌在 ``model.language_model.``、视觉嵌在
    # ``model.visual.`` 下（不同于 qwen2_vl 的顶层 ``model.`` / ``visual.``）。
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
        self.language_model = Qwen3VLForCausalLM(
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
