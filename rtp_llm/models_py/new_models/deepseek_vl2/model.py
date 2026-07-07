"""DeepSeek VL V2 top-level container for the new loader.

结构照抄 ``new_models/qwen3_vl/model.py``：``visual`` + ``language_model`` 两个子模块，
顶层 ``WEIGHTS_MAPPER`` 把 ckpt 的 ``vision.*`` / ``projector.*`` / ``image_newline``
等路由到 ``visual``，``language.model.*`` / ``language.lm_head.*`` 路由到
``language_model``，``forward`` 委托给 language_model（多模态注入在 language_model
内部完成）。

DeepSeek VL V2 的 HF ckpt 权重前缀:
  - ``language.model.*``  → 语言模型权重（MLA + MoE）
  - ``language.lm_head.*`` → lm_head
  - ``vision.*``           → SigLIP 视觉编码器 (timm)
  - ``projector.*``         → MlpProjector
  - ``image_newline``       → 2D tile 格式参数
  - ``view_seperator``      → 2D tile 格式参数
  - ``tile_indicators``     → 1D tile 格式参数

顶层 WEIGHTS_MAPPER 的映射逻辑:
  - ``language.model.`` → ``language_model.``: 语言模型收到不带 ``model.``
    前缀的权重名（如 ``layers.0...``），其自身的 WEIGHTS_MAPPER
    (``{"model.": ""}``) 对这些名字是 no-op
  - ``language.lm_head.`` → ``language_model.lm_head.``: lm_head 权重
  - ``vision.`` → ``visual.vision.``: SigLIP 权重
  - ``projector.`` → ``visual.projector.``: 投影器权重
  - ``image_newline`` / ``view_seperator`` / ``tile_indicators``:
    精确映射到 ``visual.*`` 以便按前缀分组后路由到 visual 子模块
"""

from typing import Any

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_vl2.language import DeepSeekVLV2ForCausalLM
from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
    DeepSeekVLV2VisionTransformer,
)
from rtp_llm.models_py.weight_mapper import WeightsMapper


class DeepSeekVLV2ForConditionalGeneration(RtpModule):
    """Top-level container for DeepSeek VL V2 multimodal model.

    Holds ``visual`` (vision tower) and ``language_model`` (DeepSeek V3.2
    backbone with multimodal injection) sub-modules. Weight loading routes
    ckpt keys to the appropriate child via prefix mapping + grouping.
    """

    # DeepSeek VL V2 ckpt weight prefix routing:
    # - language.model.* → language_model.* (top-level strips "language." prefix;
    #   the language model receives names like "layers.0..." / "lm_head.weight"
    #   without a "model." prefix, so its WEIGHTS_MAPPER {"model.": ""} is a no-op.)
    # - language.lm_head.* → language_model.lm_head.*
    # - vision.* → visual.vision.* (timm SigLIP)
    # - projector.* → visual.projector.* (MlpProjector)
    # - image_newline / view_seperator / tile_indicators → exact match to visual.*
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "language.model.": "language_model.",
            "language.lm_head.": "language_model.lm_head.",
            "vision.": "visual.vision.",
            "projector.": "visual.projector.",
        },
        exact_mapping={
            "image_newline": "visual.image_newline",
            "view_seperator": "visual.view_seperator",
            "tile_indicators": "visual.tile_indicators",
        },
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config

        self.visual = DeepSeekVLV2VisionTransformer(
            model_config=model_config, load_config=load_config
        )
        self.language_model = DeepSeekVLV2ForCausalLM(
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


__all__ = ["DeepSeekVLV2ForConditionalGeneration"]
