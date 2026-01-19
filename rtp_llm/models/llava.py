import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

from transformers import CLIPVisionConfig

from rtp_llm.config.model_config import ModelConfig, VitParameters
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.llama import Llama
from rtp_llm.models.llava_vit import LlavaImageEmbedding
from rtp_llm.models.llava_weight import LlavaWeightInfo
from rtp_llm.models.multimodal.multimodal_mixin import BaseVitWeights, MultiModalMixin


class Llava(Llama, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = self.model_config.mm_related_params
        self.mm_part = LlavaImageEmbedding(
            mm_related_params, model_config=self.model_config
        )
        vit_weight_dict: Dict[str, Any] = {"mm_projector": self.mm_part.mm_projector}
        if mm_related_params.config.get(
            "unfreeze_mm_vision_tower", False
        ) or "mm_vision_tower" in mm_related_params.config.get("mm_tunable_parts", []):
            vit_weight_dict["vision_tower"] = self.mm_part.vision_tower
        if "unpad" in mm_related_params.config.get("mm_patch_merge_type", "flat"):
            vit_weight_dict["image_newline"] = self.mm_part.image_newline
        mm_related_params.vit_weights = BaseVitWeights(vit_weight_dict, True)

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.llava import create_llava_config

        config = create_llava_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return LlavaWeightInfo


register_model("llava", Llava, ["LlavaLlamaForCausalLM"])
