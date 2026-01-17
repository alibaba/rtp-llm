import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.internvl_vit import InternVLImageEmbedding
from rtp_llm.models.internvl_weight import InternVLVitWeight, InternVLWeightInfo
from rtp_llm.models.llama import Llama
from rtp_llm.models.multimodal.multimodal_mixin import MultiModalMixin
from rtp_llm.models.qwen_v2 import QWenV2


class InternVL(BaseModel, MultiModalMixin):
    def _init_multimodal(
        self,
        mm_model_config: Any,  # MMModelConfig
        vit_config: VitConfig,
    ):
        # mm_related_params is in model_config, not mm_model_config
        mm_related_params = self.model_config.mm_related_params
        self.mm_part = InternVLImageEmbedding(mm_related_params)
        mm_related_params.vit_weights = InternVLVitWeight(
            {"vision_model": self.mm_part.vision_model, "mlp1": self.mm_part.mlp1}, True
        )
        mm_model_config.mm_sep_tokens = [
            [self.tokenizer.encode("<img>")[0], self.tokenizer.encode("</img>")[0]]
        ]

    @staticmethod
    def get_weight_cls():
        return InternVLWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.internvl import create_internvl_config

        config = create_internvl_config(ckpt_path)
        return config

    @staticmethod
    def _init_vit_params(config: ModelConfig, config_json: Dict[str, Any]):
        config.mm_related_params.config = config_json["vision_config"]
        config.mm_related_params.config["select_layer"] = config_json["select_layer"]
        config.mm_related_params.config["llm_hidden_size"] = config_json["llm_config"][
            "hidden_size"
        ]
        config.mm_related_params.config["downsample_ratio"] = config_json[
            "downsample_ratio"
        ]
        config.mm_related_params.config["ps_version"] = config_json["ps_version"]


register_model("internvl", InternVL, ["InternVLChatModel"])
