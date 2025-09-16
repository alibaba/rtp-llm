import json
import os
import re
from typing import Any, Dict, List, Tuple, Union

import torch
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
    ):
        mm_related_params = self.model_config.mm_related_params

        self.mm_part = LlavaImageEmbedding(self.model_config)
        vit_weight_dict: Dict[str, Any] = {"mm_projector": self.mm_part.mm_projector}
        if mm_related_params.config.get(
            "unfreeze_mm_vision_tower", False
        ) or "mm_vision_tower" in mm_related_params.config.get("mm_tunable_parts", []):
            vit_weight_dict["vision_tower"] = self.mm_part.vision_tower
        if "unpad" in mm_related_params.config.get("mm_patch_merge_type", "flat"):
            vit_weight_dict["image_newline"] = self.mm_part.image_newline
        mm_related_params.vit_weights = BaseVitWeights(vit_weight_dict, True)

    @classmethod
    def _get_mm_module(cls, config: ModelConfig):
        return torch.nn.ModuleList(
            [
                LlavaImageEmbedding(config).vision_tower,
                LlavaImageEmbedding(config).mm_projector,
            ]
        )

    @staticmethod
    def _create_config(ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.activation_type = "SiGLU"
        config.norm_type = "rmsnorm"
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_post_decoder_layernorm = True
        # hugggingface
        config_path = os.path.join(ckpt_path, "config.json")
        param_path = os.path.join(ckpt_path, "params.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                content = content.replace("LlavaForCausalLM", "LLaVAForCausalLM")
                config_json = json.loads(content)
            Llava.from_huggingface(config, config_json)
        else:
            raise Exception("llava parameter from unkown source")
        return config

    @staticmethod
    def get_weight_cls():
        return LlavaWeightInfo

    @staticmethod
    def from_huggingface(config: ModelConfig, config_json: Dict[str, Any]):
        if "text_config" in config_json:
            text_config = config_json["text_config"]
            # if text_config.get("_name_or_path", "") != "":
            #     text_config = AutoConfig.from_pretrained(text_config["_name_or_path"]).to_dict()
            Llama.from_huggingface(config, text_config)

            vision_config = config_json["vision_config"]
            config.mm_related_params.config["vision_config"] = CLIPVisionConfig(
                vision_config
            )

        else:
            Llama.from_huggingface(config, config_json)

            mm_related_params_list = [
                ("mm_use_im_patch_token", False),
                ("mm_use_im_start_end", False),
                ("image_aspect_ratio", None),
                ("tune_mm_mlp_adapter", False),
                ("image_grid_pinpoints", []),
                ("mm_projector_type", "linear"),
                ("mm_patch_merge_type", "flat"),
                ("hidden_size", 0),
                ("mm_vision_select_layer", None),
                ("mm_vision_select_feature", "patch"),
                ("unfreeze_mm_vision_tower", False),
                ("mm_tunable_parts", ""),
                ("add_faster_video", False),
                ("mm_newline_position", "grid"),
                ("mm_spatial_pool_mode", "bilinear"),
            ]

            for param_name, default_value in mm_related_params_list:
                config.mm_related_params.config[param_name] = config_json.get(
                    param_name, default_value
                )

            config.mm_related_params.config["mm_hidden_size"] = config_json.get(
                "mm_hidden_size", config_json["hidden_size"]
            )
            config.mm_related_params.special_token_ids.update(
                {"ignore_token_index": -100, "image_token_index": -200}
            )
            config.mm_related_params.special_tokens.update(
                {
                    "default_mm_token": "<image>",
                    "default_im_start_token": "<im_start>",
                    "default_im_end_token": "<im_end>",
                }
            )

            vis_tower_name = config_json.get(
                "mm_vision_tower", config_json.get("vision_tower", None)
            )
            img_expand_match = re.search("patch(\d+)-(\d+)", vis_tower_name)
            if img_expand_match:
                patch_size = int(img_expand_match.group(1))
                img_size = int(img_expand_match.group(2))
                config.mm_related_params.config["patch_size"] = patch_size
                config.mm_related_params.config["image_size"] = img_size
            config.mm_related_params.config["vit_tower_path"] = vis_tower_name
            config.mm_model_config.mm_sep_tokens = [[-200]]  # image_token_index


register_model("llava", Llava, ["LlavaLlamaForCausalLM"])
