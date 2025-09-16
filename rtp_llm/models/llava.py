import json
import os
import re
from typing import Any, Dict

from transformers import CLIPVisionConfig

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.llama import Llama
from rtp_llm.models.llama_weight import LlamaWeightInfo
from rtp_llm.utils.model_weight import W


class LlavaWeightInfo(LlamaWeightInfo):
    def _get_weight_info(self):
        llava_weight = super()._get_weight_info()

        # for llava-next
        for weight in llava_weight.layer_weights:
            if weight.name == W.attn_o_b:
                llava_weight.layer_weights.remove(weight)
                break

        return llava_weight


class Llava(Llama):
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
        config.mm_model_config.is_multimodal = True
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
