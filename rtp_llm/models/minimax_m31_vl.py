"""MiniMax-M3.1 vision-language configuration on the M3.1 text backbone."""

import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.minimax_m31 import MiniMaxM31


def configure_m31_vl(config, checkpoint: dict, ckpt_path: str) -> None:
    if not checkpoint.get("vision_config"):
        raise ValueError("MiniMax-M3.1 VL requires vision_config")
    config.mm_model_config.is_multimodal = True
    image_token = int(checkpoint.get("image_token_index", 200025))
    video_token = int(checkpoint.get("video_token_index", 200026))
    config.mm_model_config.mm_sep_tokens = [[image_token], [video_token]]
    config.mm_model_config.mm_position_ids_style = 0
    mm = config.mm_related_params.config
    mm["ckpt_path"] = ckpt_path
    mm["vision_config"] = checkpoint["vision_config"]
    mm["image_token_index"] = image_token
    mm["video_token_index"] = video_token
    mm["projector_hidden_size"] = checkpoint.get("projector_hidden_size")
    mm["multimodal_projector_bias"] = checkpoint.get("multimodal_projector_bias", True)
    mm["img_token_compression_config"] = checkpoint.get(
        "img_token_compression_config", {}
    )
    mm["projector_hidden_act"] = checkpoint.get("projector_hidden_act", "gelu")
    mm["vision_feature_layer"] = checkpoint.get("vision_feature_layer", -1)
    mm["vision_feature_select_strategy"] = checkpoint.get(
        "vision_feature_select_strategy", "full"
    )
    config.mm_related_params.special_tokens["default_mm_token"] = "]<]image[>["


class MiniMaxM31VL(MiniMaxM31):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        with open(os.path.join(ckpt_path, "config.json")) as reader:
            checkpoint = json.load(reader)
        configure_m31_vl(config, checkpoint, ckpt_path)
        config.model_type = "minimax_m31_vl"
        return config


register_model("minimax_m31_vl", MiniMaxM31VL)
