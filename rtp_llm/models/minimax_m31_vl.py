"""MiniMax-M3.1 vision-language target model registration."""

import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.minimax_m3_vl import _apply_minimax_m3_vl_config
from rtp_llm.models.minimax_m31 import MiniMaxM31, MiniMaxM31Weight


class MiniMaxM31_VL(MiniMaxM31):
    """M3.1 VL container with M3.1-owned weight and runtime classes."""

    @classmethod
    def _create_config(cls, ckpt_path):
        config = super()._create_config(ckpt_path)
        config.model_type = "minimax_m31_vl"
        return config

    @classmethod
    def _from_hf(cls, config, ckpt_path):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            config_json = json.load(reader)
        _apply_minimax_m3_vl_config(config, config_json, ckpt_path)
        cls._parse_nvfp4_mock_config(config, config_json)
        return config

    @staticmethod
    def get_weight_cls():
        return MiniMaxM31Weight


register_model("minimax_m31_vl", MiniMaxM31_VL)


__all__ = ["MiniMaxM31_VL"]
