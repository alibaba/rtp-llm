import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2


class CosyVoiceQwen(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.cosyvoice_qwen import (
            create_cosyvoice_qwen_config,
        )

        config = create_cosyvoice_qwen_config(ckpt_path)
        return config


register_model("cosyvoice_qwen", CosyVoiceQwen, ["CosyQwen2ForCausalLM"])
