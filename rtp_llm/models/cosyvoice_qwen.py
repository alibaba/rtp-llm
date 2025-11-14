import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2


class CosyVoiceQwen(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = QWenV2._create_config(ckpt_path)
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                config_json = json.loads(content)
            config.input_vocab_size = config_json.get(
            "input_vocab_size", config.vocab_size + 151938)
        config.mm_model_config.mm_sep_tokens = [[-200]]  # TODO(yinzhi): for SFT support
        return config

register_model("cosyvoice_qwen", CosyVoiceQwen, ["CosyQwen2ForCausalLM"])
