import json
import os
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model
from maga_transformer.models.qwen_v2 import QWenV2

class CosyVoiceQwen(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = QWenV2._create_config(ckpt_path)
        CosyVoiceQwen._update_config(config, ckpt_path)
        config.mm_sep_tokens = [[-200]] # TODO(yinzhi): for SFT support
        return config

    @classmethod
    def _update_config(cls, config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")

        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)

        # input vocab size = speech vocab_size + (LLM vocab size + 2)
        config.input_vocab_size = config_json.get("input_vocab_size", config.vocab_size + 151938)

register_model('cosyvoice_qwen', CosyVoiceQwen, ["CosyQwen2ForCausalLM"])

