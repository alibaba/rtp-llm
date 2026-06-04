from typing import Any

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.util import get_config_from_path


class Qwen25OmniThinkerWeight(QWenV2Weight):
    def __init__(self, **kwargs: Any):
        super().__init__(prefix="thinker.", **kwargs)


class Qwen25OmniThinker(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path
        config.max_seq_len = 32768
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.special_tokens.bos_token_id = 151644
        config.special_tokens.eos_token_id = 151645
        config.special_tokens.stop_words_id_list = [[151645], [151644]]

        config_json = get_config_from_path(ckpt_path)
        if config_json and "thinker_config" in config_json:
            text_config = config_json["thinker_config"].get("text_config", {})
            QWenV2._from_config_json(config, text_config)
        else:
            cls._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen25OmniThinkerWeight


register_model("qwen2_5_omni_thinker", Qwen25OmniThinker)
