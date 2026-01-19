from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.gpt_neox_weight import GPTNeox13BWeight, GPTNeoxWeight
from rtp_llm.utils.util import get_config_from_path


class GPTNeox(BaseModel):
    @staticmethod
    def get_weight_cls():
        return GPTNeoxWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.gpt_neox import create_gpt_neox_config

        config = create_gpt_neox_config(ckpt_path)
        return config


class GPTNeox13B(GPTNeox):
    @staticmethod
    def get_weight_cls():
        return GPTNeox13BWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.gpt_neox import create_gpt_neox_13b_config

        config = create_gpt_neox_13b_config(ckpt_path)
        return config


register_model("gpt_neox", GPTNeox, ["GPTNeoXForCausalLM"])
register_model("gpt_neox_13b", GPTNeox13B)
