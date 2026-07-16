import json
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.deepseek_v2 import DeepSeekV2


class Glm4MoeLite(DeepSeekV2):
    """GLM-4.7-Flash (model_type=glm4_moe_lite, arch=Glm4MoeLiteForCausalLM)."""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.scoring_func = 1
        if config.config_dtype is None:
            config_path = os.path.join(ckpt_path, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"config.json not found at {config_path}, "
                    f"cannot determine model dtype"
                )
            with open(config_path) as f:
                dtype = json.load(f).get("dtype")
            if dtype is None:
                raise ValueError(
                    f"Neither 'torch_dtype' nor 'dtype' found in {config_path}; "
                    f"set --act_type explicitly or add a dtype field to config.json"
                )
            config.config_dtype = dtype
        return config


register_model("glm4_moe_lite", Glm4MoeLite, ["Glm4MoeLiteForCausalLM"])
