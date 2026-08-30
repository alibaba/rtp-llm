from rtp_llm.model_factory_register import register_model
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.glm4_moe import resolve_config_dtype


class Glm4MoeLite(DeepSeekV2):
    """GLM-4.7-Flash (model_type=glm4_moe_lite, arch=Glm4MoeLiteForCausalLM)."""

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.scoring_func = 1
        # Shared with Glm4Moe: the checkpoints of this family ship the new
        # `dtype` key rather than `torch_dtype`.
        resolve_config_dtype(config, ckpt_path)
        return config


register_model("glm4_moe_lite", Glm4MoeLite, ["Glm4MoeLiteForCausalLM"])
