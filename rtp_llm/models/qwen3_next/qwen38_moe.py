"""Native text-only Qwen3.8 metadata support; no RL parallel guard is relaxed."""
import os

from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3NextBase
from rtp_llm.models.qwen3_next.qwen3_next_weight import Qwen35MoeWeight
from rtp_llm.models.qwen3_next.qwen38_config import validate_flat_qwen38_config


class Qwen38MoeText(Qwen3NextBase):
    def create_model_loader(self):
        if os.environ.get("RTP_LLM_RL_METADATA_BOOTSTRAP", "0") == "1":
            from rtp_llm.model_loader.metadata_bootstrap import create_metadata_bootstrap_loader
            return create_metadata_bootstrap_loader(self)
        return super().create_model_loader()

    @staticmethod
    def get_weight_cls():
        return Qwen35MoeWeight

    @classmethod
    def _preprocess_config_json(cls, config_json):
        return validate_flat_qwen38_config(config_json)

    @classmethod
    def _create_config(cls, ckpt_path):
        config = super()._create_config(ckpt_path)
        config.enable_fp32_lm_head = False
        config.mm_model_config.is_multimodal = False
        config.mm_model_config.mm_sep_tokens = []
        return config

    @classmethod
    def _parse_rope_config(cls, config_json, config):
        rope = config_json["rope_parameters"]
        config.attn_config.rope_config.style = 1
        config.attn_config.rope_config.base = rope["rope_theta"]
        config.partial_rotary_factor = rope["partial_rotary_factor"]
        config.attn_config.rope_config.dim = int(config.attn_config.size_per_head * config.partial_rotary_factor)


# The conditional-generation architecture remains registered to Qwen35Moe.
register_model("qwen38_moe_text", Qwen38MoeText, ["Qwen3_5MoeForCausalLM"])
