from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.glm_v2_weight import GlmV2WeightInfo
from rtp_llm.utils.util import get_config_from_path


class ChatGlmV2(BaseModel):
    @staticmethod
    def get_weight_cls():
        return GlmV2WeightInfo

    @classmethod
    def update_stop_words(cls, config: ModelConfig, config_json: Dict[str, Any]):
        if config.special_tokens is None:
            from rtp_llm.config.model_config import SpecialTokens

            config.special_tokens = SpecialTokens()
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 2)

    @staticmethod
    def get_rotary_embedding_scale(config: ModelConfig, config_json):
        config.attn_config.rope_config.scale = config_json.get("rope_ratio", 1)
        return config

    @staticmethod
    def default_config():
        config = ModelConfig()
        config.attn_config.head_num = 32
        config.attn_config.kv_head_num = 2
        config.attn_config.size_per_head = 128
        config.num_layers = 32
        config.max_seq_len = 8192
        config.vocab_size = 65024
        config.layernorm_eps = 1e-5
        config.inter_size = 13696
        config.add_bias_linear = False
        config.has_post_decoder_layernorm = False
        return config

    @staticmethod
    def modify_config(config: ModelConfig):
        config.use_attention_linear_bias = False
        config.activation_type = "SiGLU"
        config.norm_type = "rmsnorm"
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 2

        return config

    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.chatglm import create_chatglm_v2_config

        config = create_chatglm_v2_config(ckpt_path)
        return config


register_model(
    "chatglm2",
    ChatGlmV2,
    ["ChatGLMModel"],
    ["THUDM/chatglm2-6b", "THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b-32k"],
)
register_model("chat_glm_2", ChatGlmV2)
