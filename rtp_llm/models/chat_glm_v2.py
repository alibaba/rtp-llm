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
    def from_huggingface(cls, config_json: Dict[str, Any]):
        """
        "apply_query_key_layer_scaling": true,
        "apply_residual_connection_post_layernorm": false,
        "attention_softmax_in_fp32": true,
        "fp32_residual_connection": false,
        "original_rope": true,
        """
        config = ModelConfig()
        config.attn_config.head_num = 32
        config.attn_config.size_per_head = 128
        config.num_layers = 32
        config.max_seq_len = 8192
        config.vocab_size = 65024
        config.attn_config.head_num = config_json["num_attention_heads"]
        if config_json.get("multi_query_attention", False):
            config.attn_config.kv_head_num = config_json["multi_query_group_num"]
        else:
            config.attn_config.kv_head_num = config.attn_config.head_num
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.num_layers = config_json["num_layers"]
        config.max_seq_len = config_json.get("seq_length", 8192)
        config.vocab_size = config_json["padded_vocab_size"]
        config.layernorm_eps = config_json["layernorm_epsilon"]
        config.inter_size = config_json["ffn_hidden_size"]
        config.add_bias_linear = config_json["add_bias_linear"]
        config.has_post_decoder_layernorm = config_json["post_layer_norm"]
        if "pre_seq_len" in config_json:
            config.pre_seq_len = config_json["pre_seq_len"]
        if "prefix_projection" in config_json:
            config.prefix_projection = config_json["prefix_projection"]
        config.src_quantization_bit = config_json.get("quantization_bit", 0)
        config.attn_config.rope_config.dim = config.attn_config.size_per_head
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.special_tokens.pad_token_id = config_json.get("pad_token_id", 0)
        config = cls.get_rotary_embedding_scale(config, config_json)
        cls.update_stop_words(config, config_json)
        config.config_dtype = config_json.get("torch_dtype", None)
        return config

    @classmethod
    def update_stop_words(
        cls, config: ModelConfig, config_json: Dict[str, Any]
    ):
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
        config_dict = get_config_from_path(ckpt_path)
        if config_dict is not None:
            config = ChatGlmV2.from_huggingface(config_dict)
        else:
            config = ChatGlmV2.default_config()
        config = ChatGlmV2.modify_config(config)
        return config


register_model(
    "chatglm2",
    ChatGlmV2,
    ["ChatGLMModel"],
    ["THUDM/chatglm2-6b", "THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b-32k"],
)
register_model("chat_glm_2", ChatGlmV2)
