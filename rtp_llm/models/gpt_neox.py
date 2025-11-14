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
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = GPTNeox.from_huggingface(config_dict)
            config.ckpt_path = ckpt_path
        else:
            config = ModelConfig(
                head_num=40,
                head_num_kv=40,
                size_per_head=128,
                num_layers=40,
                max_seq_len=4096,
                vocab_size=250752,
                inter_size=20480,
                # inter_padding_size removed, now using inter_size directly,
            )
            config.special_tokens.eos_token_id = 2
            config.attn_config.rope_config.dim = 128
            config.attn_config.rope_config.style = 1
            config.has_pre_decoder_layernorm = False
            config.has_post_decoder_layernorm = True
            config.norm_type = "layernorm"
            config.use_norm_input_residual = True
        return config

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]) -> ModelConfig:
        config = ModelConfig(
            head_num=40,
            size_per_head=128,
            layer_num=40,
            max_seq_len=4096,
            vocab_size=250752,
        )
        config.attn_config.head_num = config_json["num_attention_heads"]
        config.attn_config.kv_head_num = config.head_num
        config.attn_config.size_per_head = (
            config_json["hidden_size"] // config_json["num_attention_heads"]
        )
        config.num_layers = config_json["num_hidden_layers"]
        config.vocab_size = config_json["vocab_size"]
        config.layernorm_eps = config_json["layer_norm_eps"]
        config.inter_size = config_json["intermediate_size"]
        # inter_padding_size removed, inter_size used directly
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        config.attn_config.rope_config.dim = int(
            config.size_per_head * config_json.get("rotary_pct", 1.0)
        )
        config.attn_config.rope_config.style = 1
        if config_json.get("rope_scaling", None):
            config.attn_config.rope_config.style = 3
            config.attn_config.rope_config.scale = config_json["rope_scaling"]["factor"]
            config.org_embedding_max_pos = config_json.get(
                "max_position_embeddings", 2048
            )

        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "layernorm"
        config.use_norm_input_residual = True
        config.tie_word_embeddings = config_json.get("tie_word_embeddings", False)
        config.config_dtype = config_json.get("torch_dtype", None)

        return config


class GPTNeox13B(GPTNeox):
    @staticmethod
    def get_weight_cls():
        return GPTNeox13BWeight

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = GPTNeox13B.from_huggingface(config_dict)
        else:
            config = ModelConfig(
                head_num=40,
                head_num_kv=40,
                size_per_head=128,
                num_layers=40,
                max_seq_len=4096,
                vocab_size=250752,
                inter_size=20480,
                # inter_padding_size removed, now using inter_size directly,
            )
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.norm_type = "rmsnorm"
        config.special_tokens.eos_token_id = 2
        return config

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]) -> ModelConfig:
        config = ModelConfig(
            head_num=config_json["num_attention_heads"],
            size_per_head=config_json["hidden_size"] // config_json["num_attention_heads"],
            num_layers=config_json["num_hidden_layers"],
            max_seq_len=4096,
            vocab_size=config_json["vocab_size"],
        )
        config.attn_config.kv_head_num = config.head_num
        config.layernorm_eps = config_json["layer_norm_eps"]
        config.inter_size = config_json["intermediate_size"]
        # inter_padding_size removed, inter_size used directly
        config.special_tokens.bos_token_id = config_json.get("bos_token_id", -1)
        config.special_tokens.eos_token_id = config_json.get("eos_token_id", 0)
        if config_json.get("rope_scaling", None):
            if config_json["rope_scaling"]["type"] == "dynamic":
                config.attn_config.rope_config.style = 3
                config.attn_config.rope_config.scale = config_json["rope_scaling"]["factor"]
                config.org_embedding_max_pos = config_json.get(
                    "max_position_embeddings", 2048
                )
        return config


register_model("gpt_neox", GPTNeox, ["GPTNeoXForCausalLM"])
register_model("gpt_neox_13b", GPTNeox13B)
