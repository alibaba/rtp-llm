"""Configuration creators for BERT and RoBERTa models."""

import logging
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_config_creators.base import require_config_json
from rtp_llm.model_config_creators.registry import register_config_creator

logger = logging.getLogger(__name__)


def _from_huggingface_bert(config: ModelConfig, config_json: Dict[str, Any]) -> None:
    """Apply BERT-specific configuration from HuggingFace config.json.

    Args:
        config: ModelConfig instance to populate
        config_json: Parsed config.json dictionary
    """
    # Check position_embedding_type == absolute
    config.attn_config.head_num = config_json["num_attention_heads"]
    # BERT has no group attention
    config.attn_config.kv_head_num = config.attn_config.head_num
    config.attn_config.size_per_head = (
        config_json["hidden_size"] // config_json["num_attention_heads"]
    )
    config.hidden_size = config_json["hidden_size"]
    config.num_layers = config_json["num_hidden_layers"]
    config.max_seq_len = config_json.get("max_position_embeddings", 512)
    config.vocab_size = config_json["vocab_size"]
    config.type_vocab_size = config_json.get("type_vocab_size", 0)
    config.layernorm_eps = config_json["layer_norm_eps"]
    config.inter_size = config_json["intermediate_size"]
    config.config_dtype = config_json.get("torch_dtype", None)


def create_bert_config(ckpt_path: str) -> ModelConfig:
    """Create BERT model configuration from checkpoint path.

    This function extracts the configuration creation logic from Bert._create_config,
    allowing configuration creation without instantiating the model class.

    Args:
        ckpt_path: Path to the BERT model checkpoint directory

    Returns:
        ModelConfig instance with BERT-specific settings

    Raises:
        FileNotFoundError: If config.json doesn't exist
        json.JSONDecodeError: If config.json is invalid JSON
    """
    config = ModelConfig()
    config.ckpt_path = ckpt_path
    config.activation_type = "gelu"
    config.norm_type = "layernorm"
    config.attn_config.rope_config.dim = 0
    config.attn_config.rope_config.style = 0
    config.has_positional_encoding = True
    config.has_pre_decoder_layernorm = True
    config.layernorm_type = "post_layernorm"
    config.attn_config.is_causal = False

    # Read and apply config.json
    config_json = require_config_json(ckpt_path)
    _from_huggingface_bert(config, config_json)

    return config


def create_roberta_config(ckpt_path: str) -> ModelConfig:
    # Start with BERT config
    config = create_bert_config(ckpt_path)

    # Apply RoBERTa-specific overrides
    config_json = require_config_json(ckpt_path)
    config.special_tokens.pad_token_id = config_json["pad_token_id"]
    config.position_ids_style = 1

    return config


def create_megatron_bert_config(ckpt_path: str) -> ModelConfig:
    config = create_bert_config(ckpt_path)
    config.has_pre_decoder_layernorm = False
    config.layernorm_type = "pre_layernorm"
    return config


def create_jina_bert_config(ckpt_path: str) -> ModelConfig:
    config = create_bert_config(ckpt_path)
    config.activation_type = "gated-gelu"
    config.use_attention_linear_bias = True
    config.has_positional_encoding = False
    config.qk_norm = True
    return config


# Register the configuration creators
register_config_creator("bert", create_bert_config)
register_config_creator("roberta", create_roberta_config)
register_config_creator("megatron_bert", create_megatron_bert_config)
register_config_creator("jina_bert_code", create_jina_bert_config)
register_config_creator("jina_bert", create_jina_bert_config)
