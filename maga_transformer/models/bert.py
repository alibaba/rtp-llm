import os
import json

from typing import Any, Dict

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.gpt import GPT
from maga_transformer.models.bert_weight import BertWeightInfo
from maga_transformer.model_factory_register import register_model
from transformers import AutoTokenizer

class Bert(GPT):
    @staticmethod
    def get_weight_cls():
        return BertWeightInfo

    @staticmethod
    def _create_config(ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='gelu',
            norm_type='layernorm',
            rotary_embedding_dim=0,
            rotary_embedding_style=0,
            has_positional_encoding=True,
            has_pre_decoder_layernorm=True,
            layernorm_type='post_layernorm',
            has_lm_head=False,
            is_causal=False
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, 'config.json')
        if not os.path.exists(config_path):
            raise Exception(f"failed to find config json from {ckpt_path}")
        with open(config_path) as reader:
            content = reader.read()            
            config_json = json.loads(content)
            Bert.from_huggingface(config, config_json)
        return config

    @staticmethod
    def from_huggingface(config: GptInitModelParameters, config_json: Dict[str, Any]):
        config.head_num = config_json['num_attention_heads']
        # bert has no group attention
        config.head_num_kv = config.head_num
        config.size_per_head = config_json['hidden_size'] // config_json['num_attention_heads']
        config.hidden_size = config_json['hidden_size']
        config.layer_num = config_json['num_hidden_layers']
        config.max_seq_len = config_json.get('max_position_embeddings', 512)
        config.vocab_size = config_json['vocab_size']
        config.type_vocab_size = config_json.get('type_vocab_size', 0)
        config.layernorm_eps = config_json['layer_norm_eps']
        config.inter_size = config_json['intermediate_size']
        # not in use
        config.special_tokens.eos_token_id = 0

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return AutoTokenizer.from_pretrained(config.tokenizer_path, trust_remote_code=True)
    
register_model('bert', Bert)