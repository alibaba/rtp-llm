import os
import logging
import json
import math

from typing import Any, Dict, List

from transformers.models.llama.tokenization_llama import LlamaTokenizer as LlamaTokenizerOrigin
from maga_transformer.distribute.worker_info import ParallelInfo, g_parallel_info
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.llama_weight import LlamaWeightInfo, GemmaWeightInfo
from maga_transformer.models.base_model import BaseModel
from maga_transformer.model_factory_register import register_model

def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

class LlamaTokenizer(LlamaTokenizerOrigin):
    def convert_tokens_to_string(self, tokens: List[int]):
        if len(tokens) == 0:
            return ""
        return super().convert_tokens_to_string(tokens)

class Llama(BaseModel):
    @staticmethod
    def get_mscale(scale: float):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    @staticmethod
    def get_weight_cls():
        return LlamaWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            size_per_head=0,
            layer_num=0,
            max_seq_len=0,
            vocab_size=0,
            ckpt_path=ckpt_path,
            activation_type='SiGLU',
            norm_type='rmsnorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            has_post_decoder_layernorm=True,
        )
        # hugggingface
        config_path = os.path.join(ckpt_path, 'config.json')
        # llama-int8
        param_path = os.path.join(ckpt_path, 'params.json')
        if os.path.exists(config_path):
            with open(config_path) as reader:
                content = reader.read()
                content = content.replace("LlamaForCausalLM", "LLaMAForCausalLM")
                config_json = json.loads(content)
            Llama.from_huggingface(config, config_json)
        elif os.path.exists(param_path):
            logging.info("llama not find config.json, use default config")
            with open(param_path) as reader:
                param_json = json.loads(reader.read())
                config_json = param_json
            Llama.from_params(config, param_json)
        else:
            raise Exception("llama parameter from unkown source")

        return config

    @staticmethod
    def from_huggingface(config, config_json: Dict[str, Any]):
        config.head_num = config_json['num_attention_heads']
        config.head_num_kv = config_json.get('num_key_value_heads', config.head_num)
        config.hidden_size = config_json['hidden_size']
        config.size_per_head = config_json['hidden_size'] // config_json['num_attention_heads']
        config.size_per_head = config_json.get('head_dim', config.size_per_head)
        config.layer_num = config_json['num_hidden_layers']
        config.max_seq_len = config_json.get('max_sequence_length', 2048)
        config.vocab_size = config_json['vocab_size']
        config.layernorm_eps = config_json.get('rms_norm_eps', config_json.get('layer_norm_eps', 1e-05))
        config.inter_size = config_json['intermediate_size']
        config.rotary_embedding_base = config_json.get('rope_theta', 10000)
        config.rotary_embedding_dim = config.size_per_head
        config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)
        rope_scaling = config_json.get('rope_scaling')
        if rope_scaling is not None:
            rope_type = rope_scaling.get('type', rope_scaling.get('rope_type'))
            if rope_type == 'linear':
                config.rotary_embedding_scale = rope_scaling['factor']
                config.org_embedding_max_pos = config_json.get('max_position_embeddings', 2048)
            elif rope_type == 'dynamic':
                config.rotary_embedding_style = 3
            elif rope_type == 'yarn':
                config.rotary_embedding_style = 5
                config.rotary_embedding_scale = rope_scaling['factor']
                config.rotary_factor1 = rope_scaling.get('beta_slow', 1)
                config.rotary_factor2 = rope_scaling.get('beta_fast', 32)
                config.org_embedding_max_pos = rope_scaling['original_max_position_embeddings']
                config.rotary_embedding_mscale = Llama.get_mscale(config.rotary_embedding_scale)
            elif rope_type == 'llama3':
                config.rotary_embedding_style = 6
                config.rotary_embedding_scale = rope_scaling['factor']
                config.rotary_factor1 = rope_scaling['low_freq_factor']
                config.rotary_factor2 = rope_scaling['high_freq_factor']
                config.org_embedding_max_pos = rope_scaling['original_max_position_embeddings']
            else:
                raise Exception(f"unsupport rope_scaling {rope_scaling}")
        # config.activation_type = config_json.get("hidden_act", config.activation_type)
        config.special_tokens.bos_token_id = config_json['bos_token_id']
        eos_token_id = config_json['eos_token_id']
        # openai endpoint will get corrent eos token id list from tokenizer
        if isinstance(eos_token_id, list):
            config.special_tokens.eos_token_id = eos_token_id[0]
        else:
            config.special_tokens.eos_token_id = eos_token_id
        config.use_logn_attn = config_json.get("use_logn_attn", False)

    @staticmethod
    def from_params(config: GptInitModelParameters, params_json: Dict[str, Any]):
        config.head_num = params_json['n_heads']
        config.head_num_kv = params_json.get('n_kv_heads', config.head_num)
        config.size_per_head = params_json['dim'] // params_json['n_heads']
        config.layer_num = params_json['n_layers']
        config.max_seq_len = 2048
        config.vocab_size = 32000
        config.layernorm_eps = params_json['norm_eps']
        config.inter_size = compute_intermediate_size(
            params_json['dim'],
            params_json.get("ffn_dim_multiplier", 1),
            params_json['multiple_of'])
        config.special_tokens.bos_token_id = 1
        config.special_tokens.eos_token_id = 2
        config.rotary_embedding_dim = config.size_per_head
        config.tie_word_embeddings = params_json.get('tie_word_embeddings', False)
        return config

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        tokenizer_config_file = os.path.join(config.tokenizer_path, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_file):
            logging.info("load super tokenzier")
            return super().get_tokenizer(config)
        else:
            logging.info("load LlamaTokenizer")
            return LlamaTokenizer.from_pretrained(config.tokenizer_path)

class Baichuan(Llama):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Llama._create_config(ckpt_path)
        if config.layer_num == 40: # 13B
            config.rotary_embedding_style = 0
            config.rotary_embedding_dim = 0
            config.use_attention_linear_bias = True
        config.special_tokens.bos_token_id = -1
        config.special_tokens.user.token_ids = [195]
        config.special_tokens.user.eos_token_ids = []
        config.special_tokens.assistant.token_ids = [196]
        config.special_tokens.assistant.eos_token_ids = [config.special_tokens.eos_token_id]
        return config

class Baichuan2(Baichuan):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Baichuan._create_config(ckpt_path)
        config.normalize_lm_head_weight = True
        return config

class Gemma(Llama):
    def __init__(self, config: GptInitModelParameters):
        if os.environ.get("ENABLE_OPENSOURCE_FMHA", None) != "OFF":
            logging.warn("opensource fmha does not support head dim 256, thus disabled for gemma model")
            os.environ["ENABLE_OPENSOURCE_FMHA"] = "OFF"
        super().__init__(config)

    @staticmethod
    def get_weight_cls():
        return GemmaWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Llama._create_config(ckpt_path)
        config.has_post_decoder_layernorm = True
        config.input_embedding_scalar = (config.hidden_size ** 0.5)
        config.rotary_embedding_dim = config.size_per_head
        config.activation_type = 'gated-gelu'
        return config

class Cohere(Llama):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Llama._create_config(ckpt_path)
        config.rotary_embedding_style = 0
        config.norm_type = 'layernorm'
        config.qk_norm = True
        return config

register_model('internlm', Llama, ["InternLMForCausalLM"])
register_model('internlm2', Llama, ["InternLM2ForCausalLM"])
register_model('llama', Llama, ["LlamaForCausalLM", "YiForCausalLM"])
register_model('xverse', Llama, ["XverseForCausalLM"])
register_model('aquila', Llama, ["AquilaModel"])
register_model('mistral', Llama, ["MistralForCausalLM"])
register_model('baichuan', Baichuan, ["BaichuanForCausalLM"])
register_model('baichuan2', Baichuan2)
register_model('gemma', Gemma, ["GemmaForCausalLM"])
register_model('cohere', Cohere, ["CohereForCausalLM"])
