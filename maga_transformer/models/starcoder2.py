from typing import Any, Dict, List
from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, transpose
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.models.base_model import BaseModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from maga_transformer.model_factory_register import register_model
import torch
import functools


def merge_qkv_b(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_b = torch.concat([q, k, v], dim=0).contiguous()
    return qkv_b


def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight


class Starcoder2WeightInfo(ModelDeployWeightInfo):

    def _get_weight_info(self):
        weights = [
            AtomicWeight(W.embedding,
                       [CkptWeightInfo('model.embed_tokens.weight', identity)],
                       identity),
            # AtomicWeight(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            AtomicWeight(W.final_ln_gamma,
                       [CkptWeightInfo('model.norm.weight', identity)],
                       identity),
            AtomicWeight(W.final_ln_beta,
                       [CkptWeightInfo('model.norm.bias', identity)],
                       identity),
        ]
        
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(W.pre_ln_beta, [
                    CkptWeightInfo('model.layers.{i}.input_layernorm.bias',
                                identity)
                ], identity),
                AtomicWeight(W.pre_ln_gamma, [
                    CkptWeightInfo('model.layers.{i}.input_layernorm.weight',
                                identity)
                ], identity),
                AttnAtomicWeight(W.attn_qkv_w, [
                    CkptWeightInfo("model.layers.{i}.self_attn.q_proj.weight",
                                identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.k_proj.weight",
                                identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.v_proj.weight",
                                identity)
                ], functools.partial(merge_qkv_hf), config=attn_config),
                AttnAtomicWeight(W.attn_qkv_b, [
                    CkptWeightInfo("model.layers.{i}.self_attn.q_proj.bias",
                                identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.k_proj.bias",
                                identity),
                    CkptWeightInfo("model.layers.{i}.self_attn.v_proj.bias",
                                identity)
                ], functools.partial(merge_qkv_b), config=attn_config),
                AttnAtomicWeight(W.attn_o_w, [
                    CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight',
                                identity)
                ], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_o_b, [
                    CkptWeightInfo('model.layers.{i}.self_attn.o_proj.bias',
                                identity)
                ], identity, config=attn_config),
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [CkptWeightInfo('model.layers.{i}.mlp.c_fc.weight', identity)],
                        transpose, config=ffn_config),
                    FfnAtomicWeight(
                        W.ffn_b3,
                        [CkptWeightInfo('model.layers.{i}.mlp.c_fc.bias', identity)],
                        identity, config=ffn_config),
                    FfnAtomicWeight(W.ffn_w2, [
                        CkptWeightInfo('model.layers.{i}.mlp.c_proj.weight', identity)
                    ], transpose, config=ffn_config),
                    FfnAtomicWeight(
                        W.ffn_b2,
                        [CkptWeightInfo('model.layers.{i}.mlp.c_proj.bias', identity)],
                        identity, config=ffn_config)], 
                          config=ffn_config),
                AtomicWeight(W.post_ln_beta, [
                    CkptWeightInfo(
                        'model.layers.{i}.post_attention_layernorm.bias', identity)
                ], identity),
                AtomicWeight(W.post_ln_gamma, [
                    CkptWeightInfo(
                        'model.layers.{i}.post_attention_layernorm.weight',
                        identity)
                ], identity),
            ]
            layer_weights.append(layer_weight)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


StarcoderTokenizer = GPT2TokenizerFast


class StarCoder2(BaseModel):

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters):
        return StarcoderTokenizer.from_pretrained(config.tokenizer_path)

    @staticmethod
    def get_weight_cls():
        return Starcoder2WeightInfo

    @staticmethod
    def from_huggingface(config_json: Dict[str, Any]):
        model_type = config_json['model_type']
        config = GptInitModelParameters(
            head_num=config_json['num_attention_heads'],
            head_num_kv=config_json['num_key_value_heads'],
            size_per_head=config_json['hidden_size'] // config_json['num_attention_heads'],
            layer_num=config_json['num_hidden_layers'],
            max_seq_len=config_json.get('max_position_embeddings', 8192),
            vocab_size=config_json['vocab_size'],
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
        )
        if model_type != 'starcoder2':
            raise BaseException(f'model type is not starcoder: {model_type}')
        config.layernorm_eps = config_json['layer_norm_epsilon']
        config.inter_size = config_json['intermediate_size']
        config.special_tokens.eos_token_id = config_json['eos_token_id']
        config.special_tokens.bos_token_id = config_json['bos_token_id']
        config.activation_type = config_json['activation_function']
        config.has_post_decoder_layernorm = True
        config.rotary_embedding_base = config_json.get('rope_theta', 1000000)
        config.rotary_embedding_dim = config.size_per_head
        config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)
        return config

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        if config_dict:
            config = StarCoder2.from_huggingface(config_dict)
        else:
            config = GptInitModelParameters(head_num=36,
                                            head_num_kv=4,
                                            size_per_head=128,
                                            inter_size=4 * 4608,
                                            layer_num=32,
                                            max_seq_len=16384,
                                            vocab_size=49152,
                                            bos_token_id=0,
                                            eos_token_id=0,
                                            rotary_embedding_dim=128,
                                            rotary_embedding_style=1,
                                            has_post_decoder_layernorm=True)
        return config


register_model('starcoder2', StarCoder2, ["Starcoder2ForCausalLM"])
