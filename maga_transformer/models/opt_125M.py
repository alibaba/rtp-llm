import functools
import torch

from maga_transformer.utils.util import get_config_from_path
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, \
    ModelWeightInfo, ModelDeployWeightInfo, CkptWeightInfo, identity, transpose
from maga_transformer.models.gpt import GPT
from maga_transformer.utils.model_weight import (W, WeightInfo, ModelWeightInfo, ModelDeployWeightInfo,
                                                 CkptWeightInfo, identity, transpose,
                                                 merge_qkv_b, merge_qkv_hf)
from maga_transformer.model_factory_register import register_model
from typing import List
# from maga_transformer.utils.group_quant_weight_util import get_layer_group_quant_weight_info



class OptWeightInfo(ModelDeployWeightInfo):
    def extent_context_position_ids(
            self,context_begin_position:int,context_end_position:int,
            token_type_ids:torch.Tensor,token_ids:torch.Tensor
    )-> List[int]:
        return list(map(lambda x:x + 2, range(context_begin_position,context_end_position)))
    
    def extend_generate_position_ids(
            self, generate_batch_size: int, num_beams: int,
            vision_token_length: List[int], seq_lengths_list: List[int]
    )-> List[int]:
        return [i + 1 for i in seq_lengths_list]


    def _get_weight_info(self):
        layer_weights = [
            # * Attention pre_layer_norm
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.weight', identity)], identity),
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.bias', identity)], identity),

            # *  Attention
            WeightInfo(W.attn_qkv_w, [
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.k_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.weight', identity),
                ], functools.partial(merge_qkv_hf)),

            WeightInfo(W.attn_qkv_b,[
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.bias', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.k_proj.bias', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.bias', identity),
                ], functools.partial(merge_qkv_b)),

            WeightInfo(W.attn_o_w,[CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.weight', identity)], transpose),
            WeightInfo(W.attn_o_b,[CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.bias', identity)], identity),

            # * FFN pre_layer_norm
            WeightInfo(W.post_ln_gamma,[CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.weight', identity)], identity),
            WeightInfo(W.post_ln_beta,[CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.bias', identity)], identity),

            # * FFN
            WeightInfo(W.ffn_w3,[CkptWeightInfo('model.decoder.layers.{i}.fc1.weight', identity)], transpose),
            WeightInfo(W.ffn_b3,[CkptWeightInfo('model.decoder.layers.{i}.fc1.bias', identity)], identity),
            WeightInfo(W.ffn_w2,[CkptWeightInfo('model.decoder.layers.{i}.fc2.weight', identity)], transpose),
            WeightInfo(W.ffn_b2,[CkptWeightInfo('model.decoder.layers.{i}.fc2.bias', identity)], identity),
        ]

        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.decoder.embed_tokens.weight', identity)], identity),
            WeightInfo(W.positional_embedding,[CkptWeightInfo('model.decoder.embed_positions.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma,[CkptWeightInfo('model.decoder.final_layer_norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta,[CkptWeightInfo('model.decoder.final_layer_norm.bias', identity)], identity),
            WeightInfo(W.lm_head,[CkptWeightInfo('lm_head.weight', identity)], identity),
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=self._get_gpt_style_tp_strategy())


class OPT_125M(GPT):
    @staticmethod
    def get_weight_cls():
        return OptWeightInfo

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config_dict = get_config_from_path(ckpt_path)
        config = GptInitModelParameters(
            head_num=config_dict['num_attention_heads'],
            size_per_head=config_dict['hidden_size'] // config_dict['num_attention_heads'],
            layer_num=config_dict.get('num_hidden_layers', 12),
            vocab_size= config_dict['vocab_size'],
            max_seq_len=config_dict['max_position_embeddings'] + 2
        )
        config.layernorm_type = 'pre_layernorm'
        config.norm_type = "layernorm"
        config.has_post_decoder_layernorm=True
        config.hidden_size = config_dict['hidden_size']
        config.inter_size = config_dict["ffn_dim"]
        config.has_positional_encoding = True
        config.activation_type = 'relu'
        config.add_special_tokens = True
        config.special_tokens.eos_token_id = config_dict.get('eos_token_id', 2)
        config.special_tokens.pad_token_id = config_dict.get('pad_token_id', 1)
        config.special_tokens.bos_token_id = config_dict.get('bos_token_id', 2)
        config.head_num_kv = config.head_num
        return config

register_model('OPT_125M', OPT_125M, ['facebook/opt-125m'])