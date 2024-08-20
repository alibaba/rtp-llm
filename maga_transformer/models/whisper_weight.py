import functools
import logging
from typing import List
import torch
from typing import List, Any
from einops import rearrange

from maga_transformer.utils.model_weight import (W, WeightInfo, ModelWeightInfo,
                                                 ModelDeployWeightInfo, CkptWeightInfo, concat_1,
                                                 concat_0, identity, zeros, transpose, merge_qkv_lora_A,
                                                 merge_qkv_lora_B, shift_one, pad, merge_qkv_b)
from maga_transformer.utils.group_quant_weight_util import get_layer_group_quant_weight_info

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

def append_k_bias(ts: List[torch.Tensor], k_size: int):
    q, v = ts
    qkv_bias = torch.concat([q, torch.zeros((k_size)), v], dim = 0).contiguous()
    return qkv_bias

class WhisperWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.positional_embedding, [CkptWeightInfo('model.decoder.embed_positions.weight', identity)], identity),
            WeightInfo(W.embedding, [CkptWeightInfo('model.decoder.embed_tokens.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.decoder.layer_norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [CkptWeightInfo('model.decoder.layer_norm.bias', identity)], identity)
        ]
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.weight', identity)]),
            WeightInfo(W.pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.bias', identity)]),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.weight', identity)], transpose),
            WeightInfo(W.attn_o_b, [CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.bias', identity)], identity),
            WeightInfo(W.attn_qkv_w, [
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.k_proj.weight', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.weight', identity)
            ], merge_qkv_hf),
            WeightInfo(W.attn_qkv_b, [
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.bias', identity),
                CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.bias', identity)
            ], functools.partial(append_k_bias, k_size = self._head_num_kv * self._size_per_head)),

            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.decoder.layers.{i}.fc1.weight', identity)], identity),
            WeightInfo(W.ffn_b1, [CkptWeightInfo('model.decoder.layers.{i}.fc1.bias', identity)], identity),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('model.decoder.layers.{i}.fc2.weight', identity)], identity),
            WeightInfo(W.ffn_b3, [CkptWeightInfo('model.decoder.layers.{i}.fc2.bias', identity)], identity),

            WeightInfo(W.cross_attn_pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn_layer_norm.weight', identity)], identity),
            WeightInfo(W.cross_attn_pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn_layer_norm.bias', identity)], identity),

            WeightInfo(W.cross_attn_k_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.k_proj.weight', identity)], identity),
            WeightInfo(W.cross_attn_k_b, [], functools.partial(zeros, shape=[self._head_num_kv * self._size_per_head])),
            WeightInfo(W.cross_attn_v_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.v_proj.weight', identity)], identity),
            WeightInfo(W.cross_attn_v_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.v_proj.bias', identity)], identity),
            WeightInfo(W.cross_attn_q_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.q_proj.weight', identity)], identity),
            WeightInfo(W.cross_attn_q_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.q_proj.bias', identity)], identity),

            WeightInfo(W.cross_attn_o_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.out_proj.weight', identity)], identity),
            WeightInfo(W.cross_attn_q_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.out_proj.bias', identity)], identity),

            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.weight', identity)]),
            WeightInfo(W.post_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.bias', identity)]),
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)