import functools
from typing import List
import torch
from typing import List

from maga_transformer.utils.model_weight import (W, CkptWeightInfo, identity, zeros, transpose)
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight, FfnConfig
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

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
            AtomicWeight(W.positional_embedding, [CkptWeightInfo('model.decoder.embed_positions.weight', identity)], identity),
            AtomicWeight(W.embedding, [CkptWeightInfo('model.decoder.embed_tokens.weight', identity)], identity),
            AtomicWeight(W.final_ln_gamma, [CkptWeightInfo('model.decoder.layer_norm.weight', identity)], identity),
            AtomicWeight(W.final_ln_beta, [CkptWeightInfo('model.decoder.layer_norm.bias', identity)], identity)
        ]
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = []
        for _ in range(self._num_layers):
            layer_weight = [
                AtomicWeight(W.pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.weight', identity)]),
                AtomicWeight(W.pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.self_attn_layer_norm.bias', identity)]),
                AttnAtomicWeight(W.attn_o_w, [CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.weight', identity)], transpose, config=attn_config),
                AttnAtomicWeight(W.attn_o_b, [CkptWeightInfo('model.decoder.layers.{i}.self_attn.out_proj.bias', identity)], identity, config=attn_config),
                AttnAtomicWeight(W.attn_qkv_w, [
                    CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.weight', identity),
                    CkptWeightInfo('model.decoder.layers.{i}.self_attn.k_proj.weight', identity),
                    CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.weight', identity)
                ], merge_qkv_hf, config=attn_config),
                AttnAtomicWeight(W.attn_qkv_b, [
                    CkptWeightInfo('model.decoder.layers.{i}.self_attn.q_proj.bias', identity),
                    CkptWeightInfo('model.decoder.layers.{i}.self_attn.v_proj.bias', identity)
                ], functools.partial(append_k_bias, k_size = self._head_num_kv * self._size_per_head), config=attn_config),
                FfnWeight(sub_weights=[
                    FfnAtomicWeight(W.ffn_w1, [CkptWeightInfo('model.decoder.layers.{i}.fc1.weight', identity)], identity, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b1, [CkptWeightInfo('model.decoder.layers.{i}.fc1.bias', identity)], identity, config=ffn_config),
                    FfnAtomicWeight(W.ffn_w3, [CkptWeightInfo('model.decoder.layers.{i}.fc2.weight', identity)], identity, config=ffn_config),
                    FfnAtomicWeight(W.ffn_b3, [CkptWeightInfo('model.decoder.layers.{i}.fc2.bias', identity)], identity, config=ffn_config)],
                        config=ffn_config),

                AtomicWeight(W.cross_attn_pre_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn_layer_norm.weight', identity)], identity),
                AtomicWeight(W.cross_attn_pre_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn_layer_norm.bias', identity)], identity),

                AtomicWeight(W.cross_attn_k_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.k_proj.weight', identity)], identity),
                AtomicWeight(W.cross_attn_k_b, [], functools.partial(zeros, shape=[self._head_num_kv * self._size_per_head])),
                AtomicWeight(W.cross_attn_v_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.v_proj.weight', identity)], identity),
                AtomicWeight(W.cross_attn_v_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.v_proj.bias', identity)], identity),
                AtomicWeight(W.cross_attn_q_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.q_proj.weight', identity)], identity),
                AtomicWeight(W.cross_attn_q_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.q_proj.bias', identity)], identity),

                AtomicWeight(W.cross_attn_o_w, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.out_proj.weight', identity)], identity),
                AtomicWeight(W.cross_attn_q_b, [CkptWeightInfo('model.decoder.layers.{i}.encoder_attn.out_proj.bias', identity)], identity),

                AtomicWeight(W.post_ln_gamma, [CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.weight', identity)]),
                AtomicWeight(W.post_ln_beta, [CkptWeightInfo('model.decoder.layers.{i}.final_layer_norm.bias', identity)]),
            ]
            layer_weights.append(layer_weight)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)