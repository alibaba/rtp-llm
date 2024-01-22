from typing import List
import os
import json
import functools
import torch

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.utils.model_weight import W, WeightInfo, ModelWeightInfo, \
    ModelDeployWeightInfo, CkptWeightInfo, identity, zeros, transpose, concat_1, concat_0
from maga_transformer.models.gpt import GPT
from maga_transformer.model_factory_register import register_model

def merge_qkv_hf(ts: List[torch.Tensor]):
    q, k, v = ts
    qkv_weight = torch.concat([q.T, k.T, v.T], dim=1).contiguous()
    return qkv_weight

def stack_(ts: List[torch.Tensor]):
    return torch.stack(ts, dim=0)

class MixtralWeightInfo(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', concat_1)], identity),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
        ]
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)], identity),
            WeightInfo(W.attn_qkv_b, [], functools.partial(zeros, shape=[self._hidden_size * 3])),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', concat_1)], transpose),
            WeightInfo(W.attn_o_b, [], functools.partial(zeros, shape=[self._hidden_size])),
            WeightInfo(W.ffn_b1, [], functools.partial(zeros, shape=[self.expert_num_, self._inter_size])),
            WeightInfo(W.ffn_b2, [], functools.partial(zeros, shape=[self.expert_num_, self._hidden_size])),
            WeightInfo(W.ffn_gate, [CkptWeightInfo('model.layers.{i}.block_sparse_moe.gate.weight', concat_0)], transpose),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)], identity),
        ]

        layer_weights.append(
                WeightInfo(W.attn_qkv_w,
                           [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', concat_0),
                            CkptWeightInfo('model.layers.{i}.self_attn.k_proj.weight', concat_0),
                            CkptWeightInfo('model.layers.{i}.self_attn.v_proj.weight', concat_0)],
                           functools.partial(merge_qkv_hf)) )
        ffn_w1 = []
        ffn_w2 = []
        ffn_w3 = []
        for num_experts in range(self.expert_num_):
            ffn_w1.append(CkptWeightInfo('model.layers.{i}.block_sparse_moe.experts.'+ str(num_experts) +'.w1.weight', transpose))
            ffn_w2.append(CkptWeightInfo('model.layers.{i}.block_sparse_moe.experts.'+ str(num_experts) +'.w2.weight', transpose))
            ffn_w3.append(CkptWeightInfo('model.layers.{i}.block_sparse_moe.experts.'+ str(num_experts) +'.w3.weight', transpose))
        
        layer_weights.append(WeightInfo(W.ffn_w1,ffn_w1, stack_))
        layer_weights.append(WeightInfo(W.ffn_w2,ffn_w2, stack_))
        layer_weights.append(WeightInfo(W.ffn_w3,ffn_w3, stack_))

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=None)

class Mixtral(GPT):
    @staticmethod
    def get_weight_cls():
        return MixtralWeightInfo

    @staticmethod
    def _create_config(ckpt_path: str):
        config_path = os.path.join(ckpt_path, 'config.json')
        with open(config_path) as f:
            config_json = json.load(f)
        config = GptInitModelParameters(
            head_num=config_json['num_attention_heads'],
            size_per_head=config_json['hidden_size'] // config_json['num_attention_heads'],
            inter_size=config_json['intermediate_size'],
            # layer_num = 1,
            layer_num=config_json['num_hidden_layers'],
            max_seq_len=config_json.get('max_sequence_length', 2048),
            vocab_size=config_json['vocab_size'],
            head_num_kv = config_json['num_key_value_heads'],
            activation_type='Silu',
            use_gated_activation=True,
            norm_type='rmsnorm',
            rotary_embedding_dim=128,
            rotary_embedding_style=1,
            has_post_decoder_layernorm=True,
            # expert_num = 2,
            # moe_k = 1,
            rotary_embedding_base = int(config_json.get('rope_theta', 10000)),
            expert_num = config_json['num_local_experts'],
            moe_k = config_json['num_experts_per_tok'],
            moe_layer_index = [i for i in range(config_json['num_hidden_layers'])])
        return config

register_model('mixtral', Mixtral)