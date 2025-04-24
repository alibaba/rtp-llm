import torch
import unicodedata
import types
import functools
import itertools
import os
import json
from typing import List, Any

from maga_transformer.eplb.ep_balancer import MoeWeightInfo
from maga_transformer.models.qwen_v2 import QWenV2, QWenV2Weight
from maga_transformer.utils.model_weight import W, WeightInfo, CkptWeightInfo, identity, transpose, stack_, stack_moe_w1, concat_0
from maga_transformer.utils.util import check_with_info
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model

def fp8_view(ts: List[torch.Tensor]):
    m, n = ts[0].shape
    return ts[0].reshape(n, m)

def qkv_concat_fp8(ts: List[torch.Tensor]):
    return fp8_view([concat_0(ts)])

def transpose_moe_down(ts: List[torch.Tensor]):
    return stack_(ts).transpose(1, 2)    

def merge_qkv_scale(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 3, "qkv scale should have 3 tensors")
    out_scale = torch.concat(ts, dim=0)
    return torch.max(out_scale, dim=0).unsqueeze(0)

def merge_qkv_hf_fp8_with_scale_t(ts: List[torch.Tensor]):
    check_with_info(len(ts) == 6, "qkv weight+scale should have 6 tensors")
    origin_scales = ts[3:]
    max_scale = merge_qkv_scale(origin_scales)
    q, k, v, q_scale_inv, k_scale_inv, v_scale_inv = ts
    q = q * q_scale_inv
    k = k * k_scale_inv
    v = v * v_scale_inv
    quanted_qkv = (torch.cat([q, k, v], dim=0) / max_scale).transpose(0, 1).to(torch.float8_e4m3fn)
    return quanted_qkv

class QWenV2MoeWeight(QWenV2Weight):
    def _get_fp8_moe_layer_weight_info(self, layer_id: int):
        return [
            WeightInfo(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight')], transpose),
            WeightInfo(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight') for expert_id in range(self.expert_num_)] + \
                       [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight') for expert_id in range(self.expert_num_)], stack_moe_w1),
            WeightInfo(W.moe_s1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight_scale_inv') for expert_id in range(self.expert_num_)] + \
                       [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight_scale_inv') for expert_id in range(self.expert_num_)], stack_moe_w1),
            WeightInfo(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight') \
                                  for expert_id in range(self.expert_num_)], stack_),
            WeightInfo(W.moe_s2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight_scale_inv') \
                        for expert_id in range(self.expert_num_)], stack_),
            # WeightInfo(W.shared_expert_gate, [CkptWeightInfo('model.layers.{i}.mlp.shared_expert_gate.weight')], transpose),
        ]

    def _get_fp8_hf_layer_weight_info(self, layer_id: int):
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo(self.prefix + 'model.layers.{i}.input_layernorm.weight')]),
            WeightInfo(W.attn_qkv_w, [
                    CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.q_proj.weight'),
                    CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.k_proj.weight'),
                    CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.v_proj.weight'),
                ],
                concat_0),
            WeightInfo(W.attn_qkv_s, [CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.q_proj.weight_scale_inv'),
                                           CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.k_proj.weight_scale_inv'),
                                           CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.v_proj.weight_scale_inv')],
                                           concat_0),
            WeightInfo(W.q_ln_gamma, [CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.q_norm.weight')]),
            WeightInfo(W.k_ln_gamma, [CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.k_norm.weight')]),
            WeightInfo(W.attn_o_w, [CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.o_proj.weight')]),
            WeightInfo(W.attn_o_s, [CkptWeightInfo(self.prefix + 'model.layers.{i}.self_attn.o_proj.weight_scale_inv')]),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo(self.prefix + 'model.layers.{i}.post_attention_layernorm.weight')]),
        ]
        layer_weights.extend(self._get_fp8_moe_layer_weight_info(layer_id))
        return layer_weights


    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)

        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        # TODO: fix me
        # inter_padding_size: int = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        # trans_pad = functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)
        # inter_padding_size: int = self._inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size
        selected_experts = self._get_selected_experts(layer_id)
        return [
            WeightInfo(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], transpose),
            WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.shared_expert.gate_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.shared_expert.down_proj.weight', identity)], transpose),
            WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.shared_expert.up_proj.weight', identity)], transpose),
            WeightInfo(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight', identity) for expert_id in selected_experts] + \
                       [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight', identity) for expert_id in selected_experts], stack_moe_w1),
            WeightInfo(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight', identity) \
                                  for expert_id in selected_experts], stack_),
            WeightInfo(W.shared_expert_gate, [CkptWeightInfo('model.layers.{i}.mlp.shared_expert_gate.weight', identity)], transpose),
        ]

class Qwen2Moe(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        Qwen2Moe.load_moe_config(ckpt_path, config)
        return config

    @classmethod
    def load_moe_config(cls, ckpt_path: str, config: GptInitModelParameters):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise Exception("qwen2 moe should have config.json")
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
        config.moe_k = config_json['num_experts_per_tok']
        config.expert_num = config_json['num_experts']
        config.moe_inter_padding_size=config_json['moe_intermediate_size']
        config.inter_size = config_json['shared_expert_intermediate_size']
        config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
        config.has_moe_norm = config_json.get("norm_topk_prob", False)
        # step for moe layer
        config.moe_style = 2
        moe_step = config_json['decoder_sparse_step']

        # todo
        # qwen2 moe is supposed to have different inter size for moe and normal layers
        # so there should be two config for ffnlayer
        if moe_step != 1:
            raise Exception("Paritial moe weights for qwen2 is not implemented yet!")
        config.moe_layer_index = [i for i in range(moe_step - 1,  config.layer_num, moe_step)]

    @staticmethod
    def get_weight_cls():
        return QWenV2MoeWeight

    def create_moe_weight_info(self):
        gate = CkptWeightInfo("model.layers.{}.mlp.experts.{}.gate_proj.weight", identity)
        up = CkptWeightInfo("model.layers.{}.mlp.experts.{}.up_proj.weight", identity)
        down = CkptWeightInfo("model.layers.{}.mlp.experts.{}.down_proj.weight", identity)
        return MoeWeightInfo(gate, up, down)


register_model('qwen_2_moe', Qwen2Moe, ["Qwen2MoeForCausalLM"])

