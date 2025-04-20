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
from maga_transformer.utils.model_weight import W, WeightInfo, CkptWeightInfo, identity, transpose_pad, transpose, stack_, stack_moe_w1
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model

class QWenV2MoeWeight(QWenV2Weight):
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

