import os
import json
from typing import Any
from rtp_llm.models.qwen_v2_moe import Qwen2Moe, QWenV2MoeWeight
from rtp_llm.utils.model_weight import W, CkptWeightInfo, identity, transpose, stack_, stack_moe_w1
from rtp_llm.model_loader.ffn_weight import FfnConfig, MoeConfig, MoeAtomicWeight, MoeWeight
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model

class QWenV3MoeWeight(QWenV2MoeWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.bias = False        

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size,
            routed_scaling_factor=1.0
        )
        return [
            MoeWeight(sub_weights = [
                MoeAtomicWeight(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], 
                                transpose, config=moe_config),
                MoeAtomicWeight(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight', identity)] + \
                        [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight', identity)], 
                        stack_moe_w1, config=moe_config),
                MoeAtomicWeight(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight', identity)], 
                                stack_, config=moe_config)],
                config=moe_config)
        ]

class Qwen3Moe(Qwen2Moe):
    @staticmethod
    def get_weight_cls():
        return QWenV3MoeWeight
    
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.use_qk_norm = True
        return config

register_model('qwen_3_moe', Qwen3Moe, ["Qwen3MoeForCausalLM"])
