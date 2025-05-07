import os
import json

from maga_transformer.models.qwen_v2_moe import Qwen2Moe, QWenV2MoeWeight
from maga_transformer.utils.model_weight import W, CkptWeightInfo, identity, transpose, stack_, stack_moe_w1
from maga_transformer.model_loader.ffn_weight import FfnConfig, MoeConfig, MoeAtomicWeight, MoeWeight
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model



class QWenV3MoeWeight(QWenV2MoeWeight):
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


register_model('qwen_3_moe', Qwen3Moe, ["Qwen3MoeForCausalLM"])
