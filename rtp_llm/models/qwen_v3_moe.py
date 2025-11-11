from typing import Any, List, Optional

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.model_weight_info import ModelWeightInfo
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.models.qwen_v2_moe import Qwen2Moe, QWenV2MoeWeight
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
)


class QWenV3MoeWeight(QWenV2MoeWeight):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.bias = False

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=(
                self._layer_inter_padding_size[layer_id]
                if self._layer_inter_padding_size
                else self._inter_padding_size
            ),
            routed_scaling_factor=1.0,
        )
        return [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ]
                        + [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                identity,
                            )
                        ],
                        stack_moe_w1,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        stack_,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]


class Qwen3Moe(Qwen2Moe):
    @staticmethod
    def get_weight_cls():
        return QWenV3MoeWeight

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        config.qk_norm = True
        return config

    def _create_python_model(self) -> Optional[GptModelBase]:
        self.py_model = GenericMoeModel(self.config, self.weight)
        return self.py_model


class Qwen3MoeEagle3Weight(QWenV2Weight):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)
        self.bias = False
        self._use_qk_norm = True

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(self.prefix + "lm_head.weight", identity)],
                identity,
            ),
        ]
        assert self._num_layers == 1
        for layer in range(self._num_layers):
            layer_weights_tmp = self._get_hf_layer_weight_info(layer)
            layer_weights_tmp.extend(
                [
                    AtomicWeight(
                        W.eagle3_fc_proj,
                        [CkptWeightInfo("fc.weight", identity)],
                        transpose,
                    ),
                    AtomicWeight(
                        W.eagle3_fc_norm_gamma,
                        [CkptWeightInfo("model.layers.0.hidden_norm.weight", identity)],
                        identity,
                    ),
                    AtomicWeight(
                        W.eagle3_input_norm_gamma,
                        [
                            CkptWeightInfo(
                                "model.layers.0.input_layernorm.weight", identity
                            )
                        ],
                        identity,
                    ),
                ]
            )
            layer_weights.append(layer_weights_tmp)

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class Qwen3MoeEagle3(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = super()._create_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return Qwen3MoeEagle3Weight


register_model("qwen_3_moe", Qwen3Moe, ["Qwen3MoeForCausalLM"])
register_model("qwen_3_moe_eagle3", Qwen3MoeEagle3, ["Qwen3MoeForCausalLMEagle"])
register_model("qwen3_coder_moe", Qwen3Moe, [])
