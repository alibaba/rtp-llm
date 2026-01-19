import json
import os

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
    MoeWithSharedWeight,
)
from rtp_llm.models.qwen_v2 import QWenV2, QWenV2Weight
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
)


class QWenV2MoeWeight(QWenV2Weight):
    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = super()._get_hf_layer_weight_info(layer_id)

        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):

        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            routed_scaling_factor=1.0,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
        )
        return [
            MoeWithSharedWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        transpose,
                        config=moe_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.gate_proj.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.down_proj.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert.up_proj.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=ffn_config,
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
                    MoeAtomicWeight(
                        W.shared_expert_gate,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert_gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]


class Qwen2Moe(QWenV2):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        from rtp_llm.model_config_creators.qwen import create_qwen_v2_moe_config

        config = create_qwen_v2_moe_config(ckpt_path)
        return config

    @staticmethod
    def get_weight_cls():
        return QWenV2MoeWeight


register_model("qwen_2_moe", Qwen2Moe, ["Qwen2MoeForCausalLM"])
