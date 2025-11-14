from typing import Any, List, Optional

from rtp_llm.config.model_config import ModelConfig
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
    def __init__(self, model_config, parallelism_config, hw_kernel_config, kv_cache_config, merge_lora=False, vit_config=None, prefix="", **kwargs: Any):
        super().__init__(model_config=model_config, parallelism_config=parallelism_config, hw_kernel_config=hw_kernel_config, kv_cache_config=kv_cache_config, merge_lora=merge_lora, vit_config=vit_config, prefix=prefix, **kwargs)
        self.bias = False

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
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
        model_config = self.model_config
        parallelism_config = self.parallelism_config
        fmha_config = self.fmha_config
        py_hw_kernel_config = self.hw_kernel_config
        moe_config = self.moe_config
        max_generate_batch_size = self.max_generate_batch_size
        
        self.py_model = GenericMoeModel(
            model_config,
            parallelism_config,
            self.weight,
            moe_config,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


class Qwen3MoeEagle3Weight(QWenV2Weight):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
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
