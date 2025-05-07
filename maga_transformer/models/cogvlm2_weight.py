import functools
from typing import List

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
)
from maga_transformer.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    transpose,
    zeros,
)
from maga_transformer.model_loader.weight_module import WeightModule, AtomicWeight
from maga_transformer.model_loader.ffn_weight import FfnAtomicWeight, FfnWeight
from maga_transformer.model_loader.attn_weight import AttnAtomicWeight
from maga_transformer.model_loader.model_weight_info import ModelWeightInfo, ModelDeployWeightInfo

from ..config.gpt_init_model_parameters import GptInitModelParameters


class CogVLM2Weight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_layer_weight_info(self):
        attn_config = self.attn_config
        ffn_config = self.ffn_config
        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.language_expert_query_key_value.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.language_expert_dense.weight",
                        identity,
                    )
                ],
                transpose,
                config=attn_config
            ),
            AtomicWeight(
                W.vision_attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_query_key_value.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.vision_attn_qkv_b,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_query_key_value.bias",
                        identity,
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.vision_attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_dense.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.post_attention_layernorm.weight", identity
                    )
                ],
                identity,
            ),
            FfnWeight(sub_weights=[
                FfnAtomicWeight(
                    W.ffn_w2,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.language_mlp.down_proj.weight", identity
                        )
                    ],
                    transpose,
                    config=ffn_config
                ),
                FfnAtomicWeight(
                    W.ffn_w1,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.language_mlp.gate_proj.weight", identity
                        )
                    ],
                    transpose,
                    config=ffn_config
                ),
                FfnAtomicWeight(
                    W.ffn_w3,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.mlp.language_mlp.up_proj.weight", identity
                        )
                    ],
                    transpose,
                    config=ffn_config
                )], config=ffn_config),
            AtomicWeight(
                W.vision_ffn_w2,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.vision_mlp.down_proj.weight", identity
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.vision_ffn_w1,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.vision_mlp.gate_proj.weight", identity
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.vision_ffn_w3,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.vision_mlp.up_proj.weight", identity
                    )
                ],
                transpose,
            ),
        ]
        return layer_weights

    def _get_hf_weight_info(self):
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
        ]

        layer_weights: List[List[WeightModule]] = [
            self._get_hf_layer_weight_info() for _ in range(self._num_layers)
        ]

        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class CogVLM2VitWeights(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = "model.vision."
        self._ft_prefix = "self.mm_part.vit."


class CogVLM2WeightInfo(CogVLM2Weight, BaseMultiModalWeightInfo):
    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        CogVLM2Weight.__init__(self, config, tp_size, tp_rank)
        BaseMultiModalWeightInfo.__init__(self, config)

    def _get_weight_info(self):
        cogvlm2_weight = super()._get_weight_info()
        self._get_vit_info(cogvlm2_weight)
        return cogvlm2_weight
