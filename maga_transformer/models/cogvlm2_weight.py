import functools
from typing import List

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
)
from maga_transformer.utils.model_weight import (
    CkptWeightInfo,
    ModelDeployWeightInfo,
    ModelWeightInfo,
    W,
    WeightInfo,
    identity,
    transpose,
    zeros,
)

from ..config.gpt_init_model_parameters import GptInitModelParameters


class CogVLM2Weight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        return self._get_hf_weight_info()

    def _get_hf_layer_weight_info(self):
        layer_weights = [
            WeightInfo(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
            ),
            WeightInfo(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.language_expert_query_key_value.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.language_expert_dense.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.vision_attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_query_key_value.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.vision_attn_qkv_b,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_query_key_value.bias",
                        identity,
                    )
                ],
                identity,
            ),
            WeightInfo(
                W.vision_attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.vision_expert_dense.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.post_attention_layernorm.weight", identity
                    )
                ],
                identity,
            ),
            WeightInfo(
                W.ffn_w2,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.language_mlp.down_proj.weight", identity
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.ffn_w1,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.language_mlp.gate_proj.weight", identity
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.ffn_w3,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.language_mlp.up_proj.weight", identity
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.vision_ffn_w2,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.vision_mlp.down_proj.weight", identity
                    )
                ],
                transpose,
            ),
            WeightInfo(
                W.vision_ffn_w1,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.mlp.vision_mlp.gate_proj.weight", identity
                    )
                ],
                transpose,
            ),
            WeightInfo(
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
            WeightInfo(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
                identity,
            ),
            WeightInfo(
                W.lm_head, [CkptWeightInfo("lm_head.weight", identity)], identity
            ),
            WeightInfo(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", identity)],
                identity,
            ),
            WeightInfo(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
        ]

        layer_weights: List[List[WeightInfo]] = [
            self._get_hf_layer_weight_info() for _ in range(self._num_layers)
        ]

        return ModelWeightInfo(
            layer_weights=layer_weights,
            weights=weights,
            tp_strategy=self._get_gpt_style_tp_strategy(),
        )


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
