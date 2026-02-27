import functools
from typing import List, Optional

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWithSharedWeight,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.llama_weight import merge_qkv_hf
from rtp_llm.models.multimodal.multimodal_mixin import (
    BaseMultiModalWeightInfo,
    BaseVitWeights,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
    zeros,
)


class DeepSeekVLV2Weight(ModelDeployWeightInfo, BaseMultiModalWeightInfo):
    q_use_lora = False
    has_e_score_correction_bias = False

    def __init__(self, vit_weights, **kwargs):
        ModelDeployWeightInfo.__init__(self, **kwargs)
        BaseMultiModalWeightInfo.__init__(self, vit_weights=vit_weights, **kwargs)

    def _process_meta(self, meta_dict, weight_keys):
        if "language.model.layers.0.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        for layer_id in range(self._num_layers):
            if (
                f"language.model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
                in weight_keys
            ):
                self.has_e_score_correction_bias = True
                break

    def _get_hf_layer_weight_info(self, layer_id: int):
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        layer_weights = [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        "language.model.layers.{i}.input_layernorm.weight", identity
                    )
                ],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "language.model.layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "language.model.layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "language.model.layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                functools.partial(
                    merge_qkv_hf,
                    hidden_size=self._hidden_size,
                    head_num_kv=self._head_num_kv,
                    head_num=self._head_num,
                ),
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "language.model.layers.{i}.self_attn.o_proj.weight", identity
                    )
                ],
                transpose,
                config=attn_config,
            ),
            AttnAtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "language.model.layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
                identity,
            ),
        ]
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        align_size = self._align_size

        ffn_config = FfnConfig(
            align_size=align_size,
            is_gated_activation=self._is_gated_activation,
            is_moe=False,
        )

        if layer_id in self.moe_layer_index_:
            moe_config = MoeConfig(
                align_size=align_size,
                expert_num=self.expert_num_,
            )
            layer_weights = [
                MoeWithSharedWeight(
                    sub_weights=[
                        MoeAtomicWeight(
                            W.moe_gate,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.gate.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=moe_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.shared_experts.gate_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.shared_experts.down_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.shared_experts.up_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        MoeAtomicWeight(
                            W.moe_w2,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                    identity,
                                )
                            ],
                            stack_,
                            config=moe_config,
                        ),
                        MoeAtomicWeight(
                            W.moe_w1,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                                    identity,
                                )
                            ]
                            + [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                                    identity,
                                )
                            ],
                            stack_moe_w1,
                            config=moe_config,
                        ),
                    ],
                    config=moe_config,
                )
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(
                    AtomicWeight(
                        W.e_score_correction_b,
                        [
                            CkptWeightInfo(
                                "language.model.layers.{i}.mlp.gate.e_score_correction_bias",
                                identity,
                            )
                        ],
                        identity,
                        data_type=torch.float32,
                    )
                )
            return layer_weights
        else:

            return [
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.gate_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.down_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=1,
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "language.model.layers.{i}.mlp.up_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad,
                                align_size=align_size,
                                dim=0,
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        return None

    def _get_weight_info(self):
        layer_weights: List[List[WeightModule]] = []
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("language.model.embed_tokens.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("language.model.norm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("language.lm_head.weight", identity)],
                identity,
            ),
        ]
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)


class DeepSeekVLV2VitWeight(BaseVitWeights):
    def _set_weight_prefix(self):
        self._ckpt_prefix = ""
        self._ft_prefix = "self.mm_part."
