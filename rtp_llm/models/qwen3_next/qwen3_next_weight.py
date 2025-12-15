import functools
from typing import List

import torch

from rtp_llm.config.gpt_init_model_parameters import HybridAttentionType
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
    MoeWithSharedWeight,
    SharedMoeConfig,
)
from rtp_llm.model_loader.liner_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
)


# since this function will be used in both weight and scale(if fp8_per_block), so we need to write with more general way
def split_q_gate(ts: List[torch.Tensor], head_num: int, head_dim: int, part: int):
    dim0, dim1 = ts[0].shape
    assert (
        dim0 % (head_num * 2) == 0
    ), f"dim0 % (head_num * 2) != 0, dim0: {dim0}, head_num: {head_num}, head_dim: {head_dim}, dim1: {dim1}"
    # for weight, it's head_dim; for scale, it's head_dim / block_size
    new_head_dim = dim0 // (head_num * 2)
    t = ts[0].reshape(head_num, 2, new_head_dim, dim1)
    if part == 0:
        return t[:, 0, :, :].reshape(-1, dim1)
    else:
        return t[:, 1, :, :].reshape(-1, dim1)


def plus_one(ts: List[torch.Tensor]):
    return ts[0] + 1


class Qwen3NextWeight(ModelDeployWeightInfo):
    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("model.embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo("model.norm.weight", plus_one)],
            ),
        ]
        all_layer_weights: List[List[WeightModule]] = []
        for idx in range(self._num_layers):
            layer_weight: List[WeightModule] = []
            layer_weight.append(
                AtomicWeight(
                    W.pre_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.input_layernorm.weight",
                            plus_one,
                        )
                    ],
                )
            )
            if (
                self.config.hybrid_attention_config.hybrid_attention_types[idx]
                == HybridAttentionType.LINEAR
            ):
                layer_weight.extend(self._create_linear_attention_weight(idx))
            else:
                layer_weight.extend(self._create_mqa_weight(idx))
            layer_weight.append(
                AtomicWeight(
                    W.post_ln_gamma,
                    [
                        CkptWeightInfo(
                            "model.layers.{i}.post_attention_layernorm.weight", plus_one
                        ),
                    ],
                )
            )
            layer_weight.extend(self._create_ffn_weight(idx))
            all_layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _create_ffn_weight(self, idx: int) -> List[WeightModule]:
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=self._inter_padding_size,
        )
        shared_moe_config = SharedMoeConfig(
            expert_num=self.expert_num_,
            inter_padding_size=self._inter_padding_size,
        )
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            inter_padding_size=self._inter_padding_size,
        )
        return [
            MoeWithSharedWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_gate,
                        [CkptWeightInfo("model.layers.{i}.mlp.gate.weight", identity)],
                        process_fun=transpose,
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
                        process_fun=transpose,
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
                        process_fun=transpose,
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
                        process_fun=transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.shared_expert_gate,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.shared_expert_gate.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=stack_,
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
                        process_fun=stack_moe_w1,
                        config=moe_config,
                    ),
                ],
                config=shared_moe_config,
            )
        ]

    def _create_linear_attention_weight(self, idx: int):
        return [
            LinearAttnAtomicWeight(
                W.linear_attn_qkvz_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.in_proj_qkvz.weight", identity
                    )
                ],
                transpose,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_ba_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.in_proj_ba.weight", identity
                    )
                ],
                transpose,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_norm_w,
                [CkptWeightInfo("model.layers.{i}.linear_attn.norm.weight", identity)],
                identity,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_dt_b,
                [CkptWeightInfo("model.layers.{i}.linear_attn.dt_bias", identity)],
                identity,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.conv1d.weight", identity
                    )
                ],
                identity,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo("model.layers.{i}.linear_attn.A_log", identity)],
                identity,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_out_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.linear_attn.out_proj.weight", identity
                    )
                ],
                transpose,
                LinearAttnConfig(self.config.linear_attention_config),
            ),
        ]

    def _create_mqa_weight(self, idx: int):
        return [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight",
                        functools.partial(
                            split_q_gate,
                            head_num=self.config.head_num,
                            head_dim=self.config.size_per_head,
                            part=0,
                        ),
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_proj.weight",
                        identity,
                    ),
                ],
                process_fun=merge_qkv_hf,
                config=self.attn_config,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.o_proj.weight",
                        identity,
                    )
                ],
                process_fun=transpose,
                config=self.attn_config,
            ),
            AttnAtomicWeight(
                W.attn_gate_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight",
                        functools.partial(
                            split_q_gate,
                            head_num=self.config.head_num,
                            head_dim=self.config.size_per_head,
                            part=1,
                        ),
                    )
                ],
                process_fun=transpose,
                config=self.attn_config,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.self_attn.q_norm.weight", plus_one)],
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.self_attn.k_norm.weight", plus_one)],
            ),
        ]
