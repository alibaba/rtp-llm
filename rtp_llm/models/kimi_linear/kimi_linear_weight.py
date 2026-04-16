import functools
from typing import Any, Dict, List

import torch

from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight
from rtp_llm.ops import HybridAttentionType, LinearAttentionConfig
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
    transpose_pad,
)


def merge_conv1d(ts: List[torch.Tensor]) -> torch.Tensor:
    """Merge separate q_conv1d, k_conv1d, v_conv1d into one tensor."""
    return torch.cat(ts, dim=0)


class KimiLinearWeight(DeepSeekV2Weight):
    """Kimi Linear weight loading.

    Inherits MLA + MoE weight loading from DeepSeekV2Weight.
    Adds KDA linear attention weights for KDA layers.
    Overrides MoE weight names for block_sparse_moe format.
    """

    def _process_meta(self, meta_dict, weight_keys):
        self.has_e_score_correction_bias = False
        # Check for q_a_proj to determine MLA q_use_lora
        if "model.layers.0.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True
        # Check for e_score_correction_bias in block_sparse_moe format
        for layer_id in range(self._num_layers):
            if (
                f"model.layers.{layer_id}.block_sparse_moe.gate.e_score_correction_bias"
                in weight_keys
            ):
                self.has_e_score_correction_bias = True
                break

    def _create_rope_w(self):
        """Kimi Linear does not use positional encoding for MLA layers."""
        return None

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_type = self.model_config.hybrid_attention_config.hybrid_attention_types[
            layer_id
        ]
        if layer_type == HybridAttentionType.LINEAR:
            layer_weights: List[WeightModule] = []
            layer_weights.extend(self._create_kda_layer_norm_weight())
            layer_weights.extend(self._create_kda_attention_weight())
            layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
            return layer_weights
        else:
            # Full MLA attention layer: use DeepSeekV2 MLA weights + FFN
            return super()._get_hf_layer_weight_info(layer_id)

    def _create_kda_layer_norm_weight(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
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
        ]

    def _create_kda_attention_weight(self) -> List[WeightModule]:
        linear_attn_cfg = LinearAttnConfig(self.model_config.linear_attention_config)
        weights: List[WeightModule] = []

        # Merged q + k + v projection
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                merge_qkv_hf,
                linear_attn_cfg,
            )
        )

        # b_proj: [num_heads, hidden] -> transpose -> [hidden, num_heads]
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_b_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.b_proj.weight", identity)],
                transpose,
                linear_attn_cfg,
            )
        )

        # Forget gate LoRA: f_a_proj [lora_rank, hidden], f_b_proj [num_heads*head_dim, lora_rank]
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_f_a_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.f_a_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            )
        )
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_f_b_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.f_b_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            )
        )

        # Output gate LoRA: g_a_proj [lora_rank, hidden], g_b_proj [num_heads*head_dim, lora_rank]
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_g_a_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.g_a_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            )
        )
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_g_b_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.g_b_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            )
        )

        # Merged conv1d: q_conv1d + k_conv1d + v_conv1d
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_conv1d.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_conv1d.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_conv1d.weight", identity
                    ),
                ],
                merge_conv1d,
                linear_attn_cfg,
            )
        )

        # o_norm (RmsNormGated weight)
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_norm_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_norm.weight", identity)],
                identity,
                linear_attn_cfg,
            )
        )

        # dt_bias: [num_heads * head_dim] — keep fp32 for gate precision
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_dt_b_kda,
                [CkptWeightInfo("model.layers.{i}.self_attn.dt_bias", identity)],
                identity,
                linear_attn_cfg,
                data_type=torch.float32,
            )
        )

        # A_log: [1, 1, num_heads, 1] -> squeeze to [num_heads] — keep fp32 for gate precision
        def squeeze_alog(ts):
            return ts[0].squeeze()

        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo("model.layers.{i}.self_attn.A_log", identity)],
                squeeze_alog,
                linear_attn_cfg,
                data_type=torch.float32,
            )
        )

        # o_proj: [hidden, num_heads * head_dim]
        weights.append(
            LinearAttnAtomicWeight(
                W.linear_attn_out_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight", identity)],
                transpose,
                linear_attn_cfg,
            )
        )

        return weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        """Override FFN weight loading for block_sparse_moe format.

        Kimi Linear uses two FFN variants:

        1. MoE layers (KimiSparseMoeBlock, most layers):
           - gate (KimiMoEGate):
               block_sparse_moe.gate.weight                  -> router weights
               block_sparse_moe.gate.e_score_correction_bias -> expert score bias
           - shared_experts (KimiMLP, shared across tokens):
               block_sparse_moe.shared_experts.gate_proj.weight  -> w1 (gate)
               block_sparse_moe.shared_experts.down_proj.weight  -> w2 (down)
               block_sparse_moe.shared_experts.up_proj.weight    -> w3 (up)
           - experts (list of KimiBlockSparseMLP, routed per token):
               block_sparse_moe.experts.{id}.w1.weight  -> gate projection
               block_sparse_moe.experts.{id}.w2.weight  -> down projection
               block_sparse_moe.experts.{id}.w3.weight  -> up projection

        2. Dense FFN layers (KimiMLP, layer 0):
               mlp.gate_proj.weight  -> w1 (gate)
               mlp.down_proj.weight  -> w2 (down)
               mlp.up_proj.weight    -> w3 (up)
        """
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
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.shared_experts.gate_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.shared_experts.down_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=1
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.shared_experts.up_proj.weight",
                                    identity,
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
                MoeWeight(
                    sub_weights=[
                        MoeAtomicWeight(
                            W.moe_gate,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.gate.weight",
                                    identity,
                                )
                            ],
                            transpose,
                            config=moe_config,
                        ),
                        MoeAtomicWeight(
                            W.moe_w2,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w2.weight",
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
                                    "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w3.weight",
                                    identity,
                                ),
                                CkptWeightInfo(
                                    "model.layers.{i}.block_sparse_moe.experts.{expert_id}.w1.weight",
                                    identity,
                                ),
                            ],
                            stack_moe_w1,
                            config=moe_config,
                        ),
                    ],
                    config=moe_config,
                ),
            ]
            if self.has_e_score_correction_bias:
                layer_weights.append(
                    AtomicWeight(
                        W.e_score_correction_b,
                        [
                            CkptWeightInfo(
                                "model.layers.{i}.block_sparse_moe.gate.e_score_correction_bias",
                                identity,
                            )
                        ],
                        identity,
                        data_type=torch.float32,
                    )
                )
            return layer_weights
        else:
            # Dense FFN layer (layer 0)
            return [
                FfnWeight(
                    sub_weights=[
                        FfnAtomicWeight(
                            W.ffn_w1,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.gate_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w2,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.down_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=1
                            ),
                            config=ffn_config,
                        ),
                        FfnAtomicWeight(
                            W.ffn_w3,
                            [
                                CkptWeightInfo(
                                    "model.layers.{i}.mlp.up_proj.weight", identity
                                )
                            ],
                            functools.partial(
                                transpose_pad, align_size=align_size, dim=0
                            ),
                            config=ffn_config,
                        ),
                    ],
                    config=ffn_config,
                ),
            ]
