import functools
from typing import Any, Dict, List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import MoeAtomicWeight, MoeConfig, MoeWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.ops import HybridAttentionType
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    transpose,
)


def _mamba_geometry_from_config(config):
    """Compute mamba2 geometry from the model config.

    d_inner = value_heads * value_dim, n_groups = key heads,
    dstate = key head dim, nheads = value heads.
    """
    linear = config.linear_attention_config
    d_inner = linear.linear_num_value_heads * linear.linear_value_head_dim
    n_groups = linear.linear_num_key_heads
    dstate = linear.linear_key_head_dim
    nheads = linear.linear_num_value_heads
    return d_inner, n_groups, dstate, nheads


def make_mamba_reorder(config):
    d_inner, n_groups, dstate, nheads = _mamba_geometry_from_config(config)

    def reorder_in_proj(ts: List[torch.Tensor]) -> torch.Tensor:
        t = ts[0]
        sizes = [d_inner, d_inner, n_groups * dstate, n_groups * dstate, nheads]
        gate, x, b, c, dt = torch.split(t, sizes, dim=0)
        return torch.cat([b, c, x, gate, dt], dim=0)

    def reorder_conv(ts: List[torch.Tensor]) -> torch.Tensor:
        t = ts[0]
        x, b, c = torch.split(
            t, [d_inner, n_groups * dstate, n_groups * dstate], dim=0
        )
        return torch.cat([b, c, x], dim=0)

    return reorder_in_proj, reorder_conv


class NemotronHWeight(ModelDeployWeightInfo):
    """NemotronH weights (backbone. prefix).

    BF16 checkpoint layout (native or dequantized from the ModelOpt NVFP4
    release): mamba in/out projections, GQA attention qkv/o, router, shared
    expert and embeddings are dense tensors; routed experts are stacked
    up/down pairs without gating projections.
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "backbone."

    def _layer_prefix(self) -> str:
        """Per-layer checkpoint template; the loader fills "{i}" per layer."""
        return self.prefix + "layers.{i}."

    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embeddings.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm_f.weight", identity)],
            ),
        ]
        reorder_in_proj, reorder_conv = make_mamba_reorder(self.model_config)
        all_layer_weights: List[List[WeightModule]] = []
        for idx in range(self._num_layers):
            layer_weight: List[WeightModule] = []
            layer_weight.extend(self._create_layer_norm_weight(idx))
            if self._block_type(idx) == "mamba":
                layer_weight.extend(
                    self._create_mamba_weight(idx, reorder_in_proj, reorder_conv)
                )
            elif self._block_type(idx) == "attention":
                layer_weight.extend(self._create_mqa_weight(idx))
            else:
                layer_weight.extend(self._create_moe_weight(idx))
            all_layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _block_type(self, idx: int) -> str:
        hybrid_types = self.model_config.hybrid_attention_config.hybrid_attention_types
        if hybrid_types[idx] == HybridAttentionType.LINEAR:
            return "mamba"
        if idx in set(self.model_config.moe_layer_index):
            return "moe"
        return "attention"

    def _create_layer_norm_weight(self, idx: int) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(self._layer_prefix() + "norm.weight", identity)],
            )
        ]

    def _create_mamba_weight(
        self, layer_idx: int, reorder_in_proj, reorder_conv
    ) -> List[WeightModule]:
        config = self.model_config.linear_attention_config
        return [
            AtomicWeight(
                W.linear_attn_in_proj_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.in_proj.weight",
                        reorder_in_proj,
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.conv1d.weight",
                        reorder_conv,
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_conv1d_b,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.conv1d.bias", reorder_conv
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_dt_b,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.dt_bias", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo(self._layer_prefix() + "mixer.A_log", identity)],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_d,
                [CkptWeightInfo(self._layer_prefix() + "mixer.D", identity)],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_norm_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.norm.weight", identity
                    )
                ],
                identity,
            ),
            AtomicWeight(
                W.linear_attn_out_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.out_proj.weight", identity
                    )
                ],
                transpose,
            ),
        ]

    def _create_mqa_weight(self, layer_idx: int) -> List[WeightModule]:
        return [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.v_proj.weight", identity
                    ),
                ],
                merge_qkv_hf,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.o_proj.weight", identity
                    )
                ],
                transpose,
            ),
        ]

    def _create_moe_weight(self, layer_idx: int) -> List[WeightModule]:
        moe_config = MoeConfig(expert_num=self.model_config.expert_num)
        weights = [
            MoeWeight(
                sub_weights=[
                    MoeAtomicWeight(
                        W.moe_w1,
                        [
                            CkptWeightInfo(
                                self._layer_prefix()
                                + "mixer.experts.{expert_id}.up_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_w2,
                        [
                            CkptWeightInfo(
                                self._layer_prefix()
                                + "mixer.experts.{expert_id}.down_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=stack_,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.moe_gate,
                        [
                            CkptWeightInfo(
                                self._layer_prefix() + "mixer.gate.weight",
                                identity,
                            )
                        ],
                        transpose,
                        config=moe_config,
                    ),
                    MoeAtomicWeight(
                        W.e_score_correction_b,
                        [
                            CkptWeightInfo(
                                self._layer_prefix()
                                + "mixer.gate.e_score_correction_bias",
                                identity,
                            )
                        ],
                        config=moe_config,
                    ),
                ],
                config=moe_config,
            )
        ]
        if self.model_config.n_shared_experts > 0:
            weights.extend(self._create_shared_expert_weight(layer_idx))
        return weights

    def _create_shared_expert_weight(self, layer_idx: int) -> List[WeightModule]:
        # The shared expert is an ungated up/down MLP with relu2 activation,
        # sized independently from the routed experts.
        return [
            AtomicWeight(
                W.ffn_w1,
                [
                    CkptWeightInfo(
                        self._layer_prefix() + "mixer.shared_experts.up_proj.weight",
                        identity,
                    )
                ],
                transpose,
            ),
            AtomicWeight(
                W.ffn_w2,
                [
                    CkptWeightInfo(
                        self._layer_prefix()
                        + "mixer.shared_experts.down_proj.weight",
                        identity,
                    )
                ],
                transpose,
            ),
        ]
