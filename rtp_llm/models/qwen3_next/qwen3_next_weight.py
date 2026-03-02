import functools
from typing import Any, Dict, List

import torch

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    MoeAtomicWeight,
    MoeConfig,
    MoeWithSharedWeight,
    SharedMoeConfig,
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
from rtp_llm.ops import HybridAttentionType, LinearAttentionConfig
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    stack_,
    stack_moe_w1,
    transpose,
)


def split_q_gate(ts: List[torch.Tensor], head_num: int, head_dim: int, part: int):
    """Split q_gate tensor into q or gate part.

    This function is used for both weight and scale (if fp8_per_block).
    For weight, new_head_dim is head_dim; for scale, it's head_dim / block_size.
    """
    dim0, dim1 = ts[0].shape
    assert (
        dim0 % (head_num * 2) == 0
    ), f"dim0 % (head_num * 2) != 0, dim0: {dim0}, head_num: {head_num}, head_dim: {head_dim}, dim1: {dim1}"
    new_head_dim = dim0 // (head_num * 2)
    t = ts[0].reshape(head_num, 2, new_head_dim, dim1)
    if part == 0:
        return t[:, 0, :, :].reshape(-1, dim1)
    else:
        return t[:, 1, :, :].reshape(-1, dim1)


def plus_one(ts: List[torch.Tensor]):
    """Add one to the tensor. Qwen3Next uses gemma_rms_norm."""
    return ts[0] + 1


def reorder_ba(ts: List[torch.Tensor], linear_attention_config: LinearAttentionConfig):
    """Reorder ba weight tensor.

    Transform from shape [head_num_k, 2 + 2, hidden_size]
    to [head_num_k * 2 + head_num_k * 2, hidden_size].
    """
    t = ts[0]
    hidden_size = t.shape[-1]
    head_num_k = linear_attention_config.linear_num_key_heads
    head_num_v = linear_attention_config.linear_num_value_heads
    group_v = head_num_v // head_num_k
    t = t.reshape(head_num_k, group_v * 2, t.shape[-1])
    b, a = t.split([group_v, group_v], dim=1)
    return torch.cat([b.reshape(-1, hidden_size), a.reshape(-1, hidden_size)], dim=0)


def reorder_qkvz(
    ts: List[torch.Tensor], linear_attention_config: LinearAttentionConfig
):
    """Reorder qkvz weight tensor.

    Transform from shape [token, head_num_k, dim_q, dim_k, dim_v * group_v, dim_v * group_v]
    to [token, head_num_k * dim_q, head_num_k * dim_k, head_k * group_v * dim_v, head_k * group_v * dim_v].

    Args:
        ts: List containing weight tensor
        linear_attention_config: Linear attention configuration (C++ LinearAttentionConfig type)

    Returns:
        Reordered weight tensor
    """
    t = ts[0]

    head_num_k = linear_attention_config.linear_num_key_heads
    head_num_v = linear_attention_config.linear_num_value_heads
    head_k_dim = linear_attention_config.linear_key_head_dim
    head_v_dim = linear_attention_config.linear_value_head_dim
    group_v = head_num_v // head_num_k

    dim0, dim1 = t.shape
    qkvz_size = (
        head_num_k * head_k_dim
        + head_num_k * head_k_dim
        + head_num_v * head_v_dim
        + head_num_v * head_v_dim
    )
    BLOCK_SIZE = 128
    if dim0 == qkvz_size:
        t = t.reshape(
            head_num_k,
            head_k_dim + head_k_dim + head_v_dim * group_v + head_v_dim * group_v,
            dim1,
        )
        q, k, v, z = torch.split(
            t,
            [head_k_dim, head_k_dim, head_v_dim * group_v, head_v_dim * group_v],
            dim=1,
        )
        q = q.reshape(-1, dim1)
        k = k.reshape(-1, dim1)
        v = v.reshape(-1, dim1)
        z = z.reshape(-1, dim1)
        return torch.cat([q, k, v, z], dim=0)
    elif dim0 == qkvz_size // BLOCK_SIZE:
        t = t.reshape(
            head_num_k,
            (head_k_dim + head_k_dim + head_v_dim * group_v + head_v_dim * group_v)
            // BLOCK_SIZE,
            dim1,
        )
        q, k, v, z = torch.split(
            t,
            [
                head_k_dim // BLOCK_SIZE,
                head_k_dim // BLOCK_SIZE,
                head_v_dim * group_v // BLOCK_SIZE,
                head_v_dim * group_v // BLOCK_SIZE,
            ],
            dim=1,
        )
        q = q.reshape(-1, dim1)
        k = k.reshape(-1, dim1)
        v = v.reshape(-1, dim1)
        z = z.reshape(-1, dim1)
        return torch.cat([q, k, v, z], dim=0)
    else:
        raise ValueError(
            f"Invalid input shape 0 for scale / weight: {t.shape}, expected: {qkvz_size} or {qkvz_size // BLOCK_SIZE}"
        )


def merge_qkvz_transpose_reorder(
    ts: List[torch.Tensor], linear_attention_config: LinearAttentionConfig
):
    """Merge and reorder qkv and z tensors for Qwen3.5Moe.

    Merges qkv (shape: [token, head_num_k * dim_q + head_num_k * dim_k + head_num_v * dim_v])
    and z (shape: [token, head_num_v, dim_v * group_v]) into a single transposed tensor.
    """
    qkv = ts[0]
    z = ts[1]
    return torch.cat([qkv, z], dim=0).T


def merge_ba_transpose_reorder(
    ts: List[torch.Tensor], linear_attention_config: LinearAttentionConfig
):
    """Merge and reorder b and a tensors, then transpose."""
    b = ts[0]
    a = ts[1]
    return torch.cat([b, a], dim=0).T


def transpose_gate_up(ts: List[torch.Tensor]):
    """Transform from [expert, hidden, gate + up] to [expert, hidden, up + gate]."""
    assert (
        len(ts[0].shape) == 3
    ), f"Expected ts[0] shape to be 3, but got {len(ts[0].shape)}"

    dim1 = ts[0].shape[1]
    assert dim1 % 2 == 0, f"Expected dim2 to be even, but got {dim1}"

    half_dim = dim1 // 2
    gate = ts[0][:, :half_dim, :]
    up = ts[0][:, half_dim:, :]

    return torch.cat([up, gate], dim=1)


class Qwen3NextBaseWeight(ModelDeployWeightInfo):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "model."

    def _get_weight_info(self):
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm.weight", plus_one)],
            ),
        ]
        all_layer_weights: List[List[WeightModule]] = []
        for idx in range(self._num_layers):
            layer_weight: List[WeightModule] = []
            layer_weight.extend(self._create_layer_norm_weight())
            if (
                self.model_config.hybrid_attention_config.hybrid_attention_types[idx]
                == HybridAttentionType.LINEAR
            ):
                layer_weight.extend(self._create_linear_attention_weight())
            else:
                layer_weight.extend(self._create_mqa_weight())
            layer_weight.extend(self._create_ffn_weight())
            all_layer_weights.append(layer_weight)
        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _create_layer_norm_weight(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_attention_layernorm.weight",
                        plus_one,
                    ),
                ],
            ),
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.input_layernorm.weight",
                        plus_one,
                    )
                ],
            ),
        ]

    def _create_mqa_weight(self) -> List[WeightModule]:
        return [
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.q_proj.weight",
                        functools.partial(
                            split_q_gate,
                            head_num=self.model_config.attn_config.head_num,
                            head_dim=self.model_config.attn_config.size_per_head,
                            part=0,
                        ),
                    ),
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.k_proj.weight",
                        identity,
                    ),
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.v_proj.weight",
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
                        self.prefix + "layers.{i}.self_attn.o_proj.weight",
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
                        self.prefix + "layers.{i}.self_attn.q_proj.weight",
                        functools.partial(
                            split_q_gate,
                            head_num=self.model_config.attn_config.head_num,
                            head_dim=self.model_config.attn_config.size_per_head,
                            part=1,
                        ),
                    )
                ],
                process_fun=transpose,
                config=self.attn_config,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.q_norm.weight", plus_one
                    )
                ],
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.k_norm.weight", plus_one
                    )
                ],
            ),
        ]

    def _create_ffn_weight(self) -> List[WeightModule]:
        moe_config = self._get_moe_config()
        shared_moe_config = self._get_shared_moe_config()
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
        )
        sub_weights = self._create_ffn_common_weights(moe_config, ffn_config)
        sub_weights.extend(self._create_moe_expert_weights(moe_config))

        return [MoeWithSharedWeight(sub_weights, shared_moe_config)]

    def _get_moe_config(self) -> MoeConfig:
        return MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
        )

    def _get_shared_moe_config(self) -> SharedMoeConfig:
        return SharedMoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
        )

    def _create_ffn_common_weights(
        self, moe_config: MoeConfig, ffn_config: FfnConfig
    ) -> List[WeightModule]:
        return [
            MoeAtomicWeight(
                W.moe_gate,
                [CkptWeightInfo(self.prefix + "layers.{i}.mlp.gate.weight", identity)],
                process_fun=transpose,
                config=moe_config,
            ),
            FfnAtomicWeight(
                W.ffn_w1,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.mlp.shared_expert.gate_proj.weight",
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
                        self.prefix + "layers.{i}.mlp.shared_expert.down_proj.weight",
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
                        self.prefix + "layers.{i}.mlp.shared_expert.up_proj.weight",
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
                        self.prefix + "layers.{i}.mlp.shared_expert_gate.weight",
                        identity,
                    )
                ],
                process_fun=transpose,
            ),
        ]

    def _create_moe_expert_weights(self, moe_config: MoeConfig) -> List[WeightModule]:
        """Create MoE expert weights in split format (default implementation)."""
        return [
            MoeAtomicWeight(
                W.moe_w2,
                [
                    CkptWeightInfo(
                        self.prefix
                        + "layers.{i}.mlp.experts.{expert_id}.down_proj.weight",
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
                        self.prefix
                        + "layers.{i}.mlp.experts.{expert_id}.up_proj.weight",
                        identity,
                    )
                ]
                + [
                    CkptWeightInfo(
                        self.prefix
                        + "layers.{i}.mlp.experts.{expert_id}.gate_proj.weight",
                        identity,
                    )
                ],
                process_fun=stack_moe_w1,
                config=moe_config,
            ),
        ]

    def _create_linear_attention_weight(self) -> List[WeightModule]:
        weights = []
        weights.append(self._create_linear_attn_qkvz_weight())
        weights.append(self._create_linear_attn_ba_weight())
        weights.extend(self._create_linear_attn_common_weights())

        return weights

    def _create_linear_attn_qkvz_weight(self) -> LinearAttnAtomicWeight:
        raise NotImplementedError(
            "Subclass must implement _create_linear_attn_qkvz_weight"
        )

    def _create_linear_attn_ba_weight(self) -> LinearAttnAtomicWeight:
        raise NotImplementedError(
            "Subclass must implement _create_linear_attn_ba_weight"
        )

    def _create_linear_attn_common_weights(self) -> List[LinearAttnAtomicWeight]:
        return [
            LinearAttnAtomicWeight(
                W.linear_attn_norm_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.linear_attn.norm.weight", identity
                    )
                ],
                identity,
                LinearAttnConfig(self.model_config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_dt_b,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.linear_attn.dt_bias", identity
                    )
                ],
                identity,
                LinearAttnConfig(self.model_config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.linear_attn.conv1d.weight", identity
                    )
                ],
                identity,
                LinearAttnConfig(self.model_config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_alog,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.linear_attn.A_log", identity
                    )
                ],
                identity,
                LinearAttnConfig(self.model_config.linear_attention_config),
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_out_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.linear_attn.out_proj.weight", identity
                    )
                ],
                transpose,
                LinearAttnConfig(self.model_config.linear_attention_config),
            ),
        ]


class Qwen3NextWeight(Qwen3NextBaseWeight):
    """Qwen3Next weight loading (model. prefix, merged weight format, split expert)."""

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "model."

    def _create_linear_attn_qkvz_weight(self) -> LinearAttnAtomicWeight:
        """Merged format: single file in_proj_qkvz.weight."""
        return LinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_qkvz.weight",
                    functools.partial(
                        reorder_qkvz,
                        linear_attention_config=self.model_config.linear_attention_config,
                    ),
                )
            ],
            transpose,
            LinearAttnConfig(self.model_config.linear_attention_config),
        )

    def _create_linear_attn_ba_weight(self) -> LinearAttnAtomicWeight:
        """Merged format: single file in_proj_ba.weight."""
        return LinearAttnAtomicWeight(
            W.linear_attn_ba_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_ba.weight",
                    functools.partial(
                        reorder_ba,
                        linear_attention_config=self.model_config.linear_attention_config,
                    ),
                )
            ],
            transpose,
            LinearAttnConfig(self.model_config.linear_attention_config),
        )


class Qwen35MoeWeight(Qwen3NextBaseWeight):
    """Qwen3.5 MoE weight loading (model.language_model. prefix, separate weight format, stackwd support)."""

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = "model.language_model."
        self._use_stack_weight = False

    def _process_meta(self, meta_dict: Any, weight_keys: List[str]):
        """Detect whether stackwd format is used.

        Qwen3.5 bf16 uses stackwd moe weight while fp8 uses splited moe weights.
        """
        if self._contains(weight_keys, "layers.0.mlp.experts.gate_up_proj"):
            self._use_stack_weight = True

    def _get_moe_config(self) -> MoeConfig:
        """Get MoeConfig with weight_stack support."""
        return MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            weight_stack=self._use_stack_weight,
        )

    def _get_shared_moe_config(self) -> SharedMoeConfig:
        """Get SharedMoeConfig with weight_stack support."""
        return SharedMoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
            weight_stack=self._use_stack_weight,
        )

    def _create_moe_expert_weights(self, moe_config: MoeConfig) -> List[WeightModule]:
        """Create MoE expert weights in stackwd or split format."""
        if self._use_stack_weight:
            return [
                MoeAtomicWeight(
                    W.moe_w2,
                    [
                        CkptWeightInfo(
                            self.prefix + "layers.{i}.mlp.experts.down_proj",
                            identity,
                        )
                    ],
                    process_fun=identity,
                    config=moe_config,
                ),
                MoeAtomicWeight(
                    W.moe_w1,
                    [
                        CkptWeightInfo(
                            self.prefix + "layers.{i}.mlp.experts.gate_up_proj",
                            identity,
                        )
                    ],
                    process_fun=transpose_gate_up,
                    config=moe_config,
                ),
            ]
        else:
            return super()._create_moe_expert_weights(moe_config)

    def _create_linear_attn_qkvz_weight(self) -> LinearAttnAtomicWeight:
        """Separate format: two files in_proj_qkv.weight + in_proj_z.weight."""
        return LinearAttnAtomicWeight(
            W.linear_attn_qkvz_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_qkv.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_z.weight",
                    identity,
                ),
            ],
            functools.partial(
                merge_qkvz_transpose_reorder,
                linear_attention_config=self.model_config.linear_attention_config,
            ),
            LinearAttnConfig(self.model_config.linear_attention_config),
        )

    def _create_linear_attn_ba_weight(self) -> LinearAttnAtomicWeight:
        """Separate format: two files in_proj_b.weight + in_proj_a.weight."""
        return LinearAttnAtomicWeight(
            W.linear_attn_ba_w,
            [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_b.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.linear_attn.in_proj_a.weight",
                    identity,
                ),
            ],
            functools.partial(
                merge_ba_transpose_reorder,
                linear_attention_config=self.model_config.linear_attention_config,
            ),
            LinearAttnConfig(self.model_config.linear_attention_config),
        )
