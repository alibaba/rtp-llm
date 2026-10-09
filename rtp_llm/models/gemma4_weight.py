from typing import Any, Dict, List

import torch
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import (
    FfnAtomicWeight,
    FfnConfig,
    FfnWeight,
    MoeAtomicWeight,
    MoeConfig,
    MoeWeight,
)
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
    transpose_stack_moe_w1,
)

# Gemma4 (VLM layout) nests the language model under "model.language_model.";
# text-only exports may use "model." - the prefix is detected per checkpoint.
GEMMA4_DEFAULT_CKPT_PREFIX = "model.language_model."


def merge_qkv_keq_v(ts: List[torch.Tensor]) -> torch.Tensor:
    """Merge q/k projections of a Gemma4 full-attention layer (attention_k_eq_v).

    Full-attention layers have no v_proj: the value path reuses the (normed) key
    projection. q: [head_num * global_head_dim, hidden],
    k: [num_global_kv_heads * global_head_dim, hidden] ->
    cat([q.T, k.T, k.T], dim=1) -> [hidden, q_out + 2 * k_out].
    """
    q, k = ts
    return torch.concat([q.T, k.T, k.T], dim=1).contiguous()


def scale_reshape(ts: List[torch.Tensor]) -> torch.Tensor:
    """Undo the loader's automatic unsqueeze(-1) for 1-D "scale" tensors.

    weight_module.py appends a trailing dim to every 1-D checkpoint tensor whose
    name contains "scale" (router.scale, router.per_expert_scale); reshape(-1)
    restores the original flat shape.
    """
    return ts[0].reshape(-1)


class Gemma4WeightInfo(ModelDeployWeightInfo):
    """Gemma4 weight loading.

    All RMSNorm gammas are loaded as-is (no +1 shift, unlike Gemma1-3).
    lm_head is tied to the embedding and resolved by the loader's
    _fix_tie_lm_head fallback. Vision-tower / embed_vision keys are simply not
    declared and therefore ignored.
    """

    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        self.prefix = GEMMA4_DEFAULT_CKPT_PREFIX
        model_config = kwargs.get("model_config") or (args[0] if args else None)
        if model_config is None:
            raise ValueError("Gemma4WeightInfo requires model_config")
        stored = model_config.mm_related_params.config
        try:
            self._full_kv_head_num = int(stored["full_layer_kv_head_num"])
            self._full_size_per_head = int(stored["full_layer_size_per_head"])
        except KeyError as error:
            raise ValueError(
                f"Gemma4WeightInfo missing instance geometry: {error.args[0]}"
            ) from error

    def _process_meta(self, meta_dict: Any, weight_keys):
        # Detect the language-model prefix from the layer-0 input_layernorm.
        # Vision-tower layers also contain "layers.0.input_layernorm.weight",
        # so keys under vision_tower are excluded explicitly.
        suffix = "layers.0.input_layernorm.weight"
        for key in weight_keys:
            if key.endswith(suffix) and "vision_tower" not in key:
                self.prefix = key[: -len(suffix)]
                break
        else:
            raise ValueError(
                f"Gemma4WeightInfo: cannot determine language-model prefix, no "
                f"non-vision key ending with {suffix!r} in {len(weight_keys)} "
                f"ckpt keys"
            )

    def _is_full_attention_layer(self, layer_id: int) -> bool:
        # layer_types: full_attention -> HybridAttentionType.NONE,
        # sliding_attention -> HybridAttentionType.SLIDING_WINDOW
        return (
            self.model_config.hybrid_attention_config.hybrid_attention_types[layer_id]
            == HybridAttentionType.NONE
        )

    def _full_layer_attn_config(self) -> AttnConfig:
        return AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._full_size_per_head,
            head_num=self._head_num,
            head_num_kv=self._full_kv_head_num,
        )

    def _get_weight_info(self) -> ModelWeightInfo:
        weights: List[WeightModule] = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
            ),
            # tied to the embedding; absent lm_head.weight falls back to the
            # embedding tensor via ModelDeployWeightInfo._fix_tie_lm_head.
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("lm_head.weight", identity)],
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm.weight", identity)],
            ),
        ]
        layer_weights: List[List[WeightModule]] = []
        for layer_id in range(self._num_layers):
            layer_weights.append(self._get_layer_weight_info(layer_id))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights)

    def _get_layer_weight_info(self, layer_id: int) -> List[WeightModule]:
        layer_weights: List[WeightModule] = []
        layer_weights.extend(self._create_layer_norm_weights())
        layer_weights.extend(self._create_attention_weights(layer_id))
        layer_weights.extend(self._create_ffn_weights())
        layer_weights.extend(self._create_router_scale_weights())
        return layer_weights

    def _create_layer_norm_weights(self) -> List[WeightModule]:
        # Sandwich structure around attention and the dense-MLP + MoE block:
        # every gamma is consumed directly (NO +1).
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.input_layernorm.weight", identity
                    )
                ],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_attention_layernorm.weight",
                        identity,
                    )
                ],
            ),
            AtomicWeight(
                W.pre_ffn_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.pre_feedforward_layernorm.weight",
                        identity,
                    )
                ],
            ),
            AtomicWeight(
                W.pre_ffn2_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.pre_feedforward_layernorm_2.weight",
                        identity,
                    )
                ],
            ),
            AtomicWeight(
                W.post_ffn_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_feedforward_layernorm.weight",
                        identity,
                    )
                ],
            ),
            AtomicWeight(
                W.post_ffn1_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_feedforward_layernorm_1.weight",
                        identity,
                    )
                ],
            ),
            AtomicWeight(
                W.post_ffn2_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_feedforward_layernorm_2.weight",
                        identity,
                    )
                ],
            ),
            # [1] scalar kept in its checkpoint shape; consumers broadcast it.
            AtomicWeight(
                W.layer_scalar,
                [CkptWeightInfo(self.prefix + "layers.{i}.layer_scalar", identity)],
            ),
        ]

    def _create_attention_weights(self, layer_id: int) -> List[WeightModule]:
        if self._is_full_attention_layer(layer_id):
            # 16Q x 512, 2KV x 512, no v_proj (K doubles as V).
            qkv = AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.k_proj.weight", identity
                    ),
                ],
                process_fun=merge_qkv_keq_v,
                config=self._full_layer_attn_config(),
            )
        else:
            # 16Q x 256, 8KV x 256 (global sliding geometry).
            qkv = AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                process_fun=merge_qkv_hf,
                config=self.attn_config,
            )
        # Gemma4 QK-norm gammas are consumed as-is (NO +1, unlike qwen3_next).
        return [
            qkv,
            AttnAtomicWeight(
                W.attn_o_w,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.o_proj.weight", identity
                    )
                ],
                process_fun=transpose,
                config=qkv.config,
            ),
            AtomicWeight(
                W.q_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.q_norm.weight", identity
                    )
                ],
            ),
            AtomicWeight(
                W.k_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.self_attn.k_norm.weight", identity
                    )
                ],
            ),
        ]

    def _create_ffn_weights(self) -> List[WeightModule]:
        ffn_config = FfnConfig(
            is_gated_activation=self._is_gated_activation,
            align_size=self._align_size,
        )
        # Dense MLP (gelu_tanh) running in parallel with the routed experts.
        dense = FfnWeight(
            sub_weights=[
                FfnAtomicWeight(
                    W.ffn_w1,
                    [
                        CkptWeightInfo(
                            self.prefix + "layers.{i}.mlp.gate_proj.weight", identity
                        )
                    ],
                    process_fun=transpose,
                    config=ffn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_w3,
                    [
                        CkptWeightInfo(
                            self.prefix + "layers.{i}.mlp.up_proj.weight", identity
                        )
                    ],
                    process_fun=transpose,
                    config=ffn_config,
                ),
                FfnAtomicWeight(
                    W.ffn_w2,
                    [
                        CkptWeightInfo(
                            self.prefix + "layers.{i}.mlp.down_proj.weight", identity
                        )
                    ],
                    process_fun=transpose,
                    config=ffn_config,
                ),
            ],
            config=ffn_config,
        )
        # Stacked routed experts: gate_up_proj [E, 2 * moe_inter, H] with gate in
        # the first half; transpose_stack_moe_w1 swaps it to the up|gate layout
        # expected by W.moe_w1. down_proj [E, H, moe_inter] loads via stack_.
        moe_config = MoeConfig(
            expert_num=self.expert_num_,
            align_size=self._align_size,
        )
        # router logits projection: [expert_num, hidden] -> [hidden, expert_num]
        moe_gate = MoeAtomicWeight(
            W.moe_gate,
            [CkptWeightInfo(self.prefix + "layers.{i}.router.proj.weight", identity)],
            process_fun=transpose,
            config=moe_config,
        )
        experts = MoeWeight(
            sub_weights=[
                moe_gate,
                MoeAtomicWeight(
                    W.moe_w2,
                    [CkptWeightInfo(self.prefix + "layers.{i}.experts.down_proj")],
                    process_fun=stack_,
                    config=moe_config,
                    stacked_ckpt_keys=True,
                    enable_pure_tp_preshard=True,
                ),
                MoeAtomicWeight(
                    W.moe_w1,
                    [CkptWeightInfo(self.prefix + "layers.{i}.experts.gate_up_proj")],
                    process_fun=stack_,
                    config=moe_config,
                    stacked_ckpt_keys=True,
                    enable_pure_tp_preshard=True,
                ),
            ],
            config=moe_config,
        )
        return [dense, experts]

    def _create_router_scale_weights(self) -> List[WeightModule]:
        # router: RMSNorm_noscale(x) * router.scale[hidden] * hidden**-0.5
        #        -> proj -> softmax -> top-k -> renorm -> * per_expert_scale[E]
        # The "scale"-named 1-D tensors get unsqueezed(-1) by the loader, so
        # scale_reshape restores the flat shapes ([hidden] and [E]).
        return [
            AtomicWeight(
                W.moe_router_scale,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.router.scale", scale_reshape
                    )
                ],
            ),
            AtomicWeight(
                W.moe_router_expert_scale,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.router.per_expert_scale",
                        scale_reshape,
                    )
                ],
            ),
        ]
