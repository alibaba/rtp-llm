from typing import Any, Dict, List

from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
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
    transpose,
)


def plus_one(ts):
    """Gemma RMSNorm: stored weight w, actual scale = w + 1."""
    return ts[0] + 1


class Gemma4WeightInfo(ModelDeployWeightInfo):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__(*args, **kwargs)
        # Gemma4ForConditionalGeneration nests text model under language_model
        self.prefix = "model.language_model."

    def _get_weight_info(self):
        weights: List[WeightModule] = []

        # Embedding (shared with lm_head via tie_word_embeddings)
        weights.append(
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
            )
        )

        # Output uses embedding weight (tie_word_embeddings=True)
        if self.model_config.tie_word_embeddings:
            weights.append(
                AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo(self.prefix + "embed_tokens.weight", identity)],
                )
            )
        else:
            weights.append(
                AtomicWeight(
                    W.lm_head,
                    [CkptWeightInfo("lm_head.weight", identity)],
                )
            )

        # Final layer norm (with Gemma +1 shift)
        weights.append(
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(self.prefix + "norm.weight", plus_one)],
            )
        )

        # Per-layer weights
        all_layer_weights: List[List[WeightModule]] = []
        hybrid_types = getattr(
            self.model_config.hybrid_attention_config,
            "hybrid_attention_types",
            [],
        )

        for idx in range(self._num_layers):
            layer_weight: List[WeightModule] = []
            layer_weight.extend(self._create_layer_norm_weight())
            layer_weight.extend(self._create_attention_weight(idx, hybrid_types))
            layer_weight.extend(self._create_ffn_weight())
            all_layer_weights.append(layer_weight)

        return ModelWeightInfo(layer_weights=all_layer_weights, weights=weights)

    def _create_layer_norm_weight(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.input_layernorm.weight",
                        plus_one,
                    )
                ],
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        self.prefix + "layers.{i}.post_attention_layernorm.weight",
                        plus_one,
                    )
                ],
            ),
            # Note: pre_feedforward_layernorm and post_feedforward_layernorm
            # exist in Gemma 4 but are handled differently in the model_desc
            # forward pass. Skip loading them as separate weight slots for now.
        ]

    def _create_attention_weight(
        self, layer_idx: int, hybrid_types: list
    ) -> List[WeightModule]:
        # Determine if this is a sliding window or global attention layer
        is_sliding = True
        if hybrid_types and layer_idx < len(hybrid_types):
            is_sliding = hybrid_types[layer_idx] == HybridAttentionType.SLIDING_WINDOW

        # For Gemma4 with attention_k_eq_v=True:
        # checkpoint has q_proj.weight and kv_proj.weight (shared K=V)
        # We duplicate kv_proj for both K and V slots
        k_eq_v = getattr(self.model_config, "gemma4_attention_k_eq_v", False)

        # Gemma4 checkpoint always has separate q_proj, k_proj, v_proj
        # (even with attention_k_eq_v=True, convert script outputs separate keys)
        # For global layers with k_eq_v: k_proj acts as both K and V
        if k_eq_v and not is_sliding:
            # Global attention layers: k_proj.weight serves as both K and V
            qkv_ckpt = [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.q_proj.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.k_proj.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.k_proj.weight",
                    identity,
                ),
            ]
        else:
            qkv_ckpt = [
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.q_proj.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.k_proj.weight",
                    identity,
                ),
                CkptWeightInfo(
                    self.prefix + "layers.{i}.self_attn.v_proj.weight",
                    identity,
                ),
            ]

        # Select the correct attention config for this layer type
        # The attn_config used for weight loading determines how merge_qkv_hf reshapes
        attn_config = self._get_attn_config_for_layer(is_sliding)

        weights = [
            AttnAtomicWeight(
                W.attn_qkv_w,
                qkv_ckpt,
                process_fun=merge_qkv_hf,
                config=attn_config,
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
                config=attn_config,
            ),
        ]

        return weights

    def _get_attn_config_for_layer(self, is_sliding: bool):
        """Get the attention config for weight loading, with correct KV dims per layer type."""
        # For sliding layers: use the default attn_config (kv_head_num=16, head_dim=256)
        # For global layers: different dimensions (kv_head_num=4, head_dim=512)
        if is_sliding:
            return self.attn_config
        else:
            global_config = getattr(self.model_config, "gemma4_global_attn_config", None)
            if global_config:
                from rtp_llm.ops import AttentionConfigs
                config = AttentionConfigs()
                config.head_num = self.model_config.attn_config.head_num
                config.kv_head_num = global_config["kv_head_num"]
                config.size_per_head = global_config["head_dim"]
                config.tokens_per_block = self.model_config.attn_config.tokens_per_block
                return config
            return self.attn_config

    def _create_ffn_weight(self) -> List[WeightModule]:
        ffn_config = FfnConfig(
            is_gated_activation=True,  # Gemma4 uses gated-gelu
            align_size=self._align_size,
        )
        return [
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [
                            CkptWeightInfo(
                                self.prefix + "layers.{i}.mlp.gate_proj.weight",
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
                                self.prefix + "layers.{i}.mlp.up_proj.weight",
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
                                self.prefix + "layers.{i}.mlp.down_proj.weight",
                                identity,
                            )
                        ],
                        process_fun=transpose,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
        ]
