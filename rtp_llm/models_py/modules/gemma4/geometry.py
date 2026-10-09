"""Per-instance Gemma4 attention geometry consumed by model assembly."""

from dataclasses import dataclass

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import HybridAttentionType, ParallelismConfig

GEMMA4_TAG_SWA = "swa"
GEMMA4_TAG_FULL = "full"


@dataclass(frozen=True)
class Gemma4LayerGeometry:
    """Per-layer attention geometry + rope schedule."""

    tag: str
    head_num: int
    kv_head_num: int
    head_dim: int
    rope_theta: float
    rope_partial_rotary_factor: float
    sliding_window: int  # 0 disables the window bound
    k_equals_v: bool = False


def _layer_type_at(config: ModelConfig, layer_idx: int) -> HybridAttentionType:
    types = config.hybrid_attention_config.hybrid_attention_types
    if layer_idx >= len(types):
        raise IndexError(
            f"Gemma4 layer {layer_idx} has no attention type; configured={len(types)}"
        )
    return types[layer_idx]


def build_gemma4_layer_geometry(
    config: ModelConfig, parallelism_config: ParallelismConfig, layer_idx: int
) -> Gemma4LayerGeometry:
    attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
    gemma4_config = config.mm_related_params.config
    layer_type = _layer_type_at(config, layer_idx)
    if layer_type == HybridAttentionType.LINEAR:
        raise ValueError("Gemma4 does not support LINEAR hybrid attention layers")
    if layer_type == HybridAttentionType.NONE:
        attn_tp = parallelism_config.get_attn_tp_size()
        return Gemma4LayerGeometry(
            tag=GEMMA4_TAG_FULL,
            head_num=attn_configs.head_num,
            kv_head_num=max(
                1, int(gemma4_config["full_layer_kv_head_num"]) // max(attn_tp, 1)
            ),
            head_dim=int(gemma4_config["full_layer_size_per_head"]),
            rope_theta=float(gemma4_config["full_layer_rope_theta"]),
            rope_partial_rotary_factor=float(
                gemma4_config["full_layer_partial_rotary_factor"]
            ),
            sliding_window=0,
            k_equals_v=True,
        )
    rope_base = float(attn_configs.rope_config.base)
    if rope_base <= 0:
        raise ValueError("Gemma4 sliding RoPE base must be positive")
    return Gemma4LayerGeometry(
        tag=GEMMA4_TAG_SWA,
        head_num=attn_configs.head_num,
        kv_head_num=attn_configs.kv_head_num,
        head_dim=attn_configs.size_per_head,
        rope_theta=rope_base,
        rope_partial_rotary_factor=1.0,
        sliding_window=int(config.attn_config.sliding_window),
    )
