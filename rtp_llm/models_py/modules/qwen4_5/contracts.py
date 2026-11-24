"""Shared contracts and typed data structures for Qwen 4.5-Flash parallel modules.

All parallel subagents import these types so integration into
``rtp_llm/models_py/model_desc/qwen4_5.py`` is strictly type-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class Qwen4_5ArchConfig:
    """Self-contained architectural specification matching Qwen4_5ForCausalLM."""

    vocab_size: int = 248320
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    intermediate_size: int = 12800
    first_k_dense_replace: int = 1
    rms_norm_eps: float = 1e-6
    bos_token_id: int = 248044
    eos_token_id: int = 248044
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 8192

    # Hybrid attention layout
    full_attention_interval: int = 4
    layer_types: Tuple[str, ...] = ()

    # Full attention (MQA-1 + output gate + KV short-conv + partial RoPE)
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    head_dim: int = 256
    attention_bias: bool = False
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10000.0
    share_kv_projection_with_rope: bool = False
    use_full_attn_kv_sconv: bool = True
    full_attn_kv_sconv_residual: bool = True
    full_attn_kv_sconv_use_activation: bool = False

    # Linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_num_key_heads: int = 32
    linear_key_head_dim: int = 64
    linear_num_value_heads: int = 128
    linear_value_head_dim: int = 64
    gdn_conv_residual: bool = False
    gdn_no_q_conv: bool = False
    norm_before_gate: bool = True
    output_gate_type: str = "sigmoid"

    # Output short-convs
    use_linear_attn_output_sconv: bool = True
    use_full_attn_output_sconv: bool = True
    use_ffn_output_sconv: bool = True
    output_sconv_use_activation: bool = False
    hybrid_sharing_consumer_layer_no_output_sconv: bool = False

    # Hyper-Connections (4-stream + hc_prev_mix)
    hc_count: int = 4
    hc_lowrank: int = 128
    hc_prev_mix: bool = True
    hc_prev_combine: bool = False
    hc_per_branch_norm: bool = True
    hc_rmsnorm_bias: bool = True

    # Latent MoE + Shared Expert + SiTU-GLU
    num_experts: int = 640
    num_experts_per_tok: int = 10
    routed_expert_hidden_size: int = 2560
    moe_intermediate_size: int = 2560
    shared_expert_intermediate_size: int = 1280
    shared_expert_gate: bool = True
    router_precision: str = "fp32_logits"
    router_score_function: str = "sigmoid"
    moe_router_pre_softmax: bool = True
    router_normalize_topk_scores: bool = True
    moe_router_enable_expert_bias: bool = True
    hidden_act: str = "situ_glu"
    hidden_act_gate_beta: float = 4.0
    hidden_act_up_beta: float = 25.0
    clip_mlp_output: Optional[float] = 1000.0
    clip_attn_output: Optional[float] = None

    # PLE (Per-Layer N-gram Embedding)
    ple_layer_ids: Tuple[int, ...] = (2,)  # 1-based layer indices
    ple_embed_dim: int = 5120
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 10
    ngram_vocab_size_base: int = 35000000
    make_ngram_vocab_size_divisible_by: int = 512
    split_ngram_parts: int = 128
    ple_quant_scheme: str = "fp8_pertensor"

    # YoCo cross-layer sharing (1-based string, e.g. "21->22,20->24,...")
    share_map: Optional[str] = None
    enable_yoco_decode: bool = True

    # Quantization
    quant_method: Optional[str] = "mxfp4"

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = tuple(
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            )
        else:
            self.layer_types = tuple(self.layer_types)
        self.ple_layer_ids = tuple(sorted(set(self.ple_layer_ids)))

    @classmethod
    def from_dict(cls, data: Dict) -> "Qwen4_5ArchConfig":
        rope = data.get("rope_parameters") or {}
        qcfg = data.get("quantization_config") or {}
        kwargs = {}
        for fld in cls.__dataclass_fields__:
            if fld in data:
                kwargs[fld] = data[fld]
        if "rope_theta" not in kwargs and "rope_theta" in rope:
            kwargs["rope_theta"] = float(rope["rope_theta"])
        if "partial_rotary_factor" not in kwargs and "partial_rotary_factor" in rope:
            kwargs["partial_rotary_factor"] = float(rope["partial_rotary_factor"])
        if "quant_method" not in kwargs and "quant_method" in qcfg:
            kwargs["quant_method"] = qcfg["quant_method"]
        if "ple_quant_scheme" not in kwargs and "ple_quant_scheme" in qcfg:
            kwargs["ple_quant_scheme"] = qcfg["ple_quant_scheme"]
        return cls(**kwargs)

    def get_attention_sharing(self) -> Dict[int, int]:
        """Return 0-based {consumer_layer_idx: source_layer_idx}."""
        if not self.share_map:
            return {}
        pairs: Dict[int, int] = {}
        for raw in self.share_map.split(","):
            item = raw.strip()
            if not item:
                continue
            src_s, dst_s = item.split("->")
            src, dst = int(src_s.strip()) - 1, int(dst_s.strip()) - 1
            if not (0 <= src < dst < self.num_hidden_layers):
                raise ValueError(f"Invalid share_map pair: {item}")
            if dst in pairs or self.layer_types[src] != self.layer_types[dst]:
                raise ValueError(f"Mismatched share_map pair: {item}")
            pairs[dst] = src
        if set(pairs.keys()) & set(pairs.values()):
            raise ValueError("A sharing source cannot also be a consumer")
        return pairs

    def has_output_sconv(self, site: str, *, is_consumer: bool) -> bool:
        flag = getattr(self, f"use_{site}", False)
        if is_consumer and self.hybrid_sharing_consumer_layer_no_output_sconv:
            return False
        return bool(flag)


@dataclass
class SharedGDNInputs:
    """Published by GDN YoCo source layer (layer 20) for consumer layers."""

    key: torch.Tensor               # [B, T, H_k, D_k] post-conv, pre-L2
    value: torch.Tensor             # [B, T, H_v, D_v] post-conv
    g: torch.Tensor                 # [B, T, H_v] log-decay (negative FP32)
    beta: torch.Tensor              # [B, T, H_v] update gate (0..1)
    ssm_state: Optional[torch.Tensor] = None  # [B, H_v, D_k, D_v] FP32 (post-update, for YoCo decode)


@dataclass
class FullAttnKVCache:
    """Per-layer KV cache for MQA-1 Full Attention."""

    k_cache: torch.Tensor           # [B, max_seq, H_kv, D]
    v_cache: torch.Tensor           # [B, max_seq, H_kv, D]
    seq_len: int = 0                # tokens currently cached


@dataclass
class GDNLayerState:
    """Per-layer recurrent + conv state for Gated DeltaNet."""

    conv_state: torch.Tensor        # [B, C_conv, K-1]
    ssm_state: Optional[torch.Tensor] = None  # [B, H_v, D_k, D_v] FP32 (None for YoCo-decode-only consumer)


@dataclass
class ShortConvState:
    """Per-site causal depthwise conv1d rolling history."""

    conv_state: torch.Tensor        # [B, C, K-1]


@dataclass
class PLELayerState:
    """Per-request PLE n-gram token history & short-conv state."""

    ngram_context: torch.Tensor     # [B, ngram_size - 1] int64 (last tokens, EOS-padded)
    conv_state: torch.Tensor        # [B, hc_count * hidden_size, ple_conv_kernel_size - 1]


@dataclass
class Qwen4_5RuntimeState:
    """Container holding all mutable inference caches for a batch."""

    batch_size: int
    max_seq_len: int
    full_attn_kv: Dict[int, FullAttnKVCache] = field(default_factory=dict)
    k_sconv: Dict[int, ShortConvState] = field(default_factory=dict)
    v_sconv: Dict[int, ShortConvState] = field(default_factory=dict)
    gdn: Dict[int, GDNLayerState] = field(default_factory=dict)
    attn_output_sconv: Dict[int, ShortConvState] = field(default_factory=dict)
    ffn_output_sconv: Dict[int, ShortConvState] = field(default_factory=dict)
    ple: Dict[int, PLELayerState] = field(default_factory=dict)
    seq_len: int = 0                # global cached sequence length
