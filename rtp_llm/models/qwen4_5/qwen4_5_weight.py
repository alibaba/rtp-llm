"""Qwen 4.5-Flash (1109-bf16-re-mxfp4-ple-fp8) weight layout, descriptors, and validation.

Self-contained weight keys (``QWEN4_5_W``), exact 151,038-tensor checkpoint
specification generator, safetensors index validator, and RTP-LLM
``ModelDeployWeightInfo``-compatible ``Qwen4_5Weight`` class with graceful
pure-Python import when compiled C++ ops are unavailable.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from rtp_llm.config.qwen4_5_config import (
    MXFP4_GROUP_SIZE,
    compute_ple_ngram_shard_shape,
    validate_qwen4_5_arch_config,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

try:
    from rtp_llm.model_loader.load_config import LoadConfig
    from rtp_llm.model_loader.model_weight_info import (
        ModelDeployWeightInfo as _BaseWeightInfo,
        ModelWeightInfo,
    )
    from rtp_llm.model_loader.tensor_source import TensorSource
    from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
    from rtp_llm.utils.model_weight import CkptWeightInfo, identity

    _HAS_RTP_WEIGHT_LOADER = True
except Exception:  # pragma: no cover - pure-Python fallback without C++ ops
    _BaseWeightInfo = object  # type: ignore[assignment,misc]
    ModelWeightInfo = Any  # type: ignore[assignment,misc]
    AtomicWeight = Any  # type: ignore[assignment,misc]
    WeightModule = Any  # type: ignore[assignment,misc]
    CkptWeightInfo = Any  # type: ignore[assignment,misc]
    LoadConfig = Any  # type: ignore[assignment,misc]
    TensorSource = Any  # type: ignore[assignment,misc]
    identity = lambda ts: ts[0]  # noqa: E731
    _HAS_RTP_WEIGHT_LOADER = False


_PREFIX_ANCHOR = "layers.0.attn_hyper_connection.hc_norm.weight"

DTYPE_NUM_BYTES: Dict[str, int] = {
    "BF16": 2,
    "F16": 2,
    "FP16": 2,
    "U8": 1,
    "F8_E4M3": 1,
    "F32": 4,
    "I64": 8,
}

DTYPE_TO_TORCH: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "FP16": torch.float16,
    "U8": torch.uint8,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", torch.uint8),
    "F32": torch.float32,
    "I64": torch.int64,
}


class QWEN4_5_W:
    """Self-contained runtime weight key constants for Qwen 4.5-Flash."""

    # Global weights (NO model.norm!)
    embedding = "qwen4_5.embedding"
    lm_head = "qwen4_5.lm_head"
    hc_mixer_norm_w = "qwen4_5.hc_mixer.hc_norm.weight"
    hc_mixer_norm_b = "qwen4_5.hc_mixer.hc_norm.bias"
    hc_mixer_mix_down = "qwen4_5.hc_mixer.input_mix_weight_down"
    hc_mixer_mix_up = "qwen4_5.hc_mixer.input_mix_weight_up"

    # Per-layer Hyper-Connections
    hc_attn_norm_w = "qwen4_5.hc_attn.hc_norm.weight"
    hc_attn_norm_b = "qwen4_5.hc_attn.hc_norm.bias"
    hc_attn_inject = "qwen4_5.hc_attn.block_inject_weight"
    hc_attn_mix_down = "qwen4_5.hc_attn.input_mix_weight_down"
    hc_attn_mix_up = "qwen4_5.hc_attn.input_mix_weight_up"

    hc_mlp_norm_w = "qwen4_5.hc_mlp.hc_norm.weight"
    hc_mlp_norm_b = "qwen4_5.hc_mlp.hc_norm.bias"
    hc_mlp_inject = "qwen4_5.hc_mlp.block_inject_weight"
    hc_mlp_mix_down = "qwen4_5.hc_mlp.input_mix_weight_down"
    hc_mlp_mix_up = "qwen4_5.hc_mlp.input_mix_weight_up"

    # PLE (Layer 1, 0-based)
    ple_ngram_shards = "qwen4_5.ple.ngram_shards"
    ple_ngram_scale = "qwen4_5.ple.ngram_scale"
    ple_multipliers = "qwen4_5.ple.layer_multipliers"
    ple_ngram_offsets = "qwen4_5.ple.ngram_heads_offsets"
    ple_ngram_vocab_sizes = "qwen4_5.ple.ngram_heads_vocab_sizes"
    ple_key_proj_w = "qwen4_5.ple.key_proj"
    ple_value_proj_w = "qwen4_5.ple.value_proj"
    ple_conv_w = "qwen4_5.ple.conv1d"
    ple_norm_key_w = "qwen4_5.ple.norm_key"
    ple_norm_query_w = "qwen4_5.ple.norm_query"
    ple_norm_conv_w = "qwen4_5.ple.norm_conv"

    # Linear Attention (Gated DeltaNet)
    linear_in_proj_qkvz = "qwen4_5.linear_attn.in_proj_qkvz"
    linear_in_proj_ba = "qwen4_5.linear_attn.in_proj_ba"
    linear_conv1d = "qwen4_5.linear_attn.conv1d"
    linear_a_log = "qwen4_5.linear_attn.A_log"
    linear_dt_bias = "qwen4_5.linear_attn.dt_bias"
    linear_norm_w = "qwen4_5.linear_attn.norm"
    linear_out_proj = "qwen4_5.linear_attn.out_proj"
    linear_attn_output_sconv = "qwen4_5.linear_attn_output_sconv"

    # Full Attention (MQA-1 + Output Gate + KV ShortConv)
    full_q_proj = "qwen4_5.self_attn.q_proj"
    full_k_proj = "qwen4_5.self_attn.k_proj"
    full_v_proj = "qwen4_5.self_attn.v_proj"
    full_o_proj = "qwen4_5.self_attn.o_proj"
    full_q_norm_w = "qwen4_5.self_attn.q_norm"
    full_k_norm_w = "qwen4_5.self_attn.k_norm"
    full_k_sconv = "qwen4_5.self_attn.k_sconv"
    full_v_sconv = "qwen4_5.self_attn.v_sconv"
    full_attn_output_sconv = "qwen4_5.full_attn_output_sconv"
    full_indexer_qk_proj = "qwen4_5.self_attn.indexer.index_qk_proj"
    full_indexer_q_norm_w = "qwen4_5.self_attn.indexer.q_layernorm"
    full_indexer_k_norm_w = "qwen4_5.self_attn.indexer.k_layernorm"

    # Dense FFN (Layer 0)
    dense_gate_proj = "qwen4_5.mlp.gate_proj"
    dense_up_proj = "qwen4_5.mlp.up_proj"
    dense_down_proj = "qwen4_5.mlp.down_proj"

    # Latent MoE + Shared Expert + Routed MXFP4 Experts (Layers 1..39)
    moe_latent_down_proj = "qwen4_5.mlp.latent_down_proj"
    moe_latent_gate_proj = "qwen4_5.mlp.latent_gate_proj"
    moe_latent_up_proj = "qwen4_5.mlp.latent_up_proj"
    moe_latent_scale_gate = "qwen4_5.mlp.latent_scale_gate"
    moe_pre_latent_ln_w = "qwen4_5.mlp.pre_moe_latent_layernorm"
    moe_router_gate = "qwen4_5.mlp.gate"
    moe_expert_bias = "qwen4_5.mlp.expert_bias"
    moe_shared_gate_proj = "qwen4_5.mlp.shared_expert.gate_proj"
    moe_shared_up_proj = "qwen4_5.mlp.shared_expert.up_proj"
    moe_shared_down_proj = "qwen4_5.mlp.shared_expert.down_proj"
    moe_shared_expert_gate = "qwen4_5.mlp.shared_expert_gate"
    moe_expert_gate_w = "qwen4_5.mlp.experts.{e}.gate_proj.weight"
    moe_expert_gate_scale = "qwen4_5.mlp.experts.{e}.gate_proj.weight_scale"
    moe_expert_up_w = "qwen4_5.mlp.experts.{e}.up_proj.weight"
    moe_expert_up_scale = "qwen4_5.mlp.experts.{e}.up_proj.weight_scale"
    moe_expert_down_w = "qwen4_5.mlp.experts.{e}.down_proj.weight"
    moe_expert_down_scale = "qwen4_5.mlp.experts.{e}.down_proj.weight_scale"

    # FFN Output ShortConv
    ffn_output_sconv = "qwen4_5.ffn_output_sconv"


@dataclass(frozen=True)
class Qwen4_5WeightDescriptor:
    """Pure-Python typed descriptor for a single checkpoint weight tensor."""

    name: str
    ckpt_name: str
    dtype: str
    shape: Tuple[int, ...]
    layer_idx: Optional[int] = None
    resident_device: str = "cuda"
    is_mxfp4_weight: bool = False
    is_mxfp4_scale: bool = False
    is_ple_shard: bool = False

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_NUM_BYTES[self.dtype]


def _build_global_descriptors(
    arch_cfg: Qwen4_5ArchConfig,
    prefix: str = "model.",
) -> List[Qwen4_5WeightDescriptor]:
    h = arch_cfg.hidden_size
    v = arch_cfg.vocab_size
    h_hc = arch_cfg.hc_count * h
    hc_lr = arch_cfg.hc_lowrank

    descs = [
        Qwen4_5WeightDescriptor(
            QWEN4_5_W.embedding, f"{prefix}embed_tokens.weight", "BF16", (v, h)
        ),
    ]
    if not arch_cfg.tie_word_embeddings:
        descs.append(
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.lm_head, "lm_head.weight", "BF16", (v, h)
            )
        )
    mixer = f"{prefix}hyper_connection_mixer."
    descs.extend(
        [
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mixer_norm_w, f"{mixer}hc_norm.weight", "BF16", (h_hc,)
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mixer_norm_b, f"{mixer}hc_norm.bias", "BF16", (h_hc,)
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mixer_mix_down,
                f"{mixer}input_mix_weight_down.weight",
                "BF16",
                (hc_lr, h_hc),
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mixer_mix_up,
                f"{mixer}input_mix_weight_up.weight",
                "BF16",
                (h_hc, hc_lr),
            ),
        ]
    )
    return descs


def _build_layer_descriptors(
    arch_cfg: Qwen4_5ArchConfig,
    layer_idx: int,
    share_map: Mapping[int, int],
    *,
    prefix: str = "model.",
    ple_shard_shape: Optional[Tuple[int, int]] = None,
) -> List[Qwen4_5WeightDescriptor]:
    h = arch_cfg.hidden_size
    hc = arch_cfg.hc_count
    h_hc = hc * h
    hc_lr = arch_cfg.hc_lowrank
    is_consumer = layer_idx in share_map
    ltype = arch_cfg.layer_types[layer_idx]
    p = f"{prefix}layers.{layer_idx}."
    descs: List[Qwen4_5WeightDescriptor] = []

    # 1. Attn Hyper-Connection (Layer 0 up is [H, HC_LR]; Layers 1..39 [HC*H, HC_LR])
    descs.extend(
        [
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_attn_norm_w,
                f"{p}attn_hyper_connection.hc_norm.weight",
                "BF16",
                (h_hc,),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_attn_norm_b,
                f"{p}attn_hyper_connection.hc_norm.bias",
                "BF16",
                (h_hc,),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_attn_inject,
                f"{p}attn_hyper_connection.block_inject_weight.weight",
                "BF16",
                (hc, h_hc),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_attn_mix_down,
                f"{p}attn_hyper_connection.input_mix_weight_down.weight",
                "BF16",
                (hc_lr, h_hc),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_attn_mix_up,
                f"{p}attn_hyper_connection.input_mix_weight_up.weight",
                "BF16",
                (h if layer_idx == 0 else h_hc, hc_lr),
                layer_idx,
            ),
        ]
    )

    # 2. MLP Hyper-Connection (Final layer omits input_mix_weight_down)
    descs.extend(
        [
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mlp_norm_w,
                f"{p}mlp_hyper_connection.hc_norm.weight",
                "BF16",
                (h_hc,),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mlp_norm_b,
                f"{p}mlp_hyper_connection.hc_norm.bias",
                "BF16",
                (h_hc,),
                layer_idx,
            ),
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mlp_inject,
                f"{p}mlp_hyper_connection.block_inject_weight.weight",
                "BF16",
                (hc, h_hc),
                layer_idx,
            ),
        ]
    )
    if layer_idx < arch_cfg.num_hidden_layers - 1:
        descs.append(
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.hc_mlp_mix_down,
                f"{p}mlp_hyper_connection.input_mix_weight_down.weight",
                "BF16",
                (hc_lr, h_hc),
                layer_idx,
            )
        )
    descs.append(
        Qwen4_5WeightDescriptor(
            QWEN4_5_W.hc_mlp_mix_up,
            f"{p}mlp_hyper_connection.input_mix_weight_up.weight",
            "BF16",
            (h_hc, hc_lr),
            layer_idx,
        )
    )

    # 3. PLE (1-based ple_layer_ids -> 0-based layer_idx)
    if (layer_idx + 1) in arch_cfg.ple_layer_ids:
        shard_shape = ple_shard_shape or compute_ple_ngram_shard_shape(arch_cfg)
        n_heads = (arch_cfg.ngram_size - 1) * arch_cfg.heads_per_ngram
        is_fp8_ple = arch_cfg.ple_quant_scheme == "fp8_pertensor"
        shard_dtype = "F8_E4M3" if is_fp8_ple else "BF16"
        ple_p = f"{p}ple."
        for s in range(arch_cfg.split_ngram_parts):
            descs.append(
                Qwen4_5WeightDescriptor(
                    f"{QWEN4_5_W.ple_ngram_shards}.{s}",
                    f"{ple_p}ple_embedding.ngram_embedding.shard_{s}.weight",
                    shard_dtype,
                    shard_shape,
                    layer_idx,
                    resident_device="cpu",
                    is_ple_shard=True,
                )
            )
        if is_fp8_ple:
            descs.append(
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_ngram_scale,
                    f"{ple_p}ple_embedding.ngram_embedding.weight_scale",
                    "BF16",
                    (1,),
                    layer_idx,
                    resident_device="cpu",
                )
            )
        descs.extend(
            [
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_multipliers,
                    f"{ple_p}ple_embedding.layer_multipliers",
                    "I64",
                    (arch_cfg.ngram_size,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_ngram_offsets,
                    f"{ple_p}ple_embedding.ngram_heads_offsets",
                    "I64",
                    (n_heads,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_ngram_vocab_sizes,
                    f"{ple_p}ple_embedding.ngram_heads_vocab_sizes",
                    "I64",
                    (n_heads,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_key_proj_w,
                    f"{ple_p}key_proj.weight",
                    "BF16",
                    (h_hc, arch_cfg.ple_embed_dim),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_value_proj_w,
                    f"{ple_p}value_proj.weight",
                    "BF16",
                    (h, arch_cfg.ple_embed_dim),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_conv_w,
                    f"{ple_p}conv1d.weight",
                    "BF16",
                    (h_hc, 1, arch_cfg.ple_conv_kernel_size),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_norm_key_w,
                    f"{ple_p}norm_key.weight",
                    "BF16",
                    (h_hc,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_norm_query_w,
                    f"{ple_p}norm_query.weight",
                    "BF16",
                    (h_hc,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.ple_norm_conv_w,
                    f"{ple_p}norm_conv.weight",
                    "BF16",
                    (h_hc,),
                    layer_idx,
                ),
            ]
        )

    # 4. Attention (Linear GDN vs Full MQA-1)
    if ltype == "linear_attention":
        la = f"{p}linear_attn."
        k_dim = arch_cfg.linear_num_key_heads * arch_cfg.linear_key_head_dim
        v_dim = arch_cfg.linear_num_value_heads * arch_cfg.linear_value_head_dim
        k_conv = arch_cfg.linear_conv_kernel_dim
        if not is_consumer:
            descs.extend(
                [
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_in_proj_qkvz,
                        f"{la}in_proj_qkvz.weight",
                        "BF16",
                        (2 * k_dim + 2 * v_dim, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_in_proj_ba,
                        f"{la}in_proj_ba.weight",
                        "BF16",
                        (2 * arch_cfg.linear_num_value_heads, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_conv1d,
                        f"{la}conv1d.weight",
                        "BF16",
                        (2 * k_dim + v_dim, 1, k_conv),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_a_log,
                        f"{la}A_log",
                        "BF16",
                        (arch_cfg.linear_num_value_heads,),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_dt_bias,
                        f"{la}dt_bias",
                        "BF16",
                        (arch_cfg.linear_num_value_heads,),
                        layer_idx,
                    ),
                ]
            )
        else:
            descs.append(
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.linear_in_proj_qkvz,
                    f"{la}in_proj_qkvz.weight",
                    "BF16",
                    (k_dim + v_dim, h),
                    layer_idx,
                )
            )
            if not arch_cfg.gdn_no_q_conv:
                descs.append(
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.linear_conv1d,
                        f"{la}conv1d.weight",
                        "BF16",
                        (k_dim, 1, k_conv),
                        layer_idx,
                    )
                )
        descs.extend(
            [
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.linear_norm_w,
                    f"{la}norm.weight",
                    "BF16",
                    (arch_cfg.linear_value_head_dim,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.linear_out_proj,
                    f"{la}out_proj.weight",
                    "BF16",
                    (h, v_dim),
                    layer_idx,
                ),
            ]
        )
        if arch_cfg.has_output_sconv(
            "linear_attn_output_sconv", is_consumer=is_consumer
        ):
            descs.append(
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.linear_attn_output_sconv,
                    f"{p}linear_attn_output_sconv.weight",
                    "BF16",
                    (h, 1, k_conv),
                    layer_idx,
                )
            )
    else:
        sa = f"{p}self_attn."
        q_dim = arch_cfg.num_attention_heads * arch_cfg.head_dim
        kv_dim = arch_cfg.num_key_value_heads * arch_cfg.head_dim
        q_out = q_dim * 2 if arch_cfg.attn_output_gate else q_dim
        k_conv = arch_cfg.linear_conv_kernel_dim
        descs.extend(
            [
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.full_q_proj,
                    f"{sa}q_proj.weight",
                    "BF16",
                    (q_out, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.full_q_norm_w,
                    f"{sa}q_norm.weight",
                    "BF16",
                    (arch_cfg.head_dim,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.full_o_proj,
                    f"{sa}o_proj.weight",
                    "BF16",
                    (h, q_dim),
                    layer_idx,
                ),
            ]
        )
        if not is_consumer:
            descs.extend(
                [
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_k_proj,
                        f"{sa}k_proj.weight",
                        "BF16",
                        (kv_dim, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_v_proj,
                        f"{sa}v_proj.weight",
                        "BF16",
                        (kv_dim, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_k_norm_w,
                        f"{sa}k_norm.weight",
                        "BF16",
                        (arch_cfg.head_dim,),
                        layer_idx,
                    ),
                ]
            )
            if arch_cfg.use_full_attn_kv_sconv:
                descs.extend(
                    [
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.full_k_sconv,
                            f"{sa}k_sconv.weight",
                            "BF16",
                            (kv_dim, 1, k_conv),
                            layer_idx,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.full_v_sconv,
                            f"{sa}v_sconv.weight",
                            "BF16",
                            (kv_dim, 1, k_conv),
                            layer_idx,
                        ),
                    ]
                )
        if arch_cfg.has_output_sconv("full_attn_output_sconv", is_consumer=is_consumer):
            descs.append(
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.full_attn_output_sconv,
                    f"{p}full_attn_output_sconv.weight",
                    "BF16",
                    (h, 1, k_conv),
                    layer_idx,
                )
            )
        if arch_cfg.has_qsa:
            idx_d = arch_cfg.indexer_head_dim
            idx_qk_out = (arch_cfg.indexer_n_heads + arch_cfg.indexer_kv_heads) * idx_d
            descs.extend(
                [
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_indexer_qk_proj,
                        f"{sa}indexer.index_qk_proj.weight",
                        "BF16",
                        (idx_qk_out, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_indexer_q_norm_w,
                        f"{sa}indexer.q_layernorm.weight",
                        "BF16",
                        (idx_d,),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.full_indexer_k_norm_w,
                        f"{sa}indexer.k_layernorm.weight",
                        "BF16",
                        (idx_d,),
                        layer_idx,
                    ),
                ]
            )

    # 5. FFN (Dense on Layer 0; Latent MoE + Shared Expert + MXFP4 Routed Experts on Layers 1..39)
    if layer_idx < arch_cfg.first_k_dense_replace:
        inter = arch_cfg.intermediate_size
        descs.extend(
            [
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.dense_gate_proj,
                    f"{p}mlp.gate_proj.weight",
                    "BF16",
                    (inter, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.dense_up_proj,
                    f"{p}mlp.up_proj.weight",
                    "BF16",
                    (inter, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.dense_down_proj,
                    f"{p}mlp.down_proj.weight",
                    "BF16",
                    (h, inter),
                    layer_idx,
                ),
            ]
        )
    else:
        r_h = arch_cfg.routed_expert_hidden_size
        m_i = arch_cfg.moe_intermediate_size
        s_i = arch_cfg.shared_expert_intermediate_size
        descs.extend(
            [
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_latent_down_proj,
                    f"{p}mlp.latent_down_proj",
                    "BF16",
                    (r_h, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_latent_gate_proj,
                    f"{p}mlp.latent_gate_proj",
                    "BF16",
                    (r_h, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_latent_up_proj,
                    f"{p}mlp.latent_up_proj",
                    "BF16",
                    (h, r_h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_latent_scale_gate,
                    f"{p}mlp.latent_scale_gate",
                    "BF16",
                    (1, h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_pre_latent_ln_w,
                    f"{p}mlp.pre_moe_latent_layernorm.weight",
                    "BF16",
                    (r_h,),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_router_gate,
                    f"{p}mlp.gate.weight",
                    "BF16",
                    (arch_cfg.num_experts, r_h),
                    layer_idx,
                ),
                Qwen4_5WeightDescriptor(
                    QWEN4_5_W.moe_expert_bias,
                    f"{p}mlp.expert_bias",
                    "F32",
                    (arch_cfg.num_experts,),
                    layer_idx,
                ),
            ]
        )
        if s_i > 0:
            descs.extend(
                [
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.moe_shared_gate_proj,
                        f"{p}mlp.shared_expert.gate_proj.weight",
                        "BF16",
                        (s_i, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.moe_shared_up_proj,
                        f"{p}mlp.shared_expert.up_proj.weight",
                        "BF16",
                        (s_i, h),
                        layer_idx,
                    ),
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.moe_shared_down_proj,
                        f"{p}mlp.shared_expert.down_proj.weight",
                        "BF16",
                        (h, s_i),
                        layer_idx,
                    ),
                ]
            )
            if arch_cfg.shared_expert_gate:
                descs.append(
                    Qwen4_5WeightDescriptor(
                        QWEN4_5_W.moe_shared_expert_gate,
                        f"{p}mlp.shared_expert_gate.weight",
                        "BF16",
                        (1, h),
                        layer_idx,
                    )
                )

        is_mxfp4 = arch_cfg.quant_method == "mxfp4"
        for e in range(arch_cfg.num_experts):
            ep = f"{p}mlp.experts.{e}."
            if is_mxfp4:
                descs.extend(
                    [
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_gate_w.format(e=e),
                            f"{ep}gate_proj.weight",
                            "U8",
                            (m_i, r_h // 2),
                            layer_idx,
                            is_mxfp4_weight=True,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_gate_scale.format(e=e),
                            f"{ep}gate_proj.weight_scale",
                            "U8",
                            (m_i, r_h // MXFP4_GROUP_SIZE),
                            layer_idx,
                            is_mxfp4_scale=True,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_up_w.format(e=e),
                            f"{ep}up_proj.weight",
                            "U8",
                            (m_i, r_h // 2),
                            layer_idx,
                            is_mxfp4_weight=True,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_up_scale.format(e=e),
                            f"{ep}up_proj.weight_scale",
                            "U8",
                            (m_i, r_h // MXFP4_GROUP_SIZE),
                            layer_idx,
                            is_mxfp4_scale=True,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_down_w.format(e=e),
                            f"{ep}down_proj.weight",
                            "U8",
                            (r_h, m_i // 2),
                            layer_idx,
                            is_mxfp4_weight=True,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_down_scale.format(e=e),
                            f"{ep}down_proj.weight_scale",
                            "U8",
                            (r_h, m_i // MXFP4_GROUP_SIZE),
                            layer_idx,
                            is_mxfp4_scale=True,
                        ),
                    ]
                )
            else:
                descs.extend(
                    [
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_gate_w.format(e=e),
                            f"{ep}gate_proj.weight",
                            "BF16",
                            (m_i, r_h),
                            layer_idx,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_up_w.format(e=e),
                            f"{ep}up_proj.weight",
                            "BF16",
                            (m_i, r_h),
                            layer_idx,
                        ),
                        Qwen4_5WeightDescriptor(
                            QWEN4_5_W.moe_expert_down_w.format(e=e),
                            f"{ep}down_proj.weight",
                            "BF16",
                            (r_h, m_i),
                            layer_idx,
                        ),
                    ]
                )

    if arch_cfg.use_ffn_output_sconv:
        descs.append(
            Qwen4_5WeightDescriptor(
                QWEN4_5_W.ffn_output_sconv,
                f"{p}ffn_output_sconv.weight",
                "BF16",
                (h, 1, arch_cfg.linear_conv_kernel_dim),
                layer_idx,
            )
        )

    return descs


def expected_checkpoint_tensor_specs(
    arch_cfg: Qwen4_5ArchConfig,
    *,
    prefix: str = "model.",
) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
    """Enumerate every expected safetensors tensor name, dtype, and exact shape.

    Returns ``{tensor_name: (dtype_str, shape_tuple)}`` for the entire checkpoint.
    For the 1109 Qwen 4.5-Flash checkpoint this yields all 151,038 tensors
    totaling exactly 461,163,888,730 bytes (429.492 GiB).
    """
    share_map = validate_qwen4_5_arch_config(arch_cfg)
    ple_shard_shape = (
        compute_ple_ngram_shard_shape(arch_cfg) if arch_cfg.ple_layer_ids else None
    )

    specs: Dict[str, Tuple[str, Tuple[int, ...]]] = {}
    for desc in _build_global_descriptors(arch_cfg, prefix=prefix):
        specs[desc.ckpt_name] = (desc.dtype, desc.shape)

    for layer_idx in range(arch_cfg.num_hidden_layers):
        for desc in _build_layer_descriptors(
            arch_cfg,
            layer_idx,
            share_map,
            prefix=prefix,
            ple_shard_shape=ple_shard_shape,
        ):
            specs[desc.ckpt_name] = (desc.dtype, desc.shape)

    return specs


def summarize_checkpoint_specs(
    specs: Mapping[str, Tuple[str, Tuple[int, ...]]],
    arch_cfg: Optional[Qwen4_5ArchConfig] = None,
) -> Dict[str, Any]:
    """Compute exact tensor count, byte totals, and logical/core/active parameter counts."""
    total_bytes = 0
    logical_params = 0
    ple_table_params = 0
    embed_and_lm_head_params = 0
    routed_expert_params = 0
    non_expert_core_params = 0
    bytes_by_dtype: Counter[str] = Counter()
    tensors_by_dtype: Counter[str] = Counter()

    for name, (dtype, shape) in specs.items():
        numel = math.prod(shape)
        nbytes = numel * DTYPE_NUM_BYTES[dtype]
        total_bytes += nbytes
        bytes_by_dtype[dtype] += nbytes
        tensors_by_dtype[dtype] += 1

        # Quantization scales and integer n-gram hash metadata are not model parameters.
        if name.endswith(".weight_scale") or dtype == "I64":
            continue

        # MXFP4 weights pack 2 FP4 values per U8 byte.
        lparams = numel * 2 if dtype == "U8" else numel
        logical_params += lparams

        if "ngram_embedding.shard_" in name:
            ple_table_params += lparams
        elif name.endswith("embed_tokens.weight") or name == "lm_head.weight":
            embed_and_lm_head_params += lparams
        elif ".mlp.experts." in name:
            routed_expert_params += lparams
        else:
            non_expert_core_params += lparams

    core_params = logical_params - ple_table_params
    if arch_cfg is not None and arch_cfg.num_experts > 0 and routed_expert_params > 0:
        active_routed = (
            routed_expert_params * arch_cfg.num_experts_per_tok
        ) // arch_cfg.num_experts
    else:
        active_routed = routed_expert_params
    active_params_excl_embed = non_expert_core_params + active_routed

    return {
        "tensor_count": len(specs),
        "total_bytes": total_bytes,
        "logical_params": logical_params,
        "core_params": core_params,
        "ple_table_params": ple_table_params,
        "embed_and_lm_head_params": embed_and_lm_head_params,
        "routed_expert_params": routed_expert_params,
        "non_expert_core_params": non_expert_core_params,
        "active_params_excl_embed": active_params_excl_embed,
        "bytes_by_dtype": dict(bytes_by_dtype),
        "tensors_by_dtype": dict(tensors_by_dtype),
    }


def validate_qwen4_5_weight_map(
    arch_cfg: Qwen4_5ArchConfig,
    weight_map: Union[Mapping[str, Any], Collection[str]],
    *,
    prefix: str = "model.",
    allow_extra_prefixes: Sequence[str] = (),
) -> Dict[str, Any]:
    """Validate exact tensor coverage (and optional dtype/shape) against a checkpoint index.

    ``weight_map`` may be:
      - ``model.safetensors.index.json["weight_map"]`` (``{tensor_name: shard_filename}``)
      - a set/collection of tensor names
      - a mapping ``{tensor_name: (dtype_str, shape_tuple)}`` or
        ``{tensor_name: {"dtype": ..., "shape": ...}}`` to additionally verify
        exact dtype and shape for every tensor.
    """
    expected = expected_checkpoint_tensor_specs(arch_cfg, prefix=prefix)
    is_mapping = isinstance(weight_map, Mapping)
    actual_keys = set(weight_map.keys()) if is_mapping else set(weight_map)

    ignored_extra = {
        k
        for k in actual_keys
        if k not in expected
        and any(k.startswith(pfx) for pfx in allow_extra_prefixes)
    }
    filtered_actual = actual_keys - ignored_extra

    missing = sorted(set(expected.keys()) - filtered_actual)
    unexpected = sorted(filtered_actual - set(expected.keys()))

    dtype_shape_mismatches: List[str] = []
    if is_mapping:
        for key in expected.keys() & filtered_actual:
            val = weight_map[key]
            exp_dtype, exp_shape = expected[key]
            act_dtype: Optional[str] = None
            act_shape: Optional[Tuple[int, ...]] = None
            if (
                isinstance(val, tuple)
                and len(val) == 2
                and isinstance(val[0], str)
                and isinstance(val[1], (tuple, list))
            ):
                act_dtype = val[0]
                act_shape = tuple(int(d) for d in val[1])
            elif isinstance(val, Mapping) and "dtype" in val and "shape" in val:
                act_dtype = str(val["dtype"])
                act_shape = tuple(int(d) for d in val["shape"])
            if act_dtype is not None and act_shape is not None:
                if act_dtype != exp_dtype or act_shape != exp_shape:
                    dtype_shape_mismatches.append(
                        f"{key}: expected ({exp_dtype}, {exp_shape}), "
                        f"got ({act_dtype}, {act_shape})"
                    )

    errors: List[str] = []
    if missing:
        errors.append(
            f"missing {len(missing)} expected tensor(s), first 5: {missing[:5]}"
        )
    if unexpected:
        errors.append(
            f"unexpected {len(unexpected)} extra tensor(s), first 5: {unexpected[:5]}"
        )
    if dtype_shape_mismatches:
        errors.append(
            f"{len(dtype_shape_mismatches)} dtype/shape mismatch(es), "
            f"first 5: {dtype_shape_mismatches[:5]}"
        )
    if errors:
        raise ValueError("Qwen 4.5-Flash weight_map validation failed: " + "; ".join(errors))

    return {
        "expected_count": len(expected),
        "matched_count": len(expected),
        "extra_ignored_count": len(ignored_extra),
    }


if _HAS_RTP_WEIGHT_LOADER:

    class Qwen4_5PleNgramWeight(AtomicWeight):
        """Rank-local host-memory loader for Qwen 4.5-Flash 128-shard FP8 PLE table."""

        supports_fastsafetensors_iteration = False

        def __init__(
            self,
            name: str,
            weights: List[CkptWeightInfo],
            *,
            scale_weight: Optional[CkptWeightInfo] = None,
            preserve_fp8: bool = True,
            **kwargs: Any,
        ) -> None:
            if "cpu_offload" in kwargs:
                raise TypeError("Qwen 4.5 PLE table residency is fixed to CPU")
            super().__init__(name, weights, **kwargs)
            self.resident_device = "cpu"
            self.scale_weight = scale_weight
            self.preserve_fp8 = preserve_fp8
            self.resident_output_devices = (
                {QWEN4_5_W.ple_ngram_scale: "cpu"} if preserve_fp8 else {}
            )

        def _local_shard_range(self, load_config: LoadConfig) -> range:
            shard_count = len(self.weights)
            tp_size = load_config.tp_size
            tp_rank = load_config.tp_rank
            if tp_size < 1 or shard_count % tp_size != 0:
                raise ValueError(
                    f"Qwen 4.5 PLE has {shard_count} shards, incompatible with tp_size={tp_size}"
                )
            per_rank = shard_count // tp_size
            start = tp_rank * per_rank
            return range(start, start + per_rank)

        def get_tensor_names(
            self, layer_id: Optional[int], load_config: LoadConfig
        ) -> set[str]:
            names = {
                self.weights[idx].tensor_name(layer_id)
                for idx in self._local_shard_range(load_config)
            }
            if self.scale_weight is not None:
                names.add(self.scale_weight.tensor_name(layer_id))
            return names

        def _load_raw_tensor(
            self,
            tensor_source: TensorSource,
            layer_id: Optional[int],
            device: str,
            load_config: LoadConfig,
        ) -> Dict[str, torch.Tensor]:
            if layer_id is None:
                raise ValueError("Qwen 4.5 PLE n-gram table is a per-layer weight")
            convert_type = (
                DTYPE_TO_TORCH["F8_E4M3"]
                if self.preserve_fp8
                else self.data_type or load_config.compute_dtype
            )
            result: Dict[str, torch.Tensor] = {}
            scale: Optional[torch.Tensor] = None
            if self.scale_weight is not None:
                scale = self.scale_weight.merge_fun(
                    tensor_source.load_tensor(
                        self.scale_weight.tensor_name(layer_id), torch.float32
                    )
                )
                if self.preserve_fp8:
                    result[QWEN4_5_W.ple_ngram_scale] = scale.to(device="cpu")
                else:
                    scale = scale.to(device=device, dtype=convert_type)
            for idx in self._local_shard_range(load_config):
                w = self.weights[idx]
                shard = w.merge_fun(
                    tensor_source.load_tensor(w.tensor_name(layer_id), convert_type)
                ).to(device=device, dtype=convert_type)
                if scale is not None and not self.preserve_fp8:
                    shard = shard * scale
                result[f"{self.name}.{idx}"] = shard
            return result

        def _split(
            self, tensor: Union[torch.Tensor, Dict[str, torch.Tensor]], load_config: LoadConfig
        ) -> Dict[str, torch.Tensor]:
            if not isinstance(tensor, dict):
                raise TypeError("Qwen4_5PleNgramWeight expected a tensor dictionary")
            return tensor

        def _postprocess(
            self,
            tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
            device: str,
            load_config: LoadConfig,
        ) -> Dict[str, torch.Tensor]:
            if not isinstance(tensor, dict):
                raise TypeError("Qwen4_5PleNgramWeight expected a tensor dictionary")
            return tensor


class Qwen4_5Weight(_BaseWeightInfo):
    """Weight layout and RTP-LLM loader descriptor builder for Qwen 4.5-Flash.

    Subclasses ``ModelDeployWeightInfo`` when compiled RTP-LLM ops are available,
    and functions as a standalone pure-Python descriptor builder otherwise.
    """

    def __init__(
        self,
        model_config: Any = None,
        parallelism_config: Any = None,
        hw_kernel_config: Any = None,
        kv_cache_config: Any = None,
        *,
        arch_config: Optional[Qwen4_5ArchConfig] = None,
        prefix: str = "model.",
        **kwargs: Any,
    ) -> None:
        self.prefix = prefix
        if arch_config is not None:
            self.arch_config = arch_config
        elif model_config is not None and hasattr(model_config, "arch_config"):
            self.arch_config = model_config.arch_config
        elif model_config is not None and getattr(
            model_config, "qwen4_5_arch_config", None
        ) is not None:
            self.arch_config = model_config.qwen4_5_arch_config
        else:
            self.arch_config = Qwen4_5ArchConfig()

        self.model_config = model_config
        if _HAS_RTP_WEIGHT_LOADER and model_config is not None and parallelism_config is not None:
            super().__init__(
                model_config=model_config,
                parallelism_config=parallelism_config,
                hw_kernel_config=hw_kernel_config,
                kv_cache_config=kv_cache_config,
                **kwargs,
            )

    def _process_meta(self, meta_dict: Any, weight_keys: Collection[str]) -> None:
        for key in weight_keys:
            if key.endswith(_PREFIX_ANCHOR) and "mtp." not in key:
                self.prefix = key[: -len(_PREFIX_ANCHOR)]
                return
        raise ValueError(
            f"Qwen4_5Weight: cannot determine prefix, no non-mtp key ending with "
            f"{_PREFIX_ANCHOR!r} in {len(weight_keys)} checkpoint keys"
        )

    def get_global_weight_descriptors(self) -> List[Qwen4_5WeightDescriptor]:
        return _build_global_descriptors(self.arch_config, prefix=self.prefix)

    def get_layer_weight_descriptors(
        self,
        layer_idx: int,
        *,
        share_map: Optional[Mapping[int, int]] = None,
        ple_shard_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Qwen4_5WeightDescriptor]:
        smap = (
            share_map
            if share_map is not None
            else validate_qwen4_5_arch_config(self.arch_config)
        )
        return _build_layer_descriptors(
            self.arch_config,
            layer_idx,
            smap,
            prefix=self.prefix,
            ple_shard_shape=ple_shard_shape,
        )

    def get_all_layer_weight_descriptors(self) -> List[List[Qwen4_5WeightDescriptor]]:
        smap = validate_qwen4_5_arch_config(self.arch_config)
        ple_shard_shape = (
            compute_ple_ngram_shard_shape(self.arch_config)
            if self.arch_config.ple_layer_ids
            else None
        )
        return [
            _build_layer_descriptors(
                self.arch_config,
                layer_idx,
                smap,
                prefix=self.prefix,
                ple_shard_shape=ple_shard_shape,
            )
            for layer_idx in range(self.arch_config.num_hidden_layers)
        ]

    def expected_tensor_specs(self) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        return expected_checkpoint_tensor_specs(self.arch_config, prefix=self.prefix)

    def validate_weight_map(
        self,
        weight_map: Union[Mapping[str, Any], Collection[str]],
        *,
        allow_extra_prefixes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        return validate_qwen4_5_weight_map(
            self.arch_config,
            weight_map,
            prefix=self.prefix,
            allow_extra_prefixes=allow_extra_prefixes,
        )

    def _get_weight_info(self) -> ModelWeightInfo:
        """Build RTP-LLM ``ModelWeightInfo`` using ``AtomicWeight`` descriptors."""
        if not _HAS_RTP_WEIGHT_LOADER:
            raise RuntimeError(
                "RTP-LLM C++ weight loader is unavailable in this pure-Python environment; "
                "use get_global_weight_descriptors() / get_all_layer_weight_descriptors()."
            )

        global_weights: List[WeightModule] = [
            AtomicWeight(
                desc.name,
                [CkptWeightInfo(desc.ckpt_name, identity)],
                data_type=DTYPE_TO_TORCH[desc.dtype],
            )
            for desc in self.get_global_weight_descriptors()
        ]

        layer_weights: List[List[WeightModule]] = []
        for layer_idx, layer_descs in enumerate(self.get_all_layer_weight_descriptors()):
            mods: List[WeightModule] = []
            ple_shard_infos: List[CkptWeightInfo] = []
            ple_scale_info: Optional[CkptWeightInfo] = None
            for desc in layer_descs:
                if desc.is_ple_shard:
                    ple_shard_infos.append(CkptWeightInfo(desc.ckpt_name, identity))
                    continue
                if desc.name == QWEN4_5_W.ple_ngram_scale:
                    ple_scale_info = CkptWeightInfo(desc.ckpt_name, identity)
                    continue
                mods.append(
                    AtomicWeight(
                        desc.name,
                        [CkptWeightInfo(desc.ckpt_name, identity)],
                        data_type=DTYPE_TO_TORCH[desc.dtype],
                    )
                )
            if ple_shard_infos:
                mods.append(
                    Qwen4_5PleNgramWeight(
                        QWEN4_5_W.ple_ngram_shards,
                        ple_shard_infos,
                        scale_weight=ple_scale_info,
                        preserve_fp8=(
                            self.arch_config.ple_quant_scheme == "fp8_pertensor"
                        ),
                    )
                )
            layer_weights.append(mods)

        return ModelWeightInfo(layer_weights=layer_weights, weights=global_weights)
