"""Qwen 4.5-Flash (Qwen4_5ForCausalLM) configuration parsing, validation, and sizing.

This module works both inside the full RTP-LLM C++/Bazel runtime (subclassing
``ModelConfig``) and in standalone pure-Python environments without compiled
C++ extensions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

try:
    from rtp_llm.config.model_config import ModelConfig as _BaseModelConfig

    _HAS_RTP_MODEL_CONFIG = True
except Exception:  # pragma: no cover - pure-Python fallback without C++ ops
    _BaseModelConfig = object  # type: ignore[assignment,misc]
    _HAS_RTP_MODEL_CONFIG = False

VALID_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
VALID_PLE_QUANT_SCHEMES = frozenset({"fp8_pertensor", "bf16", "none", None})
VALID_QUANT_METHODS = frozenset({"mxfp4", None})
SUPPORTED_ARCHITECTURES = frozenset({"Qwen4_5ForCausalLM"})
EXPECTED_MODEL_TYPE = "qwen4_5"
MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True)
class Qwen4_5Options:
    """Runtime resource tuning options for Qwen 4.5-Flash."""

    ple_prefetch_mb: int = 256
    enable_yoco_decode: bool = True

    def __post_init__(self) -> None:
        if type(self.ple_prefetch_mb) is not int or self.ple_prefetch_mb <= 0:
            raise ValueError("qwen4_5_ple_prefetch_mb must be a positive integer")
        if not isinstance(self.enable_yoco_decode, bool):
            raise ValueError("enable_yoco_decode must be a boolean")


@dataclass(frozen=True)
class Qwen4_5QuantConfig:
    """Parsed quantization specification from Qwen 4.5-Flash ``config.json``."""

    quant_method: Optional[str] = "mxfp4"
    modules_to_not_convert: Tuple[str, ...] = ()
    modules_to_convert: Tuple[str, ...] = ("ple.ple_embedding.ngram_embedding",)
    ple_quant_scheme: Optional[str] = "fp8_pertensor"

    @property
    def is_mxfp4_routed_experts(self) -> bool:
        return self.quant_method == "mxfp4"

    @property
    def is_fp8_ple(self) -> bool:
        return self.ple_quant_scheme == "fp8_pertensor"


@dataclass(frozen=True)
class Qwen4_5ParamAndStateEstimate:
    """Exact logical parameter counts, checkpoint size, and per-request KV/state bytes."""

    total_params: int
    core_params: int
    ple_table_params: int
    embed_and_lm_head_params: int
    active_params_excl_embed: int
    active_params_incl_embed: int
    checkpoint_tensor_count: int
    checkpoint_total_bytes: int

    # Per-token & per-request runtime state sizing
    enable_yoco_decode: bool
    kv_dtype_bytes: int
    full_attn_kv_layers: int
    kv_bytes_per_token: int
    gdn_ssm_layers: int
    ssm_bytes_per_request: int
    short_conv_state_count: int
    short_conv_tensor_count: int
    short_conv_bytes_per_request: int
    ple_ngram_ctx_bytes_per_request: int

    @property
    def fixed_state_bytes_per_request(self) -> int:
        return (
            self.ssm_bytes_per_request
            + self.short_conv_bytes_per_request
            + self.ple_ngram_ctx_bytes_per_request
        )

    @property
    def checkpoint_gib(self) -> float:
        return self.checkpoint_total_bytes / float(1024**3)

    @property
    def kv_kib_per_token(self) -> float:
        return self.kv_bytes_per_token / 1024.0

    @property
    def ssm_mib_per_request(self) -> float:
        return self.ssm_bytes_per_request / float(1024**2)

    @property
    def short_conv_mib_per_request(self) -> float:
        return self.short_conv_bytes_per_request / float(1024**2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_params": self.total_params,
            "core_params": self.core_params,
            "ple_table_params": self.ple_table_params,
            "embed_and_lm_head_params": self.embed_and_lm_head_params,
            "active_params_excl_embed": self.active_params_excl_embed,
            "active_params_incl_embed": self.active_params_incl_embed,
            "checkpoint_tensor_count": self.checkpoint_tensor_count,
            "checkpoint_total_bytes": self.checkpoint_total_bytes,
            "enable_yoco_decode": self.enable_yoco_decode,
            "kv_dtype_bytes": self.kv_dtype_bytes,
            "full_attn_kv_layers": self.full_attn_kv_layers,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "gdn_ssm_layers": self.gdn_ssm_layers,
            "ssm_bytes_per_request": self.ssm_bytes_per_request,
            "short_conv_state_count": self.short_conv_state_count,
            "short_conv_tensor_count": self.short_conv_tensor_count,
            "short_conv_bytes_per_request": self.short_conv_bytes_per_request,
            "ple_ngram_ctx_bytes_per_request": self.ple_ngram_ctx_bytes_per_request,
            "fixed_state_bytes_per_request": self.fixed_state_bytes_per_request,
        }


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _next_prime(n: int) -> int:
    candidate = max(2, int(n))
    if candidate == 2:
        return 2
    if candidate % 2 == 0:
        candidate += 1
    while not _is_prime(candidate):
        candidate += 2
    return candidate


def compute_ple_ngram_head_vocab_sizes(arch_cfg: Qwen4_5ArchConfig) -> Tuple[int, ...]:
    """Return the exact prime vocabulary size for each PLE n-gram hash head.

    Qwen 4.5-Flash allocates ``(ngram_size - 1) * heads_per_ngram`` consecutive
    prime hash table sizes starting at ``ngram_vocab_size_base``.
    """
    n_heads = (arch_cfg.ngram_size - 1) * arch_cfg.heads_per_ngram
    if n_heads <= 0:
        raise ValueError(
            f"PLE n-gram head count must be positive, got {n_heads} "
            f"(ngram_size={arch_cfg.ngram_size}, heads_per_ngram={arch_cfg.heads_per_ngram})"
        )
    primes = []
    cursor = int(arch_cfg.ngram_vocab_size_base)
    for _ in range(n_heads):
        p = _next_prime(cursor)
        primes.append(p)
        cursor = p + 1
    return tuple(primes)


def compute_ple_ngram_shard_shape(arch_cfg: Qwen4_5ArchConfig) -> Tuple[int, int]:
    """Return exact ``(rows_per_shard, head_dim)`` for each PLE n-gram table shard.

    Total rows across all heads are padded up to
    ``lcm(make_ngram_vocab_size_divisible_by, split_ngram_parts)`` and split
    evenly across ``split_ngram_parts`` shards. For the 1109 checkpoint this
    evaluates deterministically to ``(5468784, 256)``.
    """
    n_heads = (arch_cfg.ngram_size - 1) * arch_cfg.heads_per_ngram
    if n_heads <= 0 or arch_cfg.ple_embed_dim % n_heads != 0:
        raise ValueError(
            f"ple_embed_dim ({arch_cfg.ple_embed_dim}) must be divisible by "
            f"n_heads={n_heads}"
        )
    if arch_cfg.split_ngram_parts <= 0:
        raise ValueError(
            f"split_ngram_parts must be positive, got {arch_cfg.split_ngram_parts}"
        )
    head_dim = arch_cfg.ple_embed_dim // n_heads
    total_rows = sum(compute_ple_ngram_head_vocab_sizes(arch_cfg))
    align = math.lcm(
        max(1, int(arch_cfg.make_ngram_vocab_size_divisible_by)),
        int(arch_cfg.split_ngram_parts),
    )
    padded_rows = ((total_rows + align - 1) // align) * align
    rows_per_shard = padded_rows // arch_cfg.split_ngram_parts
    return (rows_per_shard, head_dim)


def parse_qwen4_5_share_map(
    share_map: Optional[str],
    num_hidden_layers: int,
    layer_types: Sequence[str],
) -> Dict[int, int]:
    """Parse and strictly validate Qwen 4.5-Flash 1-based YoCo ``share_map``.

    Input format is comma-separated ``"<src_1based>-><dst_1based>"`` pairs.
    Returns 0-based ``{consumer_layer_idx: source_layer_idx}``.
    """
    if share_map is None:
        return {}
    if not isinstance(share_map, str):
        raise ValueError(
            f"share_map must be a string or None, got {type(share_map).__name__}"
        )
    stripped = share_map.strip()
    if not stripped:
        return {}
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            f"layer_types length ({len(layer_types)}) != num_hidden_layers ({num_hidden_layers})"
        )

    pairs: Dict[int, int] = {}
    for raw in stripped.split(","):
        item = raw.strip()
        if not item:
            raise ValueError(f"Empty entry in share_map: {share_map!r}")
        parts = item.split("->")
        if len(parts) != 2:
            raise ValueError(f"Malformed share_map pair (expected 'src->dst'): {item!r}")
        src_s, dst_s = parts[0].strip(), parts[1].strip()
        try:
            src_1 = int(src_s)
            dst_1 = int(dst_s)
        except ValueError as exc:
            raise ValueError(f"Non-integer indices in share_map pair: {item!r}") from exc
        if src_1 < 1 or dst_1 < 1:
            raise ValueError(
                f"share_map uses 1-based layer indices (>= 1), got: {item!r}"
            )
        src, dst = src_1 - 1, dst_1 - 1
        if not (0 <= src < dst < num_hidden_layers):
            raise ValueError(
                f"Invalid share_map pair {item!r}: require 1 <= src < dst <= {num_hidden_layers}"
            )
        if dst in pairs:
            raise ValueError(
                f"Duplicate consumer layer {dst_1} (0-based {dst}) in share_map: {item!r}"
            )
        if layer_types[src] != layer_types[dst]:
            raise ValueError(
                f"Cross-type share_map pair {item!r}: layer {src_1} is {layer_types[src]!r} "
                f"but layer {dst_1} is {layer_types[dst]!r}"
            )
        pairs[dst] = src

    overlap = set(pairs.keys()) & set(pairs.values())
    if overlap:
        bad = sorted(idx + 1 for idx in overlap)
        raise ValueError(
            f"Chained sharing is forbidden; layer(s) {bad} cannot be both source and consumer"
        )
    return pairs


def parse_qwen4_5_quant_config(config_dict: Mapping[str, Any]) -> Qwen4_5QuantConfig:
    """Parse and validate ``quantization_config`` from a Qwen 4.5-Flash config dict."""
    raw_qcfg = config_dict.get("quantization_config")
    if raw_qcfg is None:
        return Qwen4_5QuantConfig(
            quant_method=None,
            modules_to_not_convert=(),
            modules_to_convert=(),
            ple_quant_scheme=None,
        )
    if not isinstance(raw_qcfg, Mapping):
        raise ValueError(
            f"quantization_config must be a mapping, got {type(raw_qcfg).__name__}"
        )

    quant_method = raw_qcfg.get("quant_method")
    if quant_method not in VALID_QUANT_METHODS:
        raise ValueError(
            f"Unsupported Qwen 4.5 quant_method {quant_method!r}; "
            f"expected one of {sorted(x for x in VALID_QUANT_METHODS if x is not None)}"
        )

    ple_quant_scheme = raw_qcfg.get("ple_quant_scheme")
    if ple_quant_scheme not in VALID_PLE_QUANT_SCHEMES:
        raise ValueError(
            f"Unsupported Qwen 4.5 ple_quant_scheme {ple_quant_scheme!r}; "
            f"expected one of {sorted(x for x in VALID_PLE_QUANT_SCHEMES if x is not None)}"
        )

    raw_ntc = raw_qcfg.get("modules_to_not_convert") or ()
    if not isinstance(raw_ntc, (list, tuple)) or not all(
        isinstance(m, str) for m in raw_ntc
    ):
        raise ValueError(
            "quantization_config.modules_to_not_convert must be a list of strings"
        )

    raw_conv = raw_qcfg.get("modules_to_convert") or ()
    if not isinstance(raw_conv, (list, tuple)) or not all(
        isinstance(m, str) for m in raw_conv
    ):
        raise ValueError(
            "quantization_config.modules_to_convert must be a list of strings"
        )

    return Qwen4_5QuantConfig(
        quant_method=quant_method,
        modules_to_not_convert=tuple(raw_ntc),
        modules_to_convert=tuple(raw_conv),
        ple_quant_scheme=ple_quant_scheme,
    )


def validate_qwen4_5_arch_config(arch_cfg: Qwen4_5ArchConfig) -> Dict[int, int]:
    """Validate all structural invariants on a ``Qwen4_5ArchConfig`` instance.

    Returns the validated 0-based ``{consumer_layer_idx: source_layer_idx}`` map.
    """
    if arch_cfg.vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {arch_cfg.vocab_size}")
    if arch_cfg.hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {arch_cfg.hidden_size}")
    if arch_cfg.num_hidden_layers <= 0:
        raise ValueError(
            f"num_hidden_layers must be positive, got {arch_cfg.num_hidden_layers}"
        )
    if arch_cfg.intermediate_size <= 0:
        raise ValueError(
            f"intermediate_size must be positive, got {arch_cfg.intermediate_size}"
        )
    if not (0 <= arch_cfg.first_k_dense_replace <= arch_cfg.num_hidden_layers):
        raise ValueError(
            f"first_k_dense_replace ({arch_cfg.first_k_dense_replace}) must be in "
            f"[0, {arch_cfg.num_hidden_layers}]"
        )

    # Layer types
    if len(arch_cfg.layer_types) != arch_cfg.num_hidden_layers:
        raise ValueError(
            f"layer_types has {len(arch_cfg.layer_types)} entries, expected "
            f"{arch_cfg.num_hidden_layers}"
        )
    for idx, ltype in enumerate(arch_cfg.layer_types):
        if ltype not in VALID_LAYER_TYPES:
            raise ValueError(
                f"layer_types[{idx}]={ltype!r} is invalid; expected one of "
                f"{sorted(VALID_LAYER_TYPES)}"
            )

    # YoCo share_map
    sharing = parse_qwen4_5_share_map(
        arch_cfg.share_map,
        arch_cfg.num_hidden_layers,
        arch_cfg.layer_types,
    )

    # Full attention
    if arch_cfg.num_attention_heads <= 0 or arch_cfg.num_key_value_heads <= 0:
        raise ValueError("num_attention_heads and num_key_value_heads must be positive")
    if arch_cfg.num_attention_heads % arch_cfg.num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads ({arch_cfg.num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({arch_cfg.num_key_value_heads})"
        )
    if arch_cfg.head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {arch_cfg.head_dim}")
    if not (0.0 < arch_cfg.partial_rotary_factor <= 1.0):
        raise ValueError(
            f"partial_rotary_factor must be in (0, 1], got {arch_cfg.partial_rotary_factor}"
        )
    if arch_cfg.rope_theta <= 0:
        raise ValueError(f"rope_theta must be positive, got {arch_cfg.rope_theta}")

    # Linear attention (GDN)
    if (
        arch_cfg.linear_num_key_heads <= 0
        or arch_cfg.linear_key_head_dim <= 0
        or arch_cfg.linear_num_value_heads <= 0
        or arch_cfg.linear_value_head_dim <= 0
    ):
        raise ValueError("linear attention head counts and dims must be positive")
    if arch_cfg.linear_num_value_heads % arch_cfg.linear_num_key_heads != 0:
        raise ValueError(
            f"linear_num_value_heads ({arch_cfg.linear_num_value_heads}) must be divisible "
            f"by linear_num_key_heads ({arch_cfg.linear_num_key_heads})"
        )
    if arch_cfg.linear_conv_kernel_dim < 1:
        raise ValueError(
            f"linear_conv_kernel_dim must be >= 1, got {arch_cfg.linear_conv_kernel_dim}"
        )

    # Hyper-Connections
    if arch_cfg.hc_count < 1 or arch_cfg.hc_lowrank <= 0:
        raise ValueError(
            f"hc_count ({arch_cfg.hc_count}) and hc_lowrank ({arch_cfg.hc_lowrank}) must be positive"
        )

    # Latent MoE + Shared Expert + SiTU-GLU
    has_moe_layers = arch_cfg.first_k_dense_replace < arch_cfg.num_hidden_layers
    if has_moe_layers:
        if arch_cfg.num_experts <= 0 or arch_cfg.num_experts_per_tok <= 0:
            raise ValueError("num_experts and num_experts_per_tok must be positive")
        if arch_cfg.num_experts_per_tok > arch_cfg.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({arch_cfg.num_experts_per_tok}) > "
                f"num_experts ({arch_cfg.num_experts})"
            )
        if arch_cfg.routed_expert_hidden_size <= 0 or arch_cfg.moe_intermediate_size <= 0:
            raise ValueError(
                "routed_expert_hidden_size and moe_intermediate_size must be positive"
            )
        if arch_cfg.shared_expert_intermediate_size < 0:
            raise ValueError("shared_expert_intermediate_size must be non-negative")
        if arch_cfg.quant_method == "mxfp4":
            if arch_cfg.routed_expert_hidden_size % MXFP4_GROUP_SIZE != 0:
                raise ValueError(
                    f"MXFP4 requires routed_expert_hidden_size ({arch_cfg.routed_expert_hidden_size}) "
                    f"divisible by {MXFP4_GROUP_SIZE}"
                )
            if arch_cfg.moe_intermediate_size % MXFP4_GROUP_SIZE != 0:
                raise ValueError(
                    f"MXFP4 requires moe_intermediate_size ({arch_cfg.moe_intermediate_size}) "
                    f"divisible by {MXFP4_GROUP_SIZE}"
                )

    if arch_cfg.hidden_act == "situ_glu":
        if arch_cfg.hidden_act_gate_beta <= 0 or arch_cfg.hidden_act_up_beta <= 0:
            raise ValueError("situ_glu gate/up beta must be positive")
    if arch_cfg.clip_mlp_output is not None and arch_cfg.clip_mlp_output <= 0:
        raise ValueError(
            f"clip_mlp_output must be positive or None, got {arch_cfg.clip_mlp_output}"
        )

    # PLE
    if arch_cfg.ple_layer_ids:
        for lid in arch_cfg.ple_layer_ids:
            if not (1 <= lid <= arch_cfg.num_hidden_layers):
                raise ValueError(
                    f"ple_layer_ids entry {lid} out of 1-based range "
                    f"[1, {arch_cfg.num_hidden_layers}]"
                )
        if arch_cfg.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {arch_cfg.ngram_size}")
        if arch_cfg.heads_per_ngram <= 0:
            raise ValueError(
                f"heads_per_ngram must be positive, got {arch_cfg.heads_per_ngram}"
            )
        if arch_cfg.ple_conv_kernel_size < 1:
            raise ValueError(
                f"ple_conv_kernel_size must be >= 1, got {arch_cfg.ple_conv_kernel_size}"
            )
        if arch_cfg.ple_quant_scheme not in VALID_PLE_QUANT_SCHEMES:
            raise ValueError(f"Invalid ple_quant_scheme: {arch_cfg.ple_quant_scheme!r}")
        compute_ple_ngram_shard_shape(arch_cfg)

    return sharing


def parse_qwen4_5_config_json(
    config_dict: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Qwen4_5ArchConfig:
    """Parse and validate a Qwen 4.5-Flash ``config.json`` dictionary.

    Returns a validated ``Qwen4_5ArchConfig`` (from ``contracts.py``).
    """
    if not isinstance(config_dict, Mapping):
        raise ValueError(f"config_dict must be a mapping, got {type(config_dict).__name__}")

    model_type = config_dict.get("model_type")
    if strict or model_type is not None:
        if model_type != EXPECTED_MODEL_TYPE:
            raise ValueError(
                f"Expected model_type={EXPECTED_MODEL_TYPE!r}, got {model_type!r}"
            )

    architectures = config_dict.get("architectures")
    if strict or architectures is not None:
        if not isinstance(architectures, (list, tuple)) or not any(
            arch in SUPPORTED_ARCHITECTURES for arch in architectures
        ):
            raise ValueError(
                f"Expected architectures containing one of {sorted(SUPPORTED_ARCHITECTURES)}, "
                f"got {architectures!r}"
            )

    qcfg = parse_qwen4_5_quant_config(config_dict)

    raw_layer_types = config_dict.get("layer_types")
    num_layers = int(config_dict.get("num_hidden_layers", 40))
    if raw_layer_types is not None:
        if not isinstance(raw_layer_types, (list, tuple)):
            raise ValueError("layer_types must be a list or tuple")
        if len(raw_layer_types) != num_layers:
            raise ValueError(
                f"layer_types has {len(raw_layer_types)} entries, expected {num_layers}"
            )

    arch_cfg = Qwen4_5ArchConfig.from_dict(dict(config_dict))
    if "quantization_config" in config_dict:
        arch_cfg.quant_method = qcfg.quant_method
        if qcfg.ple_quant_scheme is not None:
            arch_cfg.ple_quant_scheme = qcfg.ple_quant_scheme

    validate_qwen4_5_arch_config(arch_cfg)
    return arch_cfg


def load_qwen4_5_config_json(
    path: Union[str, Path],
    *,
    strict: bool = True,
) -> Tuple[Qwen4_5ArchConfig, Qwen4_5QuantConfig]:
    """Load and parse a Qwen 4.5-Flash ``config.json`` file from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    arch_cfg = parse_qwen4_5_config_json(raw, strict=strict)
    quant_cfg = parse_qwen4_5_quant_config(raw)
    return arch_cfg, quant_cfg


def estimate_qwen4_5_params_and_state(
    arch_cfg: Qwen4_5ArchConfig,
    *,
    enable_yoco_decode: Optional[bool] = None,
    kv_dtype_bytes: int = 2,
) -> Qwen4_5ParamAndStateEstimate:
    """Compute exact parameter counts, checkpoint byte size, and per-request state bytes.

    Uses the exact checkpoint tensor enumeration and KV/SSM/ShortConv layout so
    reported totals are exact to the last parameter and byte.
    """
    from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import build_qwen4_5_kv_cache_plan
    from rtp_llm.models.qwen4_5.qwen4_5_weight import (
        expected_checkpoint_tensor_specs,
        summarize_checkpoint_specs,
    )

    use_yoco = (
        arch_cfg.enable_yoco_decode
        if enable_yoco_decode is None
        else bool(enable_yoco_decode)
    )
    kv_dtype = (
        "BF16" if kv_dtype_bytes == 2 else ("FP8" if kv_dtype_bytes == 1 else "F32")
    )

    specs = expected_checkpoint_tensor_specs(arch_cfg)
    summary = summarize_checkpoint_specs(specs, arch_cfg)
    kv_plan = build_qwen4_5_kv_cache_plan(
        arch_cfg,
        enable_yoco_decode=use_yoco,
        kv_dtype=kv_dtype,
    )

    return Qwen4_5ParamAndStateEstimate(
        total_params=summary["logical_params"],
        core_params=summary["core_params"],
        ple_table_params=summary["ple_table_params"],
        embed_and_lm_head_params=summary["embed_and_lm_head_params"],
        active_params_excl_embed=summary["active_params_excl_embed"],
        active_params_incl_embed=(
            summary["active_params_excl_embed"] + summary["embed_and_lm_head_params"]
        ),
        checkpoint_tensor_count=summary["tensor_count"],
        checkpoint_total_bytes=summary["total_bytes"],
        enable_yoco_decode=use_yoco,
        kv_dtype_bytes=kv_dtype_bytes,
        full_attn_kv_layers=kv_plan.num_full_attn_kv_layers,
        kv_bytes_per_token=kv_plan.kv_bytes_per_token,
        gdn_ssm_layers=kv_plan.num_gdn_ssm_layers,
        ssm_bytes_per_request=kv_plan.ssm_bytes_per_req,
        short_conv_state_count=kv_plan.num_short_conv_states,
        short_conv_tensor_count=kv_plan.num_short_conv_tensors,
        short_conv_bytes_per_request=kv_plan.short_conv_bytes_per_req,
        ple_ngram_ctx_bytes_per_request=kv_plan.ple_ngram_ctx_bytes_per_req,
    )


class Qwen4_5ModelConfig(_BaseModelConfig):
    """RTP-LLM ``ModelConfig`` wrapper for Qwen 4.5-Flash (``qwen4_5``).

    When ``rtp_llm.ops`` is available, subclasses ``ModelConfig`` so all RTP-LLM
    engine plumbing works seamlessly. When ``rtp_llm.ops`` is unavailable, falls
    back cleanly to a pure-Python container holding the validated
    ``Qwen4_5ArchConfig`` and ``Qwen4_5QuantConfig``.
    """

    if _HAS_RTP_MODEL_CONFIG:
        _python_fields = _BaseModelConfig._python_fields | {
            "qwen4_5_arch_config",
            "qwen4_5_quant_config",
            "qwen4_5_ple_prefetch_mb",
            "qwen4_5_enable_yoco_decode",
            "qwen4_5_kv_cache_plan",
            "qwen4_5_param_and_state_estimate",
            "qwen4_5_raw_config",
        }

    def __init__(
        self,
        arch_config: Optional[Qwen4_5ArchConfig] = None,
        quant_config: Optional[Qwen4_5QuantConfig] = None,
        *,
        ckpt_path: str = "",
        raw_config: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if _HAS_RTP_MODEL_CONFIG:
            super().__init__(**kwargs)
        self._qwen4_5_arch_config: Optional[Qwen4_5ArchConfig] = arch_config
        self._qwen4_5_quant_config: Optional[Qwen4_5QuantConfig] = quant_config
        self._qwen4_5_raw_config: Dict[str, Any] = dict(raw_config) if raw_config else {}
        self._qwen4_5_ple_prefetch_mb: int = Qwen4_5Options.ple_prefetch_mb
        self._qwen4_5_enable_yoco_decode: bool = (
            arch_config.enable_yoco_decode
            if arch_config is not None
            else Qwen4_5Options.enable_yoco_decode
        )
        if not _HAS_RTP_MODEL_CONFIG:
            self.ckpt_path = ckpt_path
            self.model_type = EXPECTED_MODEL_TYPE
            if arch_config is not None:
                self.vocab_size = arch_config.vocab_size
                self.hidden_size = arch_config.hidden_size
                self.num_layers = arch_config.num_hidden_layers
                self.hc_mult = arch_config.hc_count
                self.expert_num = arch_config.num_experts
                self.moe_k = arch_config.num_experts_per_tok
                self.tie_word_embeddings = arch_config.tie_word_embeddings
        elif ckpt_path:
            self.ckpt_path = ckpt_path

    @classmethod
    def from_dict(
        cls,
        config_dict: Mapping[str, Any],
        *,
        ckpt_path: str = "",
        strict: bool = True,
    ) -> "Qwen4_5ModelConfig":
        arch_cfg = parse_qwen4_5_config_json(config_dict, strict=strict)
        quant_cfg = parse_qwen4_5_quant_config(config_dict)
        return cls(
            arch_config=arch_cfg,
            quant_config=quant_cfg,
            ckpt_path=ckpt_path,
            raw_config=config_dict,
        )

    @classmethod
    def from_json_file(
        cls,
        path: Union[str, Path],
        *,
        strict: bool = True,
    ) -> "Qwen4_5ModelConfig":
        p = Path(path)
        config_path = p / "config.json" if p.is_dir() else p
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(raw, ckpt_path=str(config_path.parent), strict=strict)

    @property
    def arch_config(self) -> Qwen4_5ArchConfig:
        if self._qwen4_5_arch_config is None:
            raise RuntimeError("Qwen4_5ModelConfig.arch_config has not been initialized")
        return self._qwen4_5_arch_config

    @property
    def qwen4_5_arch_config(self) -> Optional[Qwen4_5ArchConfig]:
        return self._qwen4_5_arch_config

    @qwen4_5_arch_config.setter
    def qwen4_5_arch_config(self, value: Qwen4_5ArchConfig) -> None:
        validate_qwen4_5_arch_config(value)
        self._qwen4_5_arch_config = value

    @property
    def qwen4_5_quant_config(self) -> Optional[Qwen4_5QuantConfig]:
        return self._qwen4_5_quant_config

    @qwen4_5_quant_config.setter
    def qwen4_5_quant_config(self, value: Optional[Qwen4_5QuantConfig]) -> None:
        self._qwen4_5_quant_config = value

    @property
    def enable_qwen4_5_ple(self) -> bool:
        return bool(
            self._qwen4_5_arch_config and self._qwen4_5_arch_config.ple_layer_ids
        )

    @property
    def attention_sharing_map(self) -> Dict[int, int]:
        if self._qwen4_5_arch_config is None:
            return {}
        return self._qwen4_5_arch_config.get_attention_sharing()

    @property
    def qwen4_5_ple_prefetch_mb(self) -> int:
        return self._qwen4_5_ple_prefetch_mb

    @qwen4_5_ple_prefetch_mb.setter
    def qwen4_5_ple_prefetch_mb(self, value: int) -> None:
        Qwen4_5Options(
            ple_prefetch_mb=value,
            enable_yoco_decode=self._qwen4_5_enable_yoco_decode,
        )
        self._qwen4_5_ple_prefetch_mb = value

    @property
    def qwen4_5_enable_yoco_decode(self) -> bool:
        return self._qwen4_5_enable_yoco_decode

    @qwen4_5_enable_yoco_decode.setter
    def qwen4_5_enable_yoco_decode(self, value: bool) -> None:
        Qwen4_5Options(
            ple_prefetch_mb=self._qwen4_5_ple_prefetch_mb,
            enable_yoco_decode=value,
        )
        self._qwen4_5_enable_yoco_decode = value
        if self._qwen4_5_arch_config is not None:
            self._qwen4_5_arch_config.enable_yoco_decode = value

    def estimate_params_and_state(
        self,
        *,
        enable_yoco_decode: Optional[bool] = None,
        kv_dtype_bytes: int = 2,
    ) -> Qwen4_5ParamAndStateEstimate:
        return estimate_qwen4_5_params_and_state(
            self.arch_config,
            enable_yoco_decode=(
                self._qwen4_5_enable_yoco_decode
                if enable_yoco_decode is None
                else enable_yoco_decode
            ),
            kv_dtype_bytes=kv_dtype_bytes,
        )

    if _HAS_RTP_MODEL_CONFIG:

        def init_precision_config(
            self, kv_cache_config: Optional[Any], act_type: Optional[str]
        ) -> None:
            qcfg = self._qwen4_5_quant_config
            if qcfg is not None and qcfg.is_mxfp4_routed_experts:
                from rtp_llm.config.model_config import apply_quant_scope_to_pricing
                from rtp_llm.config.quant_config import MXFp4QuarkQuantConfig
                from rtp_llm.ops import KvCacheDataType
                from rtp_llm.utils.weight_type import WEIGHT_TYPE

                gsize = int(getattr(qcfg, "mxfp4_group_size", MXFP4_GROUP_SIZE))
                quant_config = MXFp4QuarkQuantConfig(
                    bits=4,
                    group_size=gsize,
                    is_quanted=True,
                )
                self.quant_algo.setQuantAlgo("mxfp4-quark", 4, gsize)
                apply_quant_scope_to_pricing(self, quant_config)
                self.expert_weight_dtype = "fp4"
                self.dense_weight_dtype = "bf16"

                if kv_cache_config is not None and kv_cache_config.fp8_kv_cache:
                    self.attn_config.kv_cache_dtype = KvCacheDataType.FP8
                else:
                    self.attn_config.kv_cache_dtype = KvCacheDataType.BASE

                dt = WEIGHT_TYPE.from_str(act_type) if act_type else WEIGHT_TYPE.BF16
                self.data_type = dt.to_str()
                self.quant_config = quant_config
                return

            super().init_precision_config(kv_cache_config, act_type)

        def eval_model_weight_size(self) -> float:
            if self._qwen4_5_arch_config is not None:
                est = getattr(self, "qwen4_5_param_and_state_estimate", None)
                if est is None:
                    est = self.estimate_params_and_state()
                return float(est.checkpoint_total_bytes)
            return super().eval_model_weight_size()

        def eval_model_weight_size_per_rank(
            self, tp_size: int, ep_size: int
        ) -> float:
            if tp_size <= 0 or ep_size <= 0:
                raise ValueError(
                    f"tp_size and ep_size must be positive, got {tp_size}, {ep_size}"
                )
            if self._qwen4_5_arch_config is not None:
                est = getattr(self, "qwen4_5_param_and_state_estimate", None)
                if est is None:
                    est = self.estimate_params_and_state()
                total_bytes = float(est.checkpoint_total_bytes)
                eff_ep = max(int(ep_size), int(tp_size))
                if eff_ep <= 1:
                    return total_bytes
                # Routed experts + PLE table are EP-sharded; dense/embed/lm_head (~6%)
                # stay replicated (with lm_head TP-sliced). Keep a small conservative buffer.
                replicated_bytes = max(
                    float(est.embed_and_lm_head_params * 2),
                    total_bytes * 0.06,
                )
                sharded_bytes = max(0.0, total_bytes - replicated_bytes)
                return sharded_bytes / float(eff_ep) + replicated_bytes
            return super().eval_model_weight_size_per_rank(tp_size, ep_size)
