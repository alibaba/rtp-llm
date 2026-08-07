"""
DeepSeek MLA/MoE language models for newloader.

Top-level model: DeepSeekV32ForCausalLM
  - load_weights() applies WEIGHTS_MAPPER (prefix_mapping={"model.": ""})
    then delegates to RtpModule's streaming dispatch via super().
  - __init__ builds all submodules with HF-compatible names.
  - process_weights_after_loading() fuses QKV projections and KV cache
    projections (kc/vc from kv_b).
  - RoPE cos/sin cache is computed on-the-fly from config.
"""

import json
import logging
import math
import os
from collections.abc import Callable
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.new_models.model_base import select_block_map_for_layer
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops import MlaOpsType
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

from .model import DeepSeekV32DecoderLayer
from .moe import normalize_topk_method
from .rotary_embedding import DeepseekV3RotaryEmbedding, DeepseekV3YarnRotaryEmbedding

logger = logging.getLogger(__name__)


class MlaKernelWeightLayout:
    """Minimal W.* layout consumed by MLA kernels.

    The tensors remain owned by the new-loader modules.  This object only
    provides the legacy internal names required by the attention kernels; it
    is not involved in checkpoint mapping or weight loading.
    """

    def __init__(
        self,
        layer_weights: list[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
    ) -> None:
        self.weights = layer_weights
        self._cos_sin_cache = cos_sin_cache

    def get_global_weight_or_none(self, name: str) -> Optional[torch.Tensor]:
        if name == W.rope_cos_sin_cache:
            return self._cos_sin_cache
        return None


def build_mla_runtime_layout(
    layers: nn.ModuleList,
    cos_sin_cache: torch.Tensor,
) -> MlaKernelWeightLayout:
    """Capture MLA runtime tensor views without mutating checkpoint storage."""
    attentions = [layer.self_attn for layer in layers]
    return MlaKernelWeightLayout(
        [attention._build_mla_kernel_weights() for attention in attentions],
        cos_sin_cache,
    )


class MlaRuntimeLayoutMixin:
    """Shared MLA runtime-layout lifecycle for score and MTP models."""

    def _apply(self, fn, recurse: bool = True):
        source_cos_sin_cache = self.cos_sin_cache
        result = super()._apply(fn, recurse)
        # The score/MTP model and every sparse IndexerOp intentionally share
        # one RoPE cache. RtpModule restores aliases after recursively applying
        # ``fn``; because the top-level registration is the alias master, a
        # model-wide ``to(dtype=...)`` can otherwise overwrite the IndexerOp's
        # local FP32 correction with a lower-precision cache. Re-establish the
        # kernel contract once, after alias restoration, and bind every
        # consumer to that canonical tensor.
        cos_sin_cache = self.cos_sin_cache
        if cos_sin_cache.dtype != torch.float32:
            # Use the pre-conversion FP32 values and only adopt the target
            # device. Casting the already-lowered tensor back to FP32 would
            # preserve its dtype but irreversibly keep BF16/FP16 rounding.
            cos_sin_cache = source_cos_sin_cache.to(
                device=cos_sin_cache.device,
                dtype=torch.float32,
            )
        self.cos_sin_cache = cos_sin_cache
        for layer in self.layers:
            indexer = layer.self_attn.indexer
            if indexer is not None:
                indexer.bind_rope_cache(cos_sin_cache)
        if self._mla_kernel_layout is not None:
            # The runtime layout is a lightweight non-Module view, so refresh
            # all of its tensor references after a post-initialize migration.
            self._mla_kernel_layout = build_mla_runtime_layout(
                self.layers,
                cos_sin_cache,
            )
        return result

    def runtime_weight_view(self) -> Dict[str, torch.Tensor]:
        return {
            "embedding": self.embed_tokens.weight,
            "final_layernorm.gamma": self.norm.weight,
            "lm_head": self.lm_head.weight,
        }

    def initialize(self, init_resource):
        ok = super().initialize(init_resource)
        if not ok:
            return ok
        self._ensure_mla_kernel_layout()
        if self._keep_mla_checkpoint_weights:
            logger.info(
                "Keeping DeepSeek MLA checkpoint-only weights for debugging; "
                "GPU memory usage will be higher"
            )
        else:
            for layer in self.layers:
                layer.self_attn.release_checkpoint_only_weights()
        return ok

    def _ensure_mla_kernel_layout(self) -> None:
        if self._mla_kernel_layout is None:
            self._mla_kernel_layout = build_mla_runtime_layout(
                self.layers,
                self.cos_sin_cache,
            )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        self._ensure_mla_kernel_layout()
        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self._mla_kernel_layout,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )


# ------------------------------------------------------------------ #
#  RoPE helpers (mirrors DeepSeekV2._create_rope_w)
# ------------------------------------------------------------------ #

_YARN_REQUIRED_KEYS = (
    "factor",
    "original_max_position_embeddings",
    "mscale",
    "mscale_all_dim",
)
_YARN_TYPES = {"yarn", "deepseek_yarn"}


def _rope_type(config: dict) -> Optional[str]:
    rope_type = config.get("rope_type", config.get("type"))
    if rope_type is None:
        return None
    if not isinstance(rope_type, str):
        raise TypeError(f"DeepSeek rope_type must be a string, got {rope_type!r}")
    return rope_type.strip().lower()


def _resolve_yarn_parameters(config_json: dict) -> Optional[dict]:
    """Resolve legacy rope_scaling and Transformers rope_parameters safely."""
    rope_scaling = config_json.get("rope_scaling")
    rope_parameters = config_json.get("rope_parameters", {})
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise TypeError("DeepSeek rope_scaling must be a dictionary")
    if not isinstance(rope_parameters, dict):
        raise TypeError("DeepSeek rope_parameters must be a dictionary")

    parameter_type = _rope_type(rope_parameters)
    parameter_has_scaling = parameter_type in _YARN_TYPES or any(
        key in rope_parameters for key in _YARN_REQUIRED_KEYS
    )
    if (
        parameter_has_scaling
        and parameter_type is not None
        and parameter_type not in _YARN_TYPES
    ):
        raise ValueError(
            f"unsupported DeepSeek rope_parameters rope_type={parameter_type!r}"
        )

    if rope_scaling is None:
        yarn = rope_parameters if parameter_has_scaling else None
    else:
        scaling_type = _rope_type(rope_scaling)
        if scaling_type is not None and scaling_type not in _YARN_TYPES:
            raise ValueError(
                f"unsupported DeepSeek rope_scaling rope_type={scaling_type!r}"
            )
        yarn = rope_scaling
        if parameter_has_scaling:
            for key in _YARN_REQUIRED_KEYS:
                if (
                    key in rope_scaling
                    and key in rope_parameters
                    and rope_scaling[key] != rope_parameters[key]
                ):
                    raise ValueError(
                        "conflicting DeepSeek RoPE scaling values for "
                        f"{key}: rope_scaling={rope_scaling[key]!r}, "
                        f"rope_parameters={rope_parameters[key]!r}"
                    )

    if yarn is None:
        return None
    missing = [key for key in _YARN_REQUIRED_KEYS if key not in yarn]
    if missing:
        raise ValueError(f"DeepSeek YaRN configuration is missing keys: {missing}")
    factor = yarn["factor"]
    original_max = yarn["original_max_position_embeddings"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(float(factor))
        or factor <= 0
    ):
        raise ValueError(f"DeepSeek YaRN factor must be positive, got {factor!r}")
    for name in ("mscale", "mscale_all_dim"):
        value = yarn[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"DeepSeek YaRN {name} must be non-negative, got {value!r}"
            )
    if (
        isinstance(original_max, bool)
        or not isinstance(original_max, int)
        or original_max <= 0
    ):
        raise ValueError(
            "DeepSeek YaRN original_max_position_embeddings must be a "
            f"positive integer, got {original_max!r}"
        )
    return yarn


def build_rope_cache(
    config_json: dict,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cos/sin cache from config.json + max_seq_len.

    Returns [cos|sin] concatenated, shape [max_seq_len, rope_head_dim], float32.
    """
    rope_parameters = config_json.get("rope_parameters", {})
    if not isinstance(rope_parameters, dict):
        raise TypeError("DeepSeek rope_parameters must be a dictionary")
    rope_theta = rope_parameters.get(
        "rope_theta", config_json.get("rope_theta", 10000.0)
    )
    if (
        isinstance(rope_theta, bool)
        or not isinstance(rope_theta, (int, float))
        or rope_theta <= 0
    ):
        raise ValueError(f"DeepSeek rope_theta must be positive, got {rope_theta!r}")
    rope_head_dim = config_json.get(
        "qk_rope_head_dim", config_json.get("rope_head_dim")
    )
    if (
        isinstance(rope_head_dim, bool)
        or not isinstance(rope_head_dim, int)
        or rope_head_dim <= 0
        or rope_head_dim % 2
    ):
        raise ValueError(
            "DeepSeek RoPE head dimension must be a positive even integer, "
            f"got {rope_head_dim!r}"
        )

    yarn_parameters = _resolve_yarn_parameters(config_json)

    if yarn_parameters is None:
        rotary_emb = DeepseekV3RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            device=device,
        )
    else:
        # Match the legacy loader's CPU sin/cos evaluation exactly, then move
        # the finished cache to CUDA before the indexer captures its reference.
        rotary_emb = DeepseekV3YarnRotaryEmbedding(
            rope_head_dim,
            max_seq_len,
            rope_theta,
            scaling_factor=yarn_parameters["factor"],
            original_max_position_embeddings=yarn_parameters[
                "original_max_position_embeddings"
            ],
            beta_fast=float(yarn_parameters.get("beta_fast", 32)),
            beta_slow=float(yarn_parameters.get("beta_slow", 1)),
            mscale=yarn_parameters["mscale"],
            mscale_all_dim=yarn_parameters["mscale_all_dim"],
        )

    half_rope_dim = rope_head_dim // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    return (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )


def read_config_json(ckpt_path: str) -> Dict[str, Any]:
    """Read config.json from ckpt path, return empty dict if not found."""
    if not ckpt_path:
        return {}
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    return payload


def checkpoint_path(model_config: Any) -> str:
    value = (
        model_config.get("ckpt_path", "")
        if isinstance(model_config, dict)
        else getattr(model_config, "ckpt_path", "")
    )
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"model_config.ckpt_path must be a string, got {value!r}")
    return value


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _positive_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


def _bool_value(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {value!r}")
    return value


def _partition(load_config: Any, prefix: str) -> tuple[int, int]:
    size = getattr(load_config, f"{prefix}_size", None)
    rank = getattr(load_config, f"{prefix}_rank", None)
    size = positive_int(size, f"{prefix}_size")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise ValueError(f"invalid {prefix} partition: rank={rank!r}, size={size}")
    return size, rank


# ------------------------------------------------------------------ #
#  Config extraction (mirrors DeepSeekV2._from_hf)
# ------------------------------------------------------------------ #


def extract_config_values(
    model_config: Any,
    load_config: Any,
    config_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read config from either ModelConfig (C++ pybind) or HF dict.

    config_json is the raw config.json dict; used to resolve fields that
    the old-loader's _create_config overwrites on model_config (e.g.
    inter_size -> n_shared_experts * moe_intermediate_size).
    """

    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    eplb_config = _get(model_config, "eplb_config", None)
    enable_eplb = _get(eplb_config, "enable_eplb", False)
    if callable(enable_eplb):
        enable_eplb = enable_eplb()
    if not isinstance(enable_eplb, bool):
        raise TypeError("eplb_config.enable_eplb must be a bool")
    if enable_eplb:
        raise ValueError("EPLB is not supported by the DeepSeek newloader path")

    hidden_size = positive_int(_get(model_config, "hidden_size", 7168), "hidden_size")
    num_layers = positive_int(
        _get(
            model_config,
            "num_layers",
            _get(model_config, "num_hidden_layers", 4),
        ),
        "num_layers",
    )
    vocab_size = positive_int(_get(model_config, "vocab_size", 102400), "vocab_size")
    max_seq_len = positive_int(_get(model_config, "max_seq_len", 8192), "max_seq_len")

    # Attention config
    attn_config = _get(model_config, "attn_config", None)
    if attn_config is not None:
        num_heads = _get(attn_config, "head_num", 128)
        q_lora_rank = _get(attn_config, "q_lora_rank", 1536)
        kv_lora_rank = _get(attn_config, "kv_lora_rank", 512)
        nope_head_dim = _get(attn_config, "nope_head_dim", 128)
        rope_head_dim = _get(attn_config, "rope_head_dim", 64)
        v_head_dim = _get(attn_config, "v_head_dim", 128)
        is_sparse = _get(attn_config, "is_sparse", False)
        indexer_head_dim = _get(attn_config, "indexer_head_dim", 128)
        indexer_head_num = _get(attn_config, "indexer_head_num", 64)
        indexer_topk = _get(attn_config, "indexer_topk", 2048)
    else:
        num_heads = _get(model_config, "num_attention_heads", 128)
        q_lora_rank = _get(model_config, "q_lora_rank", 1536)
        kv_lora_rank = _get(model_config, "kv_lora_rank", 512)
        nope_head_dim = _get(model_config, "qk_nope_head_dim", 128)
        rope_head_dim = _get(model_config, "qk_rope_head_dim", 64)
        v_head_dim = _get(model_config, "v_head_dim", 128)
        is_sparse = _get(model_config, "is_sparse", False)
        indexer_head_dim = _get(model_config, "index_head_dim", 128)
        indexer_head_num = _get(model_config, "index_n_heads", 64)
        indexer_topk = _get(model_config, "index_topk", 2048)

    if attn_config is not None:
        use_mla = _bool_value(
            _get(attn_config, "use_mla", False),
            "attn_config.use_mla",
        )
        mla_ops_type = _get(model_config, "mla_ops_type", None)
        if not use_mla or mla_ops_type == MlaOpsType.MHA:
            raise ValueError(
                "DeepSeek newloader requires an MLA attention backend; "
                "the legacy expanded-MHA fallback is not supported"
            )

    rms_norm_eps = _positive_float(
        _get(
            model_config,
            "rms_norm_eps",
            _get(model_config, "layernorm_eps", 1e-6),
        ),
        "rms_norm_eps",
    )

    # MoE config
    num_experts = _get(
        model_config, "expert_num", _get(model_config, "n_routed_experts", 256)
    )
    top_k = _get(model_config, "moe_k", _get(model_config, "num_experts_per_tok", 8))
    # ModelConfig.moe_inter_size defaults to 0 and the legacy DeepSeek
    # loader never overwrites it — reading model_config first would yield
    # 0, breaking the expert buffer shapes. Prefer config.json's
    # moe_intermediate_size, fall through to the default only if absent.
    moe_intermediate_size = None
    if config_json:
        moe_intermediate_size = config_json.get("moe_intermediate_size")
    if not moe_intermediate_size:
        mc_moe = _get(model_config, "moe_inter_size", 0) or _get(
            model_config, "moe_intermediate_size", 0
        )
        moe_intermediate_size = mc_moe or 2048
    n_shared_experts = _get(model_config, "n_shared_experts", 1)
    if config_json:
        n_shared_experts = config_json.get("n_shared_experts", n_shared_experts)
    n_shared_experts = nonnegative_int(n_shared_experts, "n_shared_experts")
    shared_expert_intermediate_size = _get(
        model_config,
        "shared_expert_intermediate_size",
        None,
    )
    if shared_expert_intermediate_size in (None, 0) and config_json:
        shared_expert_intermediate_size = config_json.get(
            "shared_expert_intermediate_size"
        )
    if shared_expert_intermediate_size in (None, 0):
        shared_expert_intermediate_size = n_shared_experts * moe_intermediate_size

    # NOTE: model_config.inter_size is overridden by the old-loader to
    # n_shared_experts * moe_intermediate_size.  Read the real dense FFN
    # width from config_json (HF ckpt) when available.
    dense_intermediate_size = None
    if config_json:
        dense_intermediate_size = config_json.get("intermediate_size")
    if dense_intermediate_size is None:
        dense_intermediate_size = _get(
            model_config,
            "intermediate_size",
            _get(model_config, "inter_size", 18432),
        )

    # first_k_dense_replace / moe_layer_freq are not propagated onto
    # ModelConfig by the legacy loader either — read from config.json,
    # falling back only as a last resort. Wrong values here mismap dense
    # vs MoE layers and routes dense ckpt tensors into MoEBlock.
    first_k_dense_replace = None
    moe_layer_freq = None
    if config_json:
        first_k_dense_replace = config_json.get("first_k_dense_replace")
        moe_layer_freq = config_json.get("moe_layer_freq")
    if first_k_dense_replace is None:
        first_k_dense_replace = _get(model_config, "first_k_dense_replace", 1)
    if moe_layer_freq is None:
        moe_layer_freq = _get(model_config, "moe_layer_freq", 1)
    first_k_dense_replace = nonnegative_int(
        first_k_dense_replace, "first_k_dense_replace"
    )
    moe_layer_freq = positive_int(moe_layer_freq, "moe_layer_freq")
    moe_layer_index = [
        i
        for i in range(num_layers)
        if i >= first_k_dense_replace and i % moe_layer_freq == 0
    ]

    scoring_func = _get(model_config, "scoring_func", 1)  # 0=softmax, 1=sigmoid
    routed_scaling_factor = _get(model_config, "routed_scaling_factor", 1.0)
    n_group = _get(model_config, "moe_n_group", _get(model_config, "n_group", None))
    topk_group = _get(
        model_config, "moe_topk_group", _get(model_config, "topk_group", None)
    )
    has_moe_norm = _get(
        model_config, "has_moe_norm", _get(model_config, "norm_topk_prob", None)
    )
    if config_json:
        raw_scoring_func = config_json.get("scoring_func")
        if raw_scoring_func is not None:
            if raw_scoring_func == "softmax":
                scoring_func = 0
            elif raw_scoring_func == "sigmoid":
                scoring_func = 1
            else:
                raise ValueError(f"unsupported scoring_func={raw_scoring_func!r}")
        n_group = config_json.get("n_group", n_group)
        topk_group = config_json.get("topk_group", topk_group)
        has_moe_norm = config_json.get("norm_topk_prob", has_moe_norm)
    n_group = n_group if n_group is not None else 1
    topk_group = topk_group if topk_group is not None else 1
    has_moe_norm = has_moe_norm if has_moe_norm is not None else False
    has_moe_norm = _bool_value(has_moe_norm, "norm_topk_prob")
    if config_json:
        routed_scaling_factor = config_json.get(
            "routed_scaling_factor", routed_scaling_factor
        )
    topk_method = config_json.get("topk_method", "greedy") if config_json else "greedy"
    topk_method = normalize_topk_method(topk_method)
    # has_e_score_correction is not a ModelConfig field — the legacy loader
    # detects it from ckpt key presence on the weight class side. Derive it
    # here from config.json's topk_method ("noaux_tc" => correction bias).
    has_e_score_correction = _get(model_config, "has_e_score_correction", False)
    if not has_e_score_correction and config_json:
        has_e_score_correction = topk_method == "noaux_tc"
    has_e_score_correction = _bool_value(
        has_e_score_correction, "has_e_score_correction"
    )

    # Only the sparse indexer consumes an interleave flag. MLA RoPE itself is
    # applied by the backend and does not read a model-level is_neox_style.
    indexer_rope_interleave = _get(model_config, "indexer_rope_interleave", None)
    if indexer_rope_interleave is None and config_json:
        indexer_rope_interleave = config_json.get("indexer_rope_interleave", False)
    if indexer_rope_interleave is None:
        indexer_rope_interleave = False
    indexer_rope_interleave = _bool_value(
        indexer_rope_interleave, "indexer_rope_interleave"
    )
    indexer_is_neox_style = not indexer_rope_interleave

    # Parallelism. Attention may be replicated under context parallelism while
    # FFN weights remain sharded across the physical TP group.
    parallelism_config = getattr(load_config, "parallelism_config", None)
    attn_tp_size, attn_tp_rank = _partition(load_config, "attn_tp")
    ffn_tp_size, ffn_tp_rank = _partition(load_config, "ffn_tp")
    lm_head_tp_size, lm_head_tp_rank = _partition(load_config, "lm_head_tp")
    ep_size, ep_rank = _partition(load_config, "ep")
    quant_config = getattr(load_config, "quant_config", None)
    params_dtype = getattr(load_config, "compute_dtype", torch.bfloat16)
    if not isinstance(params_dtype, torch.dtype):
        raise TypeError(f"compute_dtype must be torch.dtype, got {params_dtype!r}")
    enable_fp32_lm_head = _bool_value(
        _get(model_config, "enable_fp32_lm_head", True),
        "enable_fp32_lm_head",
    )
    model_tie_word_embeddings = _bool_value(
        _get(model_config, "tie_word_embeddings", False),
        "model_config.tie_word_embeddings",
    )
    checkpoint_tie_word_embeddings = _bool_value(
        (config_json or {}).get("tie_word_embeddings", False),
        "config.json tie_word_embeddings",
    )
    tie_word_embeddings = model_tie_word_embeddings or checkpoint_tie_word_embeddings
    if tie_word_embeddings and (
        attn_tp_size != lm_head_tp_size or attn_tp_rank != lm_head_tp_rank
    ):
        raise ValueError(
            "tied DeepSeek embeddings require matching attention and LM-head "
            "TP partitions"
        )
    moe_config = getattr(load_config, "moe_config", None)

    # Kernel tokens per block
    blocksize = _get(model_config, "kernel_tokens_per_block", 64)
    if attn_config is not None:
        blocksize = _get(attn_config, "kernel_tokens_per_block", blocksize)
    blocksize = positive_int(blocksize, "kernel_tokens_per_block")

    q_lora_rank = nonnegative_int(q_lora_rank, "q_lora_rank")
    kv_lora_rank = positive_int(kv_lora_rank, "kv_lora_rank")
    num_heads = positive_int(num_heads, "num_attention_heads")
    nope_head_dim = positive_int(nope_head_dim, "qk_nope_head_dim")
    rope_head_dim = positive_int(rope_head_dim, "qk_rope_head_dim")
    v_head_dim = positive_int(v_head_dim, "v_head_dim")
    if rope_head_dim % 2:
        raise ValueError("qk_rope_head_dim must be even")
    if num_heads % attn_tp_size:
        raise ValueError(
            f"num_attention_heads={num_heads} must be divisible by "
            f"attn_tp_size={attn_tp_size}"
        )
    is_sparse = _bool_value(is_sparse, "is_sparse")
    dense_intermediate_size = positive_int(dense_intermediate_size, "intermediate_size")
    moe_intermediate_size = positive_int(moe_intermediate_size, "moe_intermediate_size")
    shared_expert_intermediate_size = nonnegative_int(
        shared_expert_intermediate_size, "shared_expert_intermediate_size"
    )
    num_experts = nonnegative_int(num_experts, "num_experts")
    top_k = nonnegative_int(top_k, "num_experts_per_tok")
    if moe_layer_index:
        if num_experts == 0:
            raise ValueError("DeepSeek MoE layers require a positive expert count")
        if not 0 < top_k <= num_experts:
            raise ValueError(
                f"num_experts_per_tok={top_k} must be in [1, {num_experts}]"
            )
        if num_experts % ep_size:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by ep_size={ep_size}"
            )
    if is_sparse:
        if q_lora_rank == 0:
            raise ValueError("DeepSeek sparse indexer requires q_lora_rank > 0")
        indexer_head_dim = positive_int(indexer_head_dim, "index_head_dim")
        indexer_head_num = positive_int(indexer_head_num, "index_n_heads")
        indexer_topk = positive_int(indexer_topk, "index_topk")
    scoring_func = nonnegative_int(scoring_func, "scoring_func")
    if scoring_func not in (0, 1):
        raise ValueError(f"unsupported scoring_func={scoring_func}")
    routed_scaling_factor = _positive_float(
        routed_scaling_factor, "routed_scaling_factor"
    )
    n_group = positive_int(n_group, "n_group")
    topk_group = positive_int(topk_group, "topk_group")
    if topk_group > n_group:
        raise ValueError(f"topk_group={topk_group} exceeds n_group={n_group}")
    if moe_layer_index and topk_method in {"group_limited_greedy", "noaux_tc"}:
        if num_experts % n_group:
            raise ValueError(
                f"num_experts={num_experts} must be divisible by " f"n_group={n_group}"
            )
        selected_expert_capacity = topk_group * (num_experts // n_group)
        if top_k > selected_expert_capacity:
            raise ValueError(
                f"num_experts_per_tok={top_k} exceeds grouped routing "
                f"capacity={selected_expert_capacity}"
            )

    return dict(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        v_head_dim=v_head_dim,
        rms_norm_eps=rms_norm_eps,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        dense_intermediate_size=dense_intermediate_size,
        moe_layer_index=moe_layer_index,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        n_group=n_group,
        topk_group=topk_group,
        topk_method=topk_method,
        has_moe_norm=has_moe_norm,
        has_e_score_correction=has_e_score_correction,
        is_sparse=is_sparse,
        indexer_head_dim=indexer_head_dim,
        indexer_head_num=indexer_head_num,
        indexer_topk=indexer_topk,
        indexer_is_neox_style=indexer_is_neox_style,
        blocksize=blocksize,
        tp_size=attn_tp_size,
        tp_rank=attn_tp_rank,
        attn_tp_size=attn_tp_size,
        attn_tp_rank=attn_tp_rank,
        ffn_tp_size=ffn_tp_size,
        ffn_tp_rank=ffn_tp_rank,
        lm_head_tp_size=lm_head_tp_size,
        lm_head_tp_rank=lm_head_tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        quant_config=quant_config,
        params_dtype=params_dtype,
        lm_head_params_dtype=torch.float32 if enable_fp32_lm_head else params_dtype,
        tie_word_embeddings=tie_word_embeddings,
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class DeepSeekV32ForCausalLM(MlaRuntimeLayoutMixin, GptModelBase):
    """Config-driven DeepSeek MLA/MoE implementation for newloader.

    WEIGHTS_MAPPER only strips "model." prefix.  All submodule names match
    HF ckpt keys directly, so RtpModule.load_weights can dispatch weights
    without any fusion-time mapping.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    @staticmethod
    def _checkpoint_layer_index(name: str) -> Optional[int]:
        prefix = "model.layers."
        if not name.startswith(prefix):
            return None
        layer_text, separator, _ = name[len(prefix) :].partition(".")
        if not separator or not layer_text.isdigit():
            return None
        return int(layer_text)

    def checkpoint_weight_name_filter(self) -> Callable[[str], bool]:
        num_layers = len(self.layers)

        def should_load(name: str) -> bool:
            if name.startswith("model.layers."):
                layer_idx = self._checkpoint_layer_index(name)
                if layer_idx is None:
                    return True
                return layer_idx < num_layers
            return name.startswith(("model.", "lm_head."))

        return should_load

    @staticmethod
    def _read_config_json(ckpt_path: str) -> Dict[str, Any]:
        """Read config.json from ckpt path.

        Overridable by subclasses (e.g. DeepSeek VL V2 merges the nested
        ``language_config`` section into the top-level dict so that
        _extract_config_values can find MLA / MoE fields).
        """
        return read_config_json(ckpt_path)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        has_lm_head = False

        def _track(it):
            nonlocal has_lm_head
            for name, tensor in it:
                if name.startswith(("lm_head.", "model.lm_head.")):
                    has_lm_head = True
                yield name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(_track(weights_iter))
        super().load_weights(mapped_iter)

        if not has_lm_head and self.tie_word_embeddings:
            logger.info(
                "[DeepSeekV32] lm_head.weight not found in ckpt; "
                "tying lm_head to embed_tokens"
            )
            self.lm_head._copy_local_tied_weight(self.embed_tokens.weight.data)

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
    ):
        parallelism_config = load_config.parallelism_config
        fmha_config = load_config.fmha_config
        device_resource_config = load_config.device_resource_config

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )
        self._keep_mla_checkpoint_weights = load_config.keep_mla_checkpoint_weights
        self._mla_kernel_layout: Optional[MlaKernelWeightLayout] = None

        ckpt_path = checkpoint_path(model_config)

        # Read config.json early — _extract_config_values needs it to
        # resolve fields that old-loader's _create_config overwrites on
        # model_config (e.g. inter_size).
        # _read_config_json is overridable so multimodal variants (e.g.
        # DeepSeek VL V2 can merge nested sub-configs into the top level.
        config_json = self._read_config_json(ckpt_path)
        if not config_json:
            raise FileNotFoundError(
                "DeepSeek newloader requires checkpoint config.json to preserve "
                "MLA, RoPE, MoE, and dense-layer topology"
            )

        cfg = extract_config_values(model_config, load_config, config_json)
        self.tie_word_embeddings = cfg["tie_word_embeddings"]

        # --- RoPE cache: read config.json directly for full rope fields ---
        device = torch.device(load_config.device)
        cos_sin_cache = build_rope_cache(config_json, cfg["max_seq_len"], device)
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        # --- Embedding ---
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        # --- Decoder layers ---
        moe_layer_set = set(cfg["moe_layer_index"])
        self.layers = nn.ModuleList()

        for i in range(cfg["num_layers"]):
            is_moe = i in moe_layer_set
            layer = DeepSeekV32DecoderLayer(
                hidden_size=cfg["hidden_size"],
                num_heads=cfg["num_heads"],
                q_lora_rank=cfg["q_lora_rank"],
                kv_lora_rank=cfg["kv_lora_rank"],
                nope_head_dim=cfg["nope_head_dim"],
                rope_head_dim=cfg["rope_head_dim"],
                v_head_dim=cfg["v_head_dim"],
                layer_idx=i,
                attn_tp_size=cfg["attn_tp_size"],
                attn_tp_rank=cfg["attn_tp_rank"],
                ffn_tp_size=cfg["ffn_tp_size"],
                ffn_tp_rank=cfg["ffn_tp_rank"],
                ep_size=cfg["ep_size"],
                ep_rank=cfg["ep_rank"],
                params_dtype=cfg["params_dtype"],
                layernorm_eps=cfg["rms_norm_eps"],
                quant_config=cfg["quant_config"],
                model_config=cfg["model_config"],
                parallelism_config=cfg["parallelism_config"],
                moe_config=cfg["moe_config"],
                is_moe_layer=is_moe,
                dense_intermediate_size=cfg["dense_intermediate_size"],
                moe_intermediate_size=cfg["moe_intermediate_size"],
                num_experts=cfg["num_experts"],
                top_k=cfg["top_k"],
                shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
                has_shared_expert=cfg["shared_expert_intermediate_size"] > 0,
                scoring_func=cfg["scoring_func"],
                routed_scaling_factor=cfg["routed_scaling_factor"],
                n_group=cfg["n_group"],
                topk_group=cfg["topk_group"],
                topk_method=cfg["topk_method"],
                has_moe_norm=cfg["has_moe_norm"],
                correction_bias=cfg["has_e_score_correction"],
                is_sparse=cfg["is_sparse"],
                index_n_heads=cfg["indexer_head_num"],
                index_head_dim=cfg["indexer_head_dim"],
                index_topk=cfg["indexer_topk"],
                indexer_is_neox_style=cfg["indexer_is_neox_style"],
                cos_sin_cache=cos_sin_cache,
                blocksize=cfg["blocksize"],
                prefix=f"layers.{i}",
            )
            self.layers.append(layer)

        # --- Final norm ---
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

        # --- LM head ---
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        if inputs.attention_inputs is None:
            raise ValueError("DeepSeek forward requires attention_inputs")
        input_ids = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        residual = torch.zeros_like(hidden_states)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
