"""Immutable Qwen4 options, checkpoint facts and final deployment validation.

The snapshot is built after common overrides and the effective topology are
known. Target and draft build independent snapshots; pickle preserves them.
"""

import json
from copy import deepcopy
from dataclasses import dataclass

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import (
    QWEN4_FP8_NVFP4_RECIPE,
    ModelOptFp4Config,
    init_quant_config,
)
from rtp_llm.ops import DataType, KvCacheDataType


def ple_eos_token_id(config):
    """Resolve the checkpoint's PLE document boundary, not the chat stop ID."""
    eos = getattr(config, "_qwen4_ple_eos_token_id", None)
    if eos is None:
        eos = getattr(getattr(config, "special_tokens", None), "eos_token_id", None)
    return eos


class Qwen4ModelConfig(ModelConfig):
    _python_fields = ModelConfig._python_fields | {
        "qwen4_ple_prefetch_mb",
        "linear_attn_norm_activation",
    }

    # Checkpoint capabilities are read-only; they are not model options.
    @property
    def enable_qwen4_ple(self) -> bool:
        return bool(getattr(self, "_qwen4_ple_layer_ids", ()))

    @property
    def enable_qwen4_qsa(self) -> bool:
        return int(getattr(self, "_qwen4_indexer_head_dim", 0)) > 0

    @property
    def qwen4_ple_prefetch_mb(self) -> int:
        snapshot = getattr(self, "_qwen4_runtime_snapshot", None)
        if snapshot is not None:
            return snapshot.options.ple_prefetch_mb
        return getattr(self, "_qwen4_ple_prefetch_mb", Qwen4Options.ple_prefetch_mb)

    @qwen4_ple_prefetch_mb.setter
    def qwen4_ple_prefetch_mb(self, value: int) -> None:
        self._validate_ple_budget(value)
        self._qwen4_ple_prefetch_mb = value

    def _validate_ple_budget(self, value: int) -> None:
        options = Qwen4Options(value)
        snapshot = getattr(self, "_qwen4_runtime_snapshot", None)
        if snapshot is not None and options != snapshot.options:
            raise RuntimeError(
                "qwen4_exp resource options are frozen after model loading"
            )

    def apply_override_args(self, json_model_override_args):
        model_override_args = json.loads(json_model_override_args)
        if "qwen4_ple_prefetch_mb" in model_override_args:
            self._validate_ple_budget(model_override_args["qwen4_ple_prefetch_mb"])
        super().apply_override_args(json_model_override_args)
        if "qwen4_ple_prefetch_mb" in model_override_args:
            self.qwen4_ple_prefetch_mb = model_override_args["qwen4_ple_prefetch_mb"]


@dataclass(frozen=True)
class Qwen4Options:
    ple_prefetch_mb: int = 256

    def __post_init__(self):
        if type(self.ple_prefetch_mb) is not int or self.ple_prefetch_mb <= 0:
            raise ValueError("qwen4_ple_prefetch_mb must be a positive integer")


@dataclass(frozen=True)
class IndexerGeometry:
    head_dim: int
    compress_ratio: int
    budget: int
    kv_heads: int

    @classmethod
    def from_config(cls, config):
        return cls(
            int(getattr(config, "_qwen4_indexer_head_dim", 0)),
            int(getattr(config, "_qwen4_indexer_compress_ratio", 0)),
            int(getattr(config, "_qwen4_indexer_budget", 0)),
            int(getattr(config, "_qwen4_indexer_kv_heads", 0)),
        )


@dataclass(frozen=True)
class CheckpointGeometry:
    hidden_size: int
    hc_mult: int
    ple_layer_ids: tuple[int, ...]
    split_ngram_parts: int
    indexer_head_dim: int
    indexer_compress_ratio: int
    indexer_budget: int
    indexer_kv_heads: int

    def __post_init__(self):
        if self.hidden_size <= 0 or self.hc_mult <= 0:
            raise ValueError(
                "qwen4_exp checkpoint hidden_size/hc_count must be positive"
            )
        if self.indexer_head_dim > 0 and (
            self.indexer_compress_ratio <= 0
            or self.indexer_budget <= 0
            or self.indexer_kv_heads <= 0
        ):
            raise ValueError("qwen4_exp checkpoint indexer geometry must be positive")
        if self.ple_layer_ids and self.split_ngram_parts <= 0:
            raise ValueError("qwen4_exp checkpoint PLE shard count must be positive")

    @classmethod
    def from_config(cls, config):
        return cls(
            int(config.hidden_size),
            int(config.hc_mult),
            tuple(int(x) for x in getattr(config, "_qwen4_ple_layer_ids", ())),
            int(getattr(config, "_qwen4_split_ngram_parts", 0)),
            int(getattr(config, "_qwen4_indexer_head_dim", 0)),
            int(getattr(config, "_qwen4_indexer_compress_ratio", 0)),
            int(getattr(config, "_qwen4_indexer_budget", 0)),
            int(getattr(config, "_qwen4_indexer_kv_heads", 0)),
        )


@dataclass(frozen=True)
class Qwen4RuntimeSnapshot:
    options: Qwen4Options
    geometry: CheckpointGeometry
    source_format: str
    model_recipe: str | None
    attention_tp: int
    is_mtp_draft: bool
    ffn_tp: int
    ep_size: int
    role: str
    target_verify_width: int
    kv_precision: str
    dense_storage: str
    vocab_storage: str
    mtp_dense_storage: str


def freeze_runtime_config(config, parallelism):
    snapshot = Qwen4RuntimeSnapshot(
        Qwen4Options(config.qwen4_ple_prefetch_mb),
        CheckpointGeometry.from_config(config),
        (
            "NVFP4"
            if getattr(config, "_qwen4_nvfp4_source", False)
            else "FP8" if getattr(config, "_qwen4_fp8_source", False) else "BF16"
        ),
        getattr(config.quant_config, "model_recipe", None),
        int(parallelism.get_attn_tp_size()),
        bool(config.is_mtp),
        (
            int(parallelism.get_ffn_tp_size())
            if hasattr(parallelism, "get_ffn_tp_size")
            else 1
        ),
        int(getattr(parallelism, "ep_size", 1)),
        str(getattr(parallelism, "role_type", "PDFUSION")),
        int(getattr(config, "gen_num_per_cycle", 0)) + 1,
        str(
            getattr(
                getattr(config, "attn_config", None),
                "kv_cache_dtype",
                KvCacheDataType.BASE,
            )
        ),
        getattr(config, "_qwen4_dense_storage", "fp8"),
        getattr(
            config,
            "_qwen4_vocab_storage",
            getattr(config, "_qwen4_dense_storage", "fp8"),
        ),
        getattr(config, "_qwen4_mtp_dense_storage_mode", "inherit"),
    )
    previous = getattr(config, "_qwen4_runtime_snapshot", None)
    if previous is not None and previous != snapshot:
        raise RuntimeError(
            "qwen4_exp configuration changed after the runtime snapshot was frozen"
        )
    config._qwen4_runtime_snapshot = snapshot
    return snapshot


def is_qwen4_precision_recipe(quant):
    return (
        isinstance(quant, ModelOptFp4Config)
        and quant.model_recipe == QWEN4_FP8_NVFP4_RECIPE
        and quant.bits == 4
        and quant.group_size() == 16
        and not quant.is_quanted()
        and quant.source_fp8_block_size == 128
        and quant.activation_amax == 32.0
        and not quant.mixed_attention
        and quant.require_contiguous_experts
    )


def resolve_qwen4_parallelism(config, parallelism):
    """Give loading and execution the same model-owned NVFP4 layout."""
    resolved = deepcopy(parallelism)
    # ParallelismConfig's pickle state omits local_rank. Preserve the selected
    # device when making a model-owned copy instead of restoring rank zero.
    resolved.local_rank = parallelism.local_rank
    if not is_qwen4_precision_recipe(
        getattr(config, "quant_config", None)
    ) and not getattr(config, "_qwen4_nvfp4_source", False):
        return resolved
    cp = resolved.prefill_cp_config
    if (
        resolved.dp_size != 1
        or resolved.pp_size != 1
        or resolved.world_size != resolved.tp_size
        or cp.is_enabled()
        or cp.is_prefill_enabled()
    ):
        raise ValueError("Qwen4 NVFP4 requires pure TP or pure EP without DP/PP/CP")
    if resolved.ep_size == 1 and resolved.tp_size in (1, 2):
        resolved.ffn_tp_size = resolved.tp_size
        resolved.ffn_tp_rank = resolved.tp_rank
    elif resolved.ep_size == resolved.tp_size and resolved.tp_size in (2, 8):
        # Each rank owns complete local experts and a replicated shared expert.
        resolved.ffn_tp_size = 1
        resolved.ffn_tp_rank = 0
    else:
        raise ValueError("Qwen4 NVFP4 supports TP1, TP2/EP1, and pure EP2/EP8")
    return resolved


def build_qwen4_quant_config(quantization, source_fp8):
    requested = init_quant_config(quantization) if quantization else None
    full_fp8 = bool(source_fp8)
    if requested is not None and (
        not full_fp8 or not is_qwen4_precision_recipe(requested)
    ):
        raise ValueError("qwen4_exp supports BF16 source or the QWEN4_FP8_NVFP4 recipe")
    if full_fp8:
        dense_storage = getattr(requested, "dense_storage", "fp8")
        mtp_dense_storage = getattr(requested, "mtp_dense_storage", "inherit")
        if dense_storage not in ("fp8", "source_bf16"):
            raise ValueError("Qwen4 dense_storage must be fp8 or source_bf16")
        if mtp_dense_storage not in (
            "inherit",
            "source_bf16",
            "source_bf16_shared_vocab",
        ):
            raise ValueError(
                "Qwen4 mtp_dense_storage must be inherit, source_bf16 or source_bf16_shared_vocab"
            )
        return ModelOptFp4Config(
            dense_storage=dense_storage,
            mtp_dense_storage=mtp_dense_storage,
            bits=4,
            group_size=16,
            is_quanted=False,
            activation_amax=32.0,
            online_cache_dir=getattr(requested, "online_cache_dir", None),
            prefill_library=getattr(requested, "prefill_library", None),
            prefill_library_sha256=getattr(requested, "prefill_library_sha256", None),
            source_fp8_block_size=128,
            require_contiguous_experts=True,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
        )
    else:
        return None


def validate_qwen4_runtime(config, parallelism):
    import torch

    if torch.version.hip is not None:
        raise RuntimeError("qwen4_exp requires NVIDIA CUDA for PLE/QSA")
    if (
        config is None
        or not config.enable_qwen4_qsa
        or (not config.is_mtp and not config.enable_qwen4_ple)
    ):
        raise RuntimeError("qwen4_exp requires complete checkpoint PLE/QSA geometry")
    full_fp8 = bool(getattr(config, "_qwen4_full_fp8_nvfp4", False))
    quant = getattr(config, "quant_config", None)
    if full_fp8 and getattr(config, "tie_word_embeddings", False):
        raise RuntimeError("qwen4_exp FP8/NVFP4 requires untied word embeddings")
    if full_fp8 and not is_qwen4_precision_recipe(quant):
        raise RuntimeError("qwen4_exp requires the complete FP8/NVFP4 precision recipe")
    if full_fp8 and int(getattr(parallelism, "ep_size", 1)) > int(
        parallelism.get_attn_tp_size()
    ):
        raise RuntimeError(
            "qwen4_exp FP8/NVFP4 requires ep_size <= attention TP; "
            "rejecting before PLE weight loading"
        )
    source_nvfp4 = bool(getattr(config, "_qwen4_nvfp4_source", False))
    if source_nvfp4 and not (
        isinstance(quant, ModelOptFp4Config)
        and quant.is_quanted()
        and quant.bits == 4
        and quant.group_size() == 16
    ):
        raise RuntimeError("Qwen4 NVFP4 source requires prequantized group-16 experts")
    if not full_fp8 and not source_nvfp4 and quant is not None:
        raise RuntimeError(
            "qwen4_exp BF16 execution requires an unquantized checkpoint"
        )
    gamma = int(getattr(config, "gen_num_per_cycle", 0))
    ratio = int(getattr(config, "_qwen4_indexer_compress_ratio", 0))
    if gamma > 0 and ratio > 0 and gamma + 1 > ratio:
        raise RuntimeError(
            "qwen4_exp QSA target verify requires gen_num_per_cycle + 1 "
            f"<= indexer_compress_ratio; got {gamma} + 1 > {ratio}"
        )
    if getattr(config, "data_type", None) != DataType.TYPE_BF16:
        raise RuntimeError(
            "qwen4_exp PLE/QSA currently requires act_type=BF16; refusing "
            "before indexer weights are loaded"
        )
    attn_config = getattr(config, "attn_config", None)
    if not bool(getattr(attn_config, "is_sparse", False)) or not bool(
        getattr(attn_config, "use_sparse_gqa_fmha", False)
    ):
        raise RuntimeError(
            "qwen4_exp QSA routing flags are inconsistent; dense fallback "
            "is forbidden"
        )
    kv_cache_dtype = getattr(attn_config, "kv_cache_dtype", KvCacheDataType.BASE)
    if kv_cache_dtype not in (KvCacheDataType.BASE, KvCacheDataType.FP8):
        raise RuntimeError("qwen4_exp QSA requires BF16 or scaled FP8 KV cache")
    cp = getattr(parallelism, "prefill_cp_config", None)
    if cp is not None and (cp.is_enabled() or cp.is_prefill_enabled()):
        raise RuntimeError(
            "qwen4_exp PLE/QSA does not support context/prefill parallelism; "
            "refusing before indexer weights are loaded"
        )
    if getattr(config, "enable_qwen4_ple", False):
        # Match ModelDeployWeightInfo/LoadConfig exactly. In particular,
        # ALL_GATHER context parallelism changes attention TP to one; using
        # raw tp_size here would let the 102-GB table start loading before
        # its descriptor rejected the effective topology.
        tp_size = int(parallelism.get_attn_tp_size()) if parallelism is not None else 0
        shard_count = int(getattr(config, "_qwen4_split_ngram_parts", 0))
        if tp_size < 1 or shard_count <= 0 or shard_count % tp_size:
            raise RuntimeError(
                "qwen4_exp PLE requires effective attention TP >= 1 and an "
                "even rank-local shard split, got "
                f"split_ngram_parts={shard_count}, attn_tp_size={tp_size}"
            )
