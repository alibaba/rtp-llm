"""Immutable Qwen4 options, checkpoint facts and final deployment validation.

The snapshot is built after common overrides and the effective topology are
known. Target and draft build independent snapshots; pickle preserves them.
"""

from dataclasses import dataclass

from rtp_llm.config.quant_config import (
    QWEN4_FP8_NVFP4_RECIPE,
    QWEN4_NVFP4_RECIPE,
    Fp8BlockWiseQuantConfig,
    ModelOptFp4Config,
    init_quant_config,
)
from rtp_llm.ops import DataType, KvCacheDataType


@dataclass(frozen=True)
class Qwen4Options:
    ple_prefetch_mb: int

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
    ffn_sp_size: int
    role: str
    target_verify_width: int
    kv_precision: str


def freeze_runtime_config(config, parallelism):
    snapshot = Qwen4RuntimeSnapshot(
        Qwen4Options(config.qwen4_ple_prefetch_mb),
        CheckpointGeometry.from_config(config),
        (
            "NVFP4"
            if getattr(config, "_qwen4_nvfp4_source", False)
            else "FP8" if getattr(config, "_qwen4_fp8_source", False) else "BF16"
        ),
        (
            getattr(config.quant_config, "model_recipe", None)
            or (config.quant_config.get_method() if config.quant_config else None)
        ),
        int(parallelism.get_attn_tp_size()),
        bool(config.is_mtp),
        (
            int(parallelism.get_ffn_tp_size())
            if hasattr(parallelism, "get_ffn_tp_size")
            else 1
        ),
        int(getattr(parallelism, "ep_size", 1)),
        int(getattr(parallelism, "ffn_sp_size", 1)),
        str(getattr(parallelism, "role_type", "PDFUSION")),
        int(getattr(config, "gen_num_per_cycle", 0)) + 1,
        str(
            getattr(
                getattr(config, "attn_config", None),
                "kv_cache_dtype",
                KvCacheDataType.BASE,
            )
        ),
    )
    previous = getattr(config, "_qwen4_runtime_snapshot", None)
    if previous is not None and previous != snapshot:
        raise RuntimeError(
            "qwen4_exp configuration changed after the runtime snapshot was frozen"
        )
    config._qwen4_runtime_snapshot = snapshot
    return snapshot


def build_qwen4_quant_config(quantization, source_fp8, source_nvfp4=False):
    requested = init_quant_config(quantization) if quantization else None
    if source_nvfp4:
        if (
            source_fp8
            or requested is not None
            and (
                not isinstance(requested, ModelOptFp4Config)
                or requested.model_recipe != QWEN4_NVFP4_RECIPE
                or not requested.is_quanted()
                or requested.bits != 4
                or requested.group_size() != 16
                or requested.source_fp8_block_size is not None
                or requested.activation_amax is not None
                or requested.online_cache_dir is not None
                or requested.mixed_attention
                or not requested.require_contiguous_experts
            )
        ):
            raise ValueError("qwen4_exp offline NVFP4 requires the QWEN4_NVFP4 recipe")
        return ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=True,
            model_recipe=QWEN4_NVFP4_RECIPE,
            require_contiguous_experts=True,
        )
    full_fp8 = bool(source_fp8)
    # The official checkpoint already carries routed E4M3 + block128 scales.
    # Preserve that format by default, including on SM90 where NVFP4 is absent.
    if full_fp8 and (
        requested is None or isinstance(requested, Fp8BlockWiseQuantConfig)
    ):
        if requested is not None and (
            requested.bits != 8 or requested.group_size() != 128
        ):
            raise ValueError("qwen4_exp FP8 recipe requires 8-bit 128x128 blocks")
        return Fp8BlockWiseQuantConfig(is_quanted=True)
    if requested is not None and (
        not isinstance(requested, ModelOptFp4Config)
        or requested.model_recipe != QWEN4_FP8_NVFP4_RECIPE
        or not full_fp8
        or requested.bits != 4
        or requested.group_size() != 16
        or requested.is_quanted()
        or requested.source_fp8_block_size != 128
        or requested.activation_amax != 32.0
        or requested.mixed_attention
        or not requested.require_contiguous_experts
    ):
        raise ValueError(
            "qwen4_exp supports BF16 source or an official FP8 recipe (FP8_PER_BLOCK/QWEN4_FP8_NVFP4)"
        )
    if full_fp8:
        return ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=False,
            activation_amax=32.0,
            online_cache_dir=getattr(requested, "online_cache_dir", None),
            source_fp8_block_size=128,
            require_contiguous_experts=True,
            model_recipe=QWEN4_FP8_NVFP4_RECIPE,
        )
    else:
        return None


def validate_qwen4_runtime(config, parallelism):
    if (
        config is None
        or not config.enable_qwen4_qsa
        or (not config.is_mtp and not config.enable_qwen4_ple)
    ):
        raise RuntimeError("qwen4_exp requires complete checkpoint PLE/QSA geometry")
    full_fp8 = bool(getattr(config, "_qwen4_full_fp8_nvfp4", False))
    offline_nvfp4 = bool(getattr(config, "_qwen4_nvfp4_source", False))
    quant = getattr(config, "quant_config", None)
    if offline_nvfp4:
        if (
            not isinstance(quant, ModelOptFp4Config)
            or quant.model_recipe != QWEN4_NVFP4_RECIPE
            or not quant.is_quanted()
            or quant.bits != 4
            or quant.group_size() != 16
            or full_fp8
            or getattr(config, "_qwen4_full_fp8", False)
        ):
            raise RuntimeError("qwen4_exp requires the offline NVFP4 precision recipe")
        if config.is_mtp or int(getattr(config, "gen_num_per_cycle", 0)) > 0:
            raise RuntimeError(
                "qwen4_exp offline NVFP4 currently requires MTP disabled"
            )
        pc = parallelism
        tp = int(pc.tp_size)
        if (
            tp not in (1, 2)
            or pc.world_size != tp
            or pc.ep_size != 1
            or pc.dp_size != 1
            or pc.get_ffn_tp_size() != tp
            or pc.ffn_sp_size != 1
            or pc.ffn_disaggregate_config.enable_ffn_disaggregate
            or config.eplb_config.enable_eplb()
        ):
            raise RuntimeError(
                "qwen4_exp offline NVFP4 requires TP1/TP2, EP1, DP1 without EPLB"
            )
        if config.moe_inter_size % (16 * tp) or config.hidden_size % 16:
            raise RuntimeError(
                "qwen4_exp offline NVFP4 TP shards must preserve 16-element blocks"
            )
    if full_fp8 and (
        not isinstance(quant, ModelOptFp4Config)
        or quant.model_recipe != QWEN4_FP8_NVFP4_RECIPE
        or quant.is_quanted()
        or quant.bits != 4
        or quant.group_size() != 16
        or quant.source_fp8_block_size != 128
    ):
        raise RuntimeError("qwen4_exp requires the complete FP8/NVFP4 precision recipe")
    native_fp8 = bool(getattr(config, "_qwen4_full_fp8", False)) and not full_fp8
    if native_fp8 and (
        not isinstance(quant, Fp8BlockWiseQuantConfig)
        or not quant.is_quanted()
        or quant.bits != 8
        or quant.group_size() != 128
        or not getattr(config, "_qwen4_fp8_source", False)
    ):
        raise RuntimeError("qwen4_exp requires native official block128 FP8 weights")
    if not (full_fp8 or native_fp8 or offline_nvfp4) and quant is not None:
        raise RuntimeError(
            "qwen4_exp BF16 execution requires an unquantized checkpoint"
        )
    if getattr(config, "enable_qwen4_qsa", False):
        gamma = int(getattr(config, "gen_num_per_cycle", 0))
        ratio = int(getattr(config, "_qwen4_indexer_compress_ratio", 0))
        if gamma > 0 and ratio > 0 and gamma + 1 > ratio:
            raise RuntimeError(
                "qwen4_exp QSA target verify requires gen_num_per_cycle + 1 "
                f"<= indexer_compress_ratio; got {gamma} + 1 > {ratio}"
            )
        if getattr(config, "data_type", None) != DataType.TYPE_BF16:
            raise RuntimeError(
                "qwen4_exp QSA currently requires act_type=BF16; refusing "
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
                "qwen4_exp QSA does not support context/prefill parallelism; "
                "refusing before indexer weights are loaded"
            )
    if getattr(config, "enable_qwen4_ple", False) and (
        getattr(config, "data_type", None) != DataType.TYPE_BF16
    ):
        raise RuntimeError(
            "qwen4_exp PLE currently requires act_type=BF16; refusing to "
            "load its rank-local n-gram table for an incompatible dtype"
        )
    if getattr(config, "enable_qwen4_ple", False):
        cp = getattr(parallelism, "prefill_cp_config", None)
        if cp is not None and (cp.is_enabled() or cp.is_prefill_enabled()):
            raise RuntimeError(
                "qwen4_exp PLE does not support context/prefill parallelism; "
                "refusing before its rank-local n-gram table is loaded"
            )
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
