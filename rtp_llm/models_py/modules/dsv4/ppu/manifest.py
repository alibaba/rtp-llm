"""Configuration gates for the DeepSeek V4 Flash PPU implementation."""

import os
from typing import NamedTuple


class SupportResult(NamedTuple):
    supported: bool
    reason: str = ""


def _check_execution_options(options, required):
    for key, expected in required.items():
        if options.get(key) != expected:
            return SupportResult(False, f"requires {key}={expected}")
    return SupportResult(True)


DECODE_ATTN_PROFILES = ("auto", "large_ep_full")

# Kernel-level switches replaced by DSV4_PPU_DECODE_ATTN_PROFILE. Fail closed so
# a stale deployment cannot silently lose the tiles it used to request.
_REMOVED_ATTN_SWITCHES = ("DSV4_PPU_WOA_SMALL_TILE", "DSV4_PPU_HCA_SPLIT")


def _resolve_decode_attn_profile(options, parallelism):
    for removed in _REMOVED_ATTN_SWITCHES:
        if removed in options:
            raise ValueError(
                f"{removed} was removed; set "
                "DSV4_PPU_DECODE_ATTN_PROFILE=large_ep_full instead"
            )
    profile = options.get("DSV4_PPU_DECODE_ATTN_PROFILE", "auto")
    if profile not in DECODE_ATTN_PROFILES:
        raise ValueError(
            "DSV4_PPU_DECODE_ATTN_PROFILE must be one of "
            f"{DECODE_ATTN_PROFILES}, got {profile!r}"
        )
    if profile != "auto" and (
        parallelism.ep_size not in (16, 32) or parallelism.tp_size != 1
    ):
        raise ValueError(
            "DSV4_PPU_DECODE_ATTN_PROFILE=large_ep_full requires EP16/32 with TP1, "
            f"got EP{parallelism.ep_size}/TP{parallelism.tp_size}"
        )
    return profile


def validate_cache_config(kv_cache_config):
    if (kv_cache_config is not None and kv_cache_config.reuse_cache
            and os.environ.get("DSV4_PPU_PREFIX_REUSE") != "1"):
        raise ValueError("PPU DSV4 prefix reuse requires experimental DSV4_PPU_PREFIX_REUSE=1")


DRAFT_LAYERS = {"deepseek_v4_mtp": 1, "deepseek_v4_dspark": 3}


def build_provider(model_config, parallelism, moe_config, max_generate_batch_size):
    from rtp_llm.ops import KvCacheDataType

    role = parallelism.role_type.name
    draft_depth = DRAFT_LAYERS.get(model_config.model_type)
    draft = draft_depth is not None
    if (
        model_config.hidden_size != 4096
        or model_config.model_type not in ("deepseek_v4",) + tuple(DRAFT_LAYERS)
        or model_config.num_layers != (draft_depth if draft else 43)
        or (
            draft
            and list(model_config.attn_config.layer_compress_ratios) != [0] * draft_depth
        )
    ):
        raise ValueError(
            "PPU supports DeepSeek V4 Flash, its single SWA MTP layer and the "
            "three SWA DSpARK draft layers"
        )
    if role not in ("PREFILL", "DECODE"):
        raise ValueError(f"PPU DSV4 requires separate Prefill and Decode roles, got {role}")
    if parallelism.pp_size != 1 or parallelism.prefill_cp_config.is_enabled():
        raise ValueError("PPU DSV4 requires PP1 with CP disabled")
    if parallelism.world_size != parallelism.tp_size * parallelism.dp_size:
        raise ValueError("PPU DSV4 requires world_size=TP*DP")
    if parallelism.ep_size not in (1, parallelism.world_size):
        raise ValueError("PPU DSV4 requires EP1 or EP=world_size")
    if parallelism.ep_size == 1 and parallelism.dp_size != 1:
        raise ValueError("PPU DSV4 EP1 requires DP1")
    if role == "DECODE" and parallelism.ep_size == 1:
        raise ValueError("PPU DSV4 Decode requires EP=world_size>1")
    if model_config.attn_config.kv_cache_dtype != KvCacheDataType.FP8:
        raise ValueError("PPU DSV4 requires FP8 KV and FP4 indexer cache")
    if (model_config.attn_config.kernel_tokens_per_block != 256
            or model_config.attn_config.tokens_per_block != 256):
        raise ValueError("PPU DSV4 requires matching 256-token physical and kernel blocks")
    # GEN_NUM_PER_CYCLE is the runtime proposal width and the target carries it
    # too (for its gamma+1 verify span), so the gate keys off the checkpoint:
    # a DSpARK checkpoint declares its own width, plain MTP stays at 1..3.
    dspark = model_config.model_type == "deepseek_v4_dspark" or bool(
        getattr(model_config, "dspark_target_layer_ids", None)
    )
    max_depth = 8 if dspark else 3
    if not 0 <= model_config.gen_num_per_cycle <= max_depth:
        raise ValueError(
            f"PPU DSV4 supports {model_config.model_type} speculative depth "
            f"0 through {max_depth}, got {model_config.gen_num_per_cycle}"
        )
    if getattr(model_config, "lora_infos", None) or model_config.eplb_config.enable_eplb():
        raise ValueError("PPU DSV4 LoRA and EPLB are not qualified")
    if parallelism.ep_size > 1 or role == "DECODE":
        if (
            not moe_config.use_deepep_moe
            or moe_config.use_deepep_low_latency != (role == "DECODE")
            or moe_config.use_deepep_internode
            or moe_config.use_all_gather
        ):
            raise ValueError("PPU DSV4 requires single-node engine-owned DeepEP for this role")
    if role == "DECODE" and not 0 < max_generate_batch_size <= 128:
        raise ValueError("PPU DSV4 Decode batch capacity must be in 1..128")
    required = DECODE_EXECUTION_OPTIONS if role == "DECODE" else PREFILL_EXECUTION_OPTIONS
    options = dict(required)
    options.update({key: value for key, value in os.environ.items() if key.startswith("DSV4_")})
    support = _check_execution_options(options, required)
    if not support.supported:
        raise ValueError(support.reason)
    if role == "DECODE":
        from rtp_llm.models_py.modules.dsv4.ppu.ppu_decode_provider import PpuDecodeProvider

        attn_profile = _resolve_decode_attn_profile(options, parallelism)
        return PpuDecodeProvider(
            options,
            ep_size=parallelism.ep_size,
            tp_size=parallelism.tp_size,
            attn_profile=attn_profile,
        )
    from rtp_llm.models_py.modules.dsv4.ppu.ppu_module_provider import PpuModuleProvider

    return PpuModuleProvider(options)

PREFILL_EXECUTION_OPTIONS = {
    "DSV4_HC_IMPL": "hybrid",
    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic",
    "DSV4_MHC_POST_BACKEND": "tilelang",
    "DSV4_PPU_SGLANG_MOE": "1",
    "DSV4_PPU_SGLANG_WO_A": "1",
    "DSV4_MOE_SHARED_EXPERT_OVERLAP": "0",
    "DSV4_PPU_TP_COMM_WARMUP": "1",
    "DSV4_PPU_SHARED_QKV_QUANT": "0",
    "DSV4_MOE_SCALE_GATHER_FUSED": "1",
}

DECODE_EXECUTION_OPTIONS = {
    "DSV4_HC_IMPL": "hybrid",
    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic",
    "DSV4_MHC_POST_BACKEND": "tilelang",
    "DSV4_PPU_SGLANG_MOE": "1",
    "DSV4_PPU_SGLANG_WO_A": "1",
    "DSV4_MOE_SHARED_EXPERT_OVERLAP": "0",
    "DSV4_PPU_TP_COMM_WARMUP": "0",
    "DSV4_PPU_DECODE_HC_REDUCTION": "fused",
    "DSV4_PPU_DECODE_HC_NORM": "fused",
    "DSV4_PPU_DECODE_FP8_QUANT": "v2",
    "DSV4_PPU_DECODE_QKV": "merged",
    "DSV4_PPU_DECODE_INDEXER": "overlap",
    "DSV4_PPU_DECODE_ATTN_MODE": "overlap",
    "DSV4_SHARED_EXPERT_MODE": "overlap",
    "DSV4_PPU_DECODE_SHARED_SCHEDULE": "before_route",
    "DSV4_PPU_DECODE_MOE_OUTPUT": "bf16",
    "DSV4_PPU_DECODE_METADATA": "reference",
    "DSV4_PPU_DECODE_ROPE": "reference",
    "DSV4_PPU_DECODE_MOE_HINT": "capacity",
}
