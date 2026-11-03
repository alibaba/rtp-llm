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


def validate_cache_config(kv_cache_config):
    if (kv_cache_config is not None and kv_cache_config.reuse_cache
            and os.environ.get("DSV4_PPU_PREFIX_REUSE") != "1"):
        raise ValueError("PPU DSV4 prefix reuse requires experimental DSV4_PPU_PREFIX_REUSE=1")


def build_provider(model_config, parallelism, moe_config, max_generate_batch_size):
    from rtp_llm.ops import KvCacheDataType

    role = parallelism.role_type.name
    draft = model_config.model_type == "deepseek_v4_mtp"
    if (
        model_config.hidden_size != 4096
        or model_config.num_layers != (1 if draft else 43)
        or model_config.model_type not in ("deepseek_v4", "deepseek_v4_mtp")
        or (draft and list(model_config.attn_config.layer_compress_ratios) != [0])
    ):
        raise ValueError("PPU supports DeepSeek V4 Flash and its single SWA MTP layer")
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
    if model_config.attn_config.kv_cache_dtype != KvCacheDataType.FP8:
        raise ValueError("PPU DSV4 requires FP8 KV and FP4 indexer cache")
    if model_config.attn_config.kernel_tokens_per_block != 256:
        raise ValueError("PPU DSV4 requires 256 tokens per kernel block")
    if model_config.gen_num_per_cycle not in (0, 1, 2, 3):
        raise ValueError("PPU DSV4 supports MTP depths 1 through 3")
    if getattr(model_config, "lora_infos", None) or model_config.eplb_config.enable_eplb():
        raise ValueError("PPU DSV4 LoRA and EPLB are not qualified")
    if parallelism.ep_size > 1 or role == "DECODE":
        if (
            not moe_config.use_deepep_moe
            or moe_config.use_deepep_low_latency != (role == "DECODE")
            or moe_config.use_deepep_internode
            or moe_config.use_all_gather
            or parallelism.ffn_disaggregate_config.enable_ffn_disaggregate
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
        from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider

        return PpuDecodeProvider(options)
    from rtp_llm.platforms.ppu.models.dsv4.ppu_module_provider import PpuModuleProvider

    return PpuModuleProvider(options)

PREFILL_EXECUTION_OPTIONS = {
    "DSV4_HC_IMPL": "hybrid",
    "DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic",
    "DSV4_MHC_POST_BACKEND": "tilelang",
    "DSV4_MHC_POST_PDL": "0",
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
    "DSV4_MHC_POST_PDL": "0",
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
