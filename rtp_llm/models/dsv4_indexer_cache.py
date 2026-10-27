"""Explicit DSv4 indexer representation and supported execution geometry."""

import os


def fp4_indexer_enabled() -> bool:
    mode = os.environ.get("DSV4_INDEXER_CACHE_MODE", "fp8")
    if mode not in ("fp8", "fp4"):
        raise ValueError(f"DSV4_INDEXER_CACHE_MODE must be fp8 or fp4, got {mode!r}")
    return mode == "fp4"


def indexer_entry_bytes() -> int:
    return 68 if fp4_indexer_enabled() else 132


def validate_fp4_indexer_config(
    *,
    model_type,
    speculative_type,
    generation_slots,
    reuse_cache,
    tokens_per_block,
    kernel_tokens_per_block,
) -> None:
    if not fp4_indexer_enabled():
        return
    from rtp_llm.ops import SpeculativeType

    if model_type != "deepseek_v4":
        raise ValueError("FP4 indexer is only supported for deepseek_v4")
    supported_speculation = (
        speculative_type == SpeculativeType.NONE and generation_slots == 0
    ) or (speculative_type == SpeculativeType.DSPARK and generation_slots == 3)
    # sam/mtp-gen1-20261022: allow traditional MTP with one generation slot
    # (pp1 local-PD experiment); FP4 indexer numerics need GSM8K validation.
    supported_speculation = supported_speculation or (
        speculative_type == SpeculativeType.MTP and generation_slots in (1, 3)
    )
    if not supported_speculation:
        raise ValueError(
            "FP4 indexer supports SP0, DSpark with three generation slots, "
            "or MTP with one generation slot"
        )
    if (
        reuse_cache
        and os.environ.get("DSV4_FP4_INDEXER_ALLOW_PREFIX_REUSE", "0") != "1"
    ):
        raise ValueError("FP4 indexer requires prefix reuse disabled")
    if tokens_per_block != 256 or kernel_tokens_per_block != 256:
        raise ValueError(
            "Experimental FP4 indexer requires 256-token allocation and kernel blocks"
        )
    if os.environ.get("RTP_LLM_RELEASE_PROFILE", "").strip():
        raise ValueError(
            "Experimental FP4 indexer is not qualified by an FP8 release profile"
        )
