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
    if not supported_speculation:
        raise ValueError(
            "FP4 indexer supports SP0 or DSpark with three generation slots"
        )
    if reuse_cache:
        raise ValueError("FP4 indexer requires prefix reuse disabled")
    from rtp_llm.models.dsv4_kv_cache import resolve_dsv4_tokens_per_block

    # Match ModelConfig's inherited kernel size and DeepSeekV4's promotion of
    # the framework allocation default before checking the physical geometry.
    kernel_tokens_per_block = kernel_tokens_per_block or tokens_per_block
    promoted = resolve_dsv4_tokens_per_block(tokens_per_block)
    if promoted is not None:
        if kernel_tokens_per_block == tokens_per_block:
            kernel_tokens_per_block = promoted
        tokens_per_block = promoted
    if tokens_per_block != 256 or kernel_tokens_per_block != 256:
        raise ValueError(
            "Experimental FP4 indexer requires 256-token allocation and kernel blocks"
        )
