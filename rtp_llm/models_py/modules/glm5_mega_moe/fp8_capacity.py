"""Retain GLM's old FP8 chunk guard only for unverified backend layouts."""

import logging

logger = logging.getLogger(__name__)


def glm53_fp8_capacity(requested: int, experts: int, ranks: int, topk: int) -> int:
    legacy_limit = 640 * experts // (ranks * topk)
    if requested <= legacy_limit:
        return requested

    import deep_gemm

    candidates_fn = getattr(deep_gemm, "get_block_m_candidates_for_mega_moe_fp8", None)
    alignment_fn = getattr(
        getattr(deep_gemm, "_C", None), "get_token_alignment_for_mega_moe_fp8", None
    )
    if callable(candidates_fn) and callable(alignment_fn):
        candidates = candidates_fn()
        alignment = alignment_fn()
        if (
            alignment > 0
            and candidates
            and all(b > 0 and alignment % b == 0 for b in candidates)
        ):
            return requested

    logger.warning(
        "GLM53 FP8 MegaMoE: backend has no verified ring-aligned BLOCK_M candidates; "
        "limiting each call to %d tokens (requested %d)",
        legacy_limit,
        requested,
    )
    return legacy_limit
