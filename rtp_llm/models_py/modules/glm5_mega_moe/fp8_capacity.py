"""Retain GLM's old FP8 chunk guard only for unverified backend layouts."""

import logging
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

# The hash-pinned public wheel predates the Python query, but its bundled
# mega_moe_fp8.cuh defines this exact candidate set and a 1920-token ring.
_PUBLIC_DEEPGEMM_VERSION = "2.8.2+522942a.cu132"
_PUBLIC_BLOCK_M_CANDIDATES = (8, 16, 32, 64, 96, 128, 192, 240)


def _candidate_block_ms(deep_gemm):
    query = getattr(deep_gemm, "get_block_m_candidates_for_mega_moe_fp8", None)
    if callable(query):
        return query()
    try:
        if version("deep-gemm") == _PUBLIC_DEEPGEMM_VERSION:
            return _PUBLIC_BLOCK_M_CANDIDATES
    except PackageNotFoundError:
        pass
    return ()


def glm53_fp8_capacity(requested: int, experts: int, ranks: int, topk: int) -> int:
    legacy_limit = 640 * experts // (ranks * topk)
    if requested <= legacy_limit:
        return requested

    import deep_gemm

    alignment_fn = getattr(
        getattr(deep_gemm, "_C", None), "get_token_alignment_for_mega_moe_fp8", None
    )
    if callable(alignment_fn):
        candidates = _candidate_block_ms(deep_gemm)
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
