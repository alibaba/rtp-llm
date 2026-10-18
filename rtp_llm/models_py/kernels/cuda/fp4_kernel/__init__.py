import logging

from rtp_llm.models_py.utils.arch import is_cuda

from .fp4_kernel import (
    cutlass_scaled_fp4_mm_wrapper,
    is_legacy_cutlass_fp4_available,
    scaled_fp4_quant_wrapper,
)

logger = logging.getLogger(__name__)

scaled_fp4_grouped_quant = None
silu_and_mul_scaled_fp4_grouped_quant = None
flashinfer_cutedsl_moe_masked = None

if is_cuda():
    # FlashInfer/CuTeDSL is an independent optional backend and remains usable
    # on SM120 even though the legacy bindings above are not compiled there.
    try:
        from .flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
            scaled_fp4_grouped_quant,
            silu_and_mul_scaled_fp4_grouped_quant,
        )
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        logger.info("FlashInfer CuTeDSL FP4 backend unavailable: %s", error)


def is_flashinfer_cutedsl_fp4_available() -> bool:
    """Whether every wrapper required by the CuTeDSL FP4 backend loaded."""
    return all(
        wrapper is not None
        for wrapper in (
            flashinfer_cutedsl_moe_masked,
            scaled_fp4_grouped_quant,
            silu_and_mul_scaled_fp4_grouped_quant,
        )
    )


__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "flashinfer_cutedsl_moe_masked",
    "is_flashinfer_cutedsl_fp4_available",
    "is_legacy_cutlass_fp4_available",
]
