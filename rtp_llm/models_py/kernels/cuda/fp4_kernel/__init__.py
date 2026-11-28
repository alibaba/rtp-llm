import logging

from rtp_llm.models_py.utils.arch import is_cuda, is_sm10x, is_sm12x

logger = logging.getLogger(__name__)

scaled_fp4_grouped_quant = None
silu_and_mul_scaled_fp4_grouped_quant = None
flashinfer_cutedsl_moe_masked = None
_cutedsl_quant_available = False


if is_cuda():
    from .fp4_kernel import (
        cutlass_scaled_fp4_mm_wrapper,
        is_legacy_cutlass_fp4_available,
        scaled_fp4_quant_wrapper,
    )

    try:
        from .flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
            scaled_fp4_grouped_quant,
            silu_and_mul_scaled_fp4_experts_quant,
            silu_and_mul_scaled_fp4_grouped_quant,
        )

        _cutedsl_quant_available = callable(silu_and_mul_scaled_fp4_experts_quant)
    except (ImportError, AttributeError, OSError, RuntimeError) as error:
        if not is_sm12x():
            raise
        logger.info("FlashInfer CuTeDSL FP4 backend unavailable: %s", error)
else:
    cutlass_scaled_fp4_mm_wrapper = None
    scaled_fp4_quant_wrapper = None

    def is_legacy_cutlass_fp4_available() -> bool:
        return False


def is_flashinfer_cutedsl_fp4_available() -> bool:
    """The grouped CuTeDSL kernel requires SM10x and native quantization."""
    return (
        is_cuda()
        and is_sm10x()
        and _cutedsl_quant_available
        and all(
            wrapper is not None
            for wrapper in (
                flashinfer_cutedsl_moe_masked,
                scaled_fp4_grouped_quant,
                silu_and_mul_scaled_fp4_grouped_quant,
            )
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

