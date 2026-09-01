import logging

import torch

from rtp_llm.models_py.utils.arch import is_cuda, is_sm10x

logger = logging.getLogger(__name__)

cutlass_scaled_fp4_mm_wrapper = None
scaled_fp4_quant_wrapper = None
scaled_fp4_grouped_quant = None
silu_and_mul_scaled_fp4_grouped_quant = None
flashinfer_cutedsl_moe_masked = None

if is_cuda():
    # The in-tree legacy bindings are deliberately absent from the SM120
    # extension.  Import them only on the exact SM10x family that builds the
    # symbols; importing the package on consumer Blackwell must remain valid.
    if torch.cuda.is_available() and is_sm10x():
        from .fp4_kernel import cutlass_scaled_fp4_mm_wrapper, scaled_fp4_quant_wrapper

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
__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "flashinfer_cutedsl_moe_masked",
]
