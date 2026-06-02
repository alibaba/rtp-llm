from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from .fp4_kernel import cutlass_scaled_fp4_mm_wrapper
    from .fp4_kernel import scaled_fp4_quant_wrapper
    from .flashinfer_cutedsl_moe import scaled_fp4_grouped_quant
    from .flashinfer_cutedsl_moe import silu_and_mul_scaled_fp4_grouped_quant
    from .flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
else:
    cutlass_scaled_fp4_mm_wrapper = None
    scaled_fp4_quant_wrapper = None
    scaled_fp4_grouped_quant = None
    silu_and_mul_scaled_fp4_grouped_quant = None
    flashinfer_cutedsl_moe_masked = None
__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "flashinfer_cutedsl_moe_masked",
]

