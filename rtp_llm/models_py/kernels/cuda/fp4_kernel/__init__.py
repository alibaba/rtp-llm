from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from .fp4_kernel import cutlass_scaled_fp4_mm_wrapper
    from .fp4_kernel import scaled_fp4_quant_wrapper
else:
    cutlass_scaled_fp4_mm_wrapper = None
    scaled_fp4_quant_wrapper = None

__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "scaled_fp4_quant_wrapper",
]

