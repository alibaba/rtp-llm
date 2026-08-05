from rtp_llm.models_py.utils.arch import is_cuda

if is_cuda():
    from .flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
        scaled_fp4_grouped_quant,
        silu_and_mul_scaled_fp4_grouped_quant,
    )
    from .fp4_kernel import (
        create_per_token_group_quant_fp4_output_scale,
        cutlass_scaled_fp4_mm_wrapper,
        per_token_group_quant_fp4,
        scaled_fp4_quant_wrapper,
    )
else:
    cutlass_scaled_fp4_mm_wrapper = None
    create_per_token_group_quant_fp4_output_scale = None
    per_token_group_quant_fp4 = None
    scaled_fp4_quant_wrapper = None
    scaled_fp4_grouped_quant = None
    silu_and_mul_scaled_fp4_grouped_quant = None
    flashinfer_cutedsl_moe_masked = None
__all__ = [
    "cutlass_scaled_fp4_mm_wrapper",
    "create_per_token_group_quant_fp4_output_scale",
    "per_token_group_quant_fp4",
    "scaled_fp4_quant_wrapper",
    "scaled_fp4_grouped_quant",
    "silu_and_mul_scaled_fp4_grouped_quant",
    "flashinfer_cutedsl_moe_masked",
]
