try:
    from .get_best_config import load_all_configs
    load_all_configs()
    from .fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
        cutlass_moe_mm_fp8_scaled,
        get_best_config_swap_ab,
        per_block_cast_to_fp8,
        per_token_cast_to_fp8,
        requant_weight_ue8m0,
        scaled_fp8_per_tensor_quant,
        scaled_fp8_per_token_quant,
        sgl_per_token_group_quant_fp8,
    )
except (ImportError, AttributeError, OSError):
    # CUDA kernels not available on ROCm — provide None stubs
    sgl_per_token_group_quant_fp8 = None
    scaled_fp8_per_tensor_quant = None
    scaled_fp8_per_token_quant = None
    cutlass_moe_mm_fp8_scaled = None
    get_best_config_swap_ab = None
    per_token_cast_to_fp8 = None
    per_block_cast_to_fp8 = None
    requant_weight_ue8m0 = None
    create_per_token_group_quant_fp8_output_scale = None

__all__ = [
    "sgl_per_token_group_quant_fp8",
    "scaled_fp8_per_tensor_quant",
    "scaled_fp8_per_token_quant",
    "cutlass_moe_mm_fp8_scaled",
    "get_best_config_swap_ab",
    "per_token_cast_to_fp8",
    "per_block_cast_to_fp8",
    "requant_weight_ue8m0",
    "create_per_token_group_quant_fp8_output_scale",
]
