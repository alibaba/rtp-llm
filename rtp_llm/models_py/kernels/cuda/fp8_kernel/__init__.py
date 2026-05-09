from rtp_llm.models_py.utils.arch import is_cuda

from .get_best_config import load_all_configs

# load all configs once at import time
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

if is_cuda():
    try:
        from .flashinfer_cutedsl_moe_masked_fp8 import (
            flashinfer_cutedsl_moe_masked_fp8,
            quant_mxfp8_grouped_activation,
            quant_mxfp8_per_expert,
            reshape_mxfp8_weight,
        )
    except ImportError:
        flashinfer_cutedsl_moe_masked_fp8 = None
        quant_mxfp8_grouped_activation = None
        quant_mxfp8_per_expert = None
        reshape_mxfp8_weight = None
else:
    flashinfer_cutedsl_moe_masked_fp8 = None
    quant_mxfp8_grouped_activation = None
    quant_mxfp8_per_expert = None
    reshape_mxfp8_weight = None

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
    "flashinfer_cutedsl_moe_masked_fp8",
    "quant_mxfp8_grouped_activation",
    "quant_mxfp8_per_expert",
    "reshape_mxfp8_weight",
]
