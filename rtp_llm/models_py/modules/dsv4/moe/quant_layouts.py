"""Compatibility re-export for DSV4 quant layout helpers."""

from rtp_llm.models_py.modules.dsv4.quant_layouts import (  # noqa: F401
    FP4_BLOCK,
    FP8_BLOCK,
    _per_token_cast_to_fp8_packed_ue8m0,
    prepare_fp4_weight_scale_for_deepgemm,
)
