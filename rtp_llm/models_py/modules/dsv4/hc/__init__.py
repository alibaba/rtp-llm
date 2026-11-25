"""DeepSeek-V4 Hyper-Connection implementations."""

from rtp_llm.models_py.modules.dsv4.hc._fused_mhc_pre_norm import (
    can_use_fused_mhc_pre_norm,
    copy_2d_buffer,
    fused_mhc_pre_norm,
    fused_mhc_pre_norm_quant,
    mhc_mean_copy,
)
from rtp_llm.models_py.modules.dsv4.hc.factory import build_hc_head, build_hc_unit

__all__ = [
    "build_hc_head",
    "build_hc_unit",
    "can_use_fused_mhc_pre_norm",
    "copy_2d_buffer",
    "fused_mhc_pre_norm",
    "fused_mhc_pre_norm_quant",
    "mhc_mean_copy",
]
