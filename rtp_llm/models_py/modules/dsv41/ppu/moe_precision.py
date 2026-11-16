"""Import-light selection of the PPU routed-expert GEMM precision."""

from __future__ import annotations

import os
from typing import Mapping, Optional

from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode

MOE_PRECISION_ENV = "RTP_V41_PPU_MOE_PRECISION"
MOE_BACKEND_ENV = "RTP_V41_PPU_MOE_BACKEND"
GC_MOE_MODE_ENV = "RTP_V41_PPU_GC_MOE_MODE"
GC_MOE_TOKEN_BOUND_ENV = "RTP_V41_PPU_GC_MOE_DECODE_TOKEN_BOUND"


def gc_moe_mode(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    mode = env.get(GC_MOE_MODE_ENV, "off").strip().lower()
    if mode not in ("off", "routed", "w4a4", "w8a8"):
        raise ValueError(f"{GC_MOE_MODE_ENV} must be off, routed, w4a4, or w8a8, got {mode!r}")
    return mode


def megamoe_module_name(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    value = env.get("RTP_V41_PPU_MEGAMOE_MODULE", "deep_gemm")
    if value not in ("deep_gemm", "deep_gemm_mega"):
        raise ValueError("RTP_V41_PPU_MEGAMOE_MODULE must be deep_gemm or deep_gemm_mega")
    return value


def moe_backend(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    value = env.get(MOE_BACKEND_ENV, "grouped").strip().lower()
    if value not in ("grouped", "megamoe"):
        raise ValueError(f"{MOE_BACKEND_ENV} must be grouped or megamoe, got {value!r}")
    return value


def moe_precision(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    backend = moe_backend(env)
    default = "w4a4" if backend == "megamoe" or quant_mode(env) == MXFP4_FP8_PTPC else "w4a8"
    value = env.get(MOE_PRECISION_ENV, default).strip().lower()
    if value not in ("w4a8", "w4a4"):
        raise ValueError(f"{MOE_PRECISION_ENV} must be w4a8 or w4a4, got {value!r}")
    if backend == "megamoe" and value != "w4a4":
        raise ValueError(
            "PPU890 MegaMoE requires W4A4; unset "
            + MOE_PRECISION_ENV
            + " or set it to w4a4"
        )
    return value


def moe_kernel_name(precision: str, backend: str = "grouped") -> str:
    if backend == "megamoe":
        if precision != "w4a4":
            raise ValueError("PPU890 MegaMoE requires W4A4")
        return "fp4_fp4_mega_moe"
    activation = "fp8" if precision == "w4a8" else "fp4"
    return f"m_grouped_gemm_{activation}_fp4_bf16_nt_nopad"
