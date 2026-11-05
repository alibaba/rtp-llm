"""Import-light selection of the PPU routed-expert GEMM precision."""

from __future__ import annotations

import os
from typing import Mapping, Optional

MOE_PRECISION_ENV = "RTP_V41_PPU_MOE_PRECISION"


def moe_precision(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    value = env.get(MOE_PRECISION_ENV, "w4a8").strip().lower()
    if value not in ("w4a8", "w4a4"):
        raise ValueError(f"{MOE_PRECISION_ENV} must be w4a8 or w4a4, got {value!r}")
    return value


def moe_kernel_name(precision: str) -> str:
    activation = "fp8" if precision == "w4a8" else "fp4"
    return f"m_grouped_gemm_{activation}_fp4_bf16_nt_nopad"
