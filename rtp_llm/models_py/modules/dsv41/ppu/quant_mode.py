"""Import-light, opt-in Flash1017 mixed quantization selection."""

import os

QUANT_MODE_ENV = "RTP_V41_PPU_QUANT_MODE"
LEGACY = "legacy"
MXFP4_FP8_PTPC = "mxfp4_fp8_ptpc"
PTPC_CONFIG_SHA256 = "102d5e5862e77618b88f589febae523f3d3ac743a3dc5c72c1756aedc760b58d"
PTPC_INDEX_SHA256 = "a94e452343136142985952b107ed1e9751671cc81a1054f4aad2e4b580869b2f"


def quant_mode(env=None):
    env = os.environ if env is None else env
    mode = env.get(QUANT_MODE_ENV, LEGACY).strip().lower()
    if mode not in (LEGACY, MXFP4_FP8_PTPC):
        raise ValueError(f"{QUANT_MODE_ENV} must be legacy or mxfp4_fp8_ptpc")
    if mode == MXFP4_FP8_PTPC:
        if env.get("RTP_V41_BACKEND", "ppu") != "ppu":
            raise ValueError(f"{QUANT_MODE_ENV} requires the PPU backend")
        precision = env.get("RTP_V41_PPU_MOE_PRECISION", "w4a4").strip().lower()
        if precision != "w4a4":
            raise ValueError(f"{QUANT_MODE_ENV}={mode} requires MoE w4a4")
    return mode
