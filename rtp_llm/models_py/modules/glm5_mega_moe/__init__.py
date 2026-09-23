"""GLM-5 adapters for DeepGEMM MegaMoE kernels."""

from .input_packer import get_mega_moe_input_packer
from .jit_warmup import generate_mega_moe_jit_token_counts, mega_moe_jit_warmup_enabled
from .mega_buf import (
    get_or_create_mega_buf,
    get_or_create_mega_output,
    mega_moe_available,
    mega_moe_enabled,
)
from .mega_fp8_buf import (
    get_or_create_mega_buf_fp8,
    mega_moe_fp8_available,
    mega_moe_fp8_enabled,
)
from .mega_moe import GLM5MegaMoE
from .mega_moe_fp8 import GLM5MegaMoEFP8
from .mega_moe_fp8_se import GLM5MegaMoEFP8SE
from .mega_moe_fused import GLM5MegaMoEFused

__all__ = [
    "GLM5MegaMoE",
    "GLM5MegaMoEFused",
    "GLM5MegaMoEFP8",
    "GLM5MegaMoEFP8SE",
    "get_or_create_mega_buf",
    "get_or_create_mega_buf_fp8",
    "get_or_create_mega_output",
    "mega_moe_available",
    "mega_moe_enabled",
    "mega_moe_fp8_available",
    "mega_moe_fp8_enabled",
    "get_mega_moe_input_packer",
    "mega_moe_jit_warmup_enabled",
    "generate_mega_moe_jit_token_counts",
]
