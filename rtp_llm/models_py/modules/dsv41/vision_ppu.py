"""PPU BF16 vision SwiGLU using the frozen, rounded SGLang activation.

The input contains gate then up. SiLU is rounded to BF16 before the multiply,
matching ``F.silu(gate) * up``; there is no clipping or activation quantization.
Warm the entry with an eager call before Graph capture. Unsupported metadata
returns None before allocation or JIT; selected provider errors propagate.
"""

from functools import lru_cache

import torch

_INTERMEDIATE = 2816
# The upstream activation indexes vector offsets with uint32. This conservative
# element-count bound also protects the wider input offsets before conversion.
_MAX_ROWS = ((1 << 32) - 1) // (2 * _INTERMEDIATE)


def _supported(gateup):
    if not (
        isinstance(gateup, torch.Tensor)
        and gateup.layout == torch.strided
        and gateup.is_cuda
        and not torch.is_grad_enabled()
        and not gateup.requires_grad
        and not gateup.is_neg()
        and not gateup.is_conj()
        and gateup.dtype == torch.bfloat16
        and gateup.ndim == 2
        and 0 <= gateup.shape[0] <= _MAX_ROWS
        and gateup.shape[1] == 2 * _INTERMEDIATE
        and gateup.is_contiguous()
        and (gateup.shape[0] == 0 or gateup.data_ptr() % 16 == 0)
        and torch.version.hip is None
    ):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    return is_ppu()


@lru_cache(maxsize=None)
def _activation_module(arch):
    # A warm cache lookup does not enter this function during capture.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm PPU vision SwiGLU eagerly before Graph capture")
    from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel

    return load_sglang_kernel(
        "vision_silu_mul_rounded",
        "csrc/elementwise/activation.cuh",
        "sglang::ActivationKernel<bf16_t,false>::run_activation_with_rounding",
        arch,
        ("--fmad=false",),
        bundle="vision_activation",
    )


def apply_vision_silu_mul(gateup):
    """Read PPU BF16 ``[M,5632]``, return fresh BF16 ``[M,2816]`` or None.

    The input is read-only, including when it is an aligned view of a larger
    allocation. Empty supported inputs return an empty output without JIT.
    """
    if not _supported(gateup):
        return None
    with torch.cuda.device(gateup.device):
        module = (
            _activation_module(torch.cuda.get_device_capability(gateup.device))
            if gateup.shape[0]
            else None
        )
        output = torch.empty(
            (gateup.shape[0], _INTERMEDIATE),
            dtype=torch.bfloat16,
            device=gateup.device,
        )
        if module is not None:
            module.forward(gateup, output, "silu", 0.0)
    return output
