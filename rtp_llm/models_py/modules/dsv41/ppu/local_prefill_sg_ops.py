"""Explicit SG-aligned local Prefill primitives; business routing is caller-owned.

The SwiGLU entry uses the frozen SG BF16-SiLU/BF16-multiply contract, not
the legacy FP32 activation contract. Neither entry reads environment flags.
"""

from functools import lru_cache

import torch

from rtp_llm.models_py.kernels.ppu.jit.sglang_jit import load_sglang_kernel


def _normal_bf16_rows(value, width):
    return (
        isinstance(value, torch.Tensor)
        and value.layout == torch.strided
        and not value.is_neg() and not value.is_conj()
        and value.dtype == torch.bfloat16 and value.ndim == 2
        and 0 <= value.shape[0] <= 16384 and value.shape[1] == width
        and value.is_contiguous() and not value.requires_grad
        and not torch.is_grad_enabled()
    )


def _supported_device(value):
    if not value.is_cuda:
        return False
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    if not is_ppu() or torch.version.hip is not None:
        return False
    with torch.cuda.device(value.device):
        return (torch.cuda.get_device_capability(value.device) == (8, 9)
                and not torch.cuda.is_current_stream_capturing())


def _device_context(value):
    return torch.cuda.device(value.device)


def supports_swiglu(gateup, limit):
    return (
        _normal_bf16_rows(gateup, 4608)
        and type(limit) in (int, float) and limit == 10.0
        and (gateup.shape[0] == 0 or gateup.data_ptr() % 16 == 0)
        and _supported_device(gateup)
    )


def supports_hc(hidden):
    return _normal_bf16_rows(hidden, 5120) and _supported_device(hidden)


@lru_cache(maxsize=None)
def _swiglu_module(arch):
    return load_sglang_kernel(
        "prefill_silu_mul_fp8",
        "compat/elementwise/silu_and_mul_post_per_token_quant_fp8.cuh",
        "sglang::SiluMulFp8TP<256,true,false>::run",
        arch,
        ("-use_fast_math",),
        bundle="prefill_swiglu",
    )


def swiglu_quant(gateup, limit):
    """Read BF16 [M,4608], return owned E4M3 [M,2304]/FP32 [M,1].

    SG rounds SiLU to BF16, then multiplies in BF16 before row FP8 quant.
    Unsupported metadata raises before allocation/launch; native errors
    propagate. The input and all caller-owned weights remain read-only.
    """
    if not supports_swiglu(gateup, limit):
        raise ValueError("SG Prefill SwiGLU requires normal PPU89 BF16 [M,4608] and limit10")
    rows = gateup.shape[0]
    output = torch.empty((rows, 2304), dtype=torch.float8_e4m3fn, device=gateup.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=gateup.device)
    if rows:
        with _device_context(gateup):
            module = _swiglu_module(torch.cuda.get_device_capability(gateup.device))
            module.forward(gateup, output, scale, float(limit), 0.0, 0.0, 1e-10)
    return output, scale


def _expand_hc(hidden):
    from rtp_llm.models_py.modules.dsv41.prefill._hc_expand_triton import expand_hc

    return expand_hc(hidden)


def hc_broadcast(hidden):
    """Materialize four independent lanes, preserving every BF16 input bit."""
    if not supports_hc(hidden):
        raise ValueError("SG Prefill HC broadcast requires normal PPU89 BF16 [M,5120]")
    if hidden.shape[0] == 0:
        return hidden.new_empty((0, 4, 5120))
    with _device_context(hidden):
        return _expand_hc(hidden)
