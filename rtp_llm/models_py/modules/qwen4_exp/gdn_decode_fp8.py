"""Share the legacy FP8 group quantizer read with Qwen4 GDN BA absmax."""

import os

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.utils.device_capability import cached_device_capability


@gluon.jit
def _group_and_partial(X, Q, S, P, M, SS, GROUPS: gl.constexpr, WARPS: gl.constexpr):
    # Keep per-group scales and emit one unclamped absmax per CTA for BA.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [WARPS, 1], [1, 0])
    tile = gl.program_id(0)
    group = tile * GROUPS + gl.arange(0, GROUPS, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    valid = group < M * 20
    offset = group[:, None] * 128 + col[None, :]
    val = gl.load(X + offset, valid[:, None], other=0).to(gl.float32)
    absolute = gl.abs(val)
    raw_max = gl.max(gl.where(absolute == absolute, absolute, 0.0), 1)
    scale = gl.div_rn(gl.maximum(raw_max, 1.0e-4), 448.0)
    code = gl.div_rn(val, scale[:, None])
    code = gl.where(code == code, code, -448.0)
    code = gl.minimum(448.0, gl.maximum(-448.0, code))
    gl.store(Q + offset, code, valid[:, None])
    gl.store(S + group // 20 + (group % 20) * SS, scale, valid)
    gl.store(P + tile, gl.max(raw_max, 0))


@triton.jit
def _quant_from_partial(
    x, output, scale, partial, N, OUT, PARTS, REDUCE: tl.constexpr, BLOCK: tl.constexpr
):
    tile = tl.program_id(0)
    blocks = tl.arange(0, REDUCE)
    maximum = tl.max(tl.load(partial + blocks, blocks < PARTS, other=0), 0)
    s = tl.div_rn(maximum, 448.0)
    # Match the native CUDA per-tensor scale's flush to zero.
    s = tl.where(s < 1.1754943508222875e-38, 0, s)
    if tile == 0:
        tl.store(scale, s)
    inverse = tl.div_rn(1.0, tl.maximum(s, 1.0e-9))
    index = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(x + index, index < N, other=0).to(tl.float32) * inverse
    value = tl.where(value == value, value, 448.0)
    value = tl.maximum(-448.0, tl.minimum(448.0, value))
    tl.store(
        output + index, tl.where(index < N, value, 0).to(tl.float8e4nv), index < OUT
    )


def _quantized_pair(x):
    m = x.shape[0]
    groups = warps = 2 if m <= 24 else 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = create_per_token_group_quant_fp8_output_scale(
        x.shape, x.device, 128, True, True, False
    )
    tiles = triton.cdiv(m * 20, groups)
    partial = torch.empty(tiles, device=x.device, dtype=torch.float32)
    _group_and_partial[(tiles,)](
        x, q, s, partial, m, s.stride(1), groups, warps, num_warps=warps
    )
    pad = triton.cdiv(m, 16) * 16
    bq = torch.empty((pad, x.shape[1]), dtype=q.dtype, device=x.device)
    bs = torch.empty(1, device=x.device, dtype=torch.float32)
    _quant_from_partial[(triton.cdiv(bq.numel(), 1024),)](
        x,
        bq,
        bs,
        partial,
        x.numel(),
        bq.numel(),
        tiles,
        triton.next_power_of_2(tiles),
        1024,
        enable_fp_fusion=False,
    )
    return q, s, bq, bs


def _unmodified(module, methods):
    return (
        not module._forward_hooks
        and not module._forward_pre_hooks
        and all(method not in module.__dict__ for method in methods)
    )


def try_project_pair(x, qkvz, ba):
    """Run the existing FP8 GEMMs using each projection's original scale."""
    if not (
        x.ndim == 2
        and 16 <= x.shape[0] <= 64
        and x.shape[1] == 2560
        and x.is_cuda
        and cached_device_capability(x.device) == (9, 0)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and not x.requires_grad
        and type(qkvz) is CudaFp8GEMMLinear
        and (qkvz.N, qkvz.K) == (8192, 2560)
        and type(ba) is Fp8Linear
        and ba.weight.shape == (48, 2560)
        and ba.weight.dtype == torch.float8_e4m3fn
        and ba.use_scaled_mm
        and ba.compute_weight is None
        and x.device == qkvz.weight.device == ba.weight.device
        and not torch.nn.modules.module._global_forward_hooks
        and not torch.nn.modules.module._global_forward_pre_hooks
        and _unmodified(qkvz, ("forward", "_should_use_flashinfer"))
        and _unmodified(ba, ("forward",))
        and os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
        in ("auto", "legacy")
    ):
        return None
    backend = (
        qkvz._flashinfer_linear
        if qkvz._should_use_flashinfer(x)
        else qkvz._deepgemm_linear
    )
    if getattr(backend, "scale_ue8m0", False) or not _unmodified(
        backend, ("forward", "quantize_input", "forward_quantized")
    ):
        return None
    q, scales, ba_q, ba_scale = _quantized_pair(x)
    return (
        backend.forward_quantized(q, scales),
        ba._run_quantized(ba_q, ba_scale, x.shape[:-1], x.shape[0]).to(x.dtype),
    )
