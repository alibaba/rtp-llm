"""Prepare both GDN FP8 input projections from one absmax pass."""

import os

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    FP8_GROUP_QUANT_V2_MIN_ELEMENTS,
    create_per_token_group_quant_fp8_output_scale,
    scaled_fp8_per_tensor_quant,
)
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@gluon.jit(do_not_specialize=["M", "SS"], do_not_specialize_on_alignment=["M", "SS"])
def _group_and_partial(
    X,
    Q,
    S,
    P,
    M,
    SS,
    GROUPS: gl.constexpr,
    WARPS: gl.constexpr,
    SMALL: gl.constexpr,
    V2: gl.constexpr,
):
    # Use a warp per group for short inputs and pack four groups per warp
    # for longer inputs. The BA reduction uses the maximum before clamping.
    if SMALL:
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [WARPS, 1], [1, 0])
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [WARPS, 1], [1, 0])
    tile = gl.program_id(0)
    group = tile * GROUPS + gl.arange(0, GROUPS, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    valid = group < M * 20
    offset = group[:, None] * 128 + col[None, :]
    val = gl.load(X + offset, valid[:, None], other=0).to(gl.float32)
    absolute = gl.abs(val)
    raw_max = gl.max(gl.where(absolute == absolute, absolute, 0.0), 1)
    amax = gl.maximum(raw_max, 1.0e-4)
    if V2:
        scale = amax * (1.0 / 448.0)
        code = val * gl.div_rn(448.0, amax)[:, None]
    else:
        scale = gl.div_rn(amax, 448.0)
        code = gl.div_rn(val, scale[:, None])
    code = gl.where(code == code, code, -448.0)
    code = gl.minimum(448.0, gl.maximum(-448.0, code))
    gl.store(Q + offset, code, valid[:, None])
    gl.store(S + group // 20 + (group % 20) * SS, scale, valid)
    gl.store(P + tile, gl.max(raw_max, 0))


@triton.jit
def _quant_from_partial(
    x,
    output,
    scale,
    partial,
    N,
    OUT,
    PARTS,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    tile = tl.program_id(0)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    raw = tl.load(x + index, index < N, other=0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    blocks = tl.arange(0, REDUCE)
    maximum = tl.max(tl.load(partial + blocks, blocks < PARTS, other=0), 0)
    s = tl.div_rn(maximum, 448.0)
    # Match the native CUDA per-tensor scale's flush to zero.
    s = tl.where(s < 1.1754943508222875e-38, 0, s)
    if tile == 0:
        tl.store(scale, s)
    inverse = tl.div_rn(1.0, tl.maximum(s, 1.0e-9))
    value = raw * inverse
    value = tl.where(value == value, value, 448.0)
    value = tl.maximum(-448.0, tl.minimum(448.0, value))
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        output + index, tl.where(index < N, value, 0).to(tl.float8e4nv), index < OUT
    )


@triton.jit(do_not_specialize=["PARTS"])
def _ba_scale(partial, scale, PARTS, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(partial + offsets, offsets < PARTS, other=0), 0)
    result = tl.div_rn(maximum, 448.0)
    tl.store(scale, tl.where(result < 1.1754943508222875e-38, 0, result))


def _quantize_qkvz(x, *, v2):
    m = x.shape[0]
    small = m <= 64
    if small:
        groups = warps = 2 if m <= 24 else 4
    else:
        groups, warps = (32, 4) if m <= 128 else (16, 2)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = create_per_token_group_quant_fp8_output_scale(
        x.shape, x.device, 128, True, True, False
    )
    tiles = triton.cdiv(m * 20, groups)
    partial = torch.empty(tiles, device=x.device, dtype=torch.float32)
    _group_and_partial[(tiles,)](
        x, q, s, partial, m, s.stride(1), groups, warps, small, v2, num_warps=warps
    )
    return q, s, partial


def _quantize_ba(x, partial):
    pad = triton.cdiv(x.shape[0], 16) * 16
    bq = torch.empty((pad, x.shape[1]), dtype=torch.float8_e4m3fn, device=x.device)
    tiles = partial.numel()
    bs = torch.empty(1, device=x.device, dtype=torch.float32)
    single_pass_cap = 512 if torch.cuda.is_current_stream_capturing() else 128
    if x.shape[0] > single_pass_cap:
        # Reduce once for long inputs and reuse the vectorized BA quantizer.
        if tiles > 65536:
            partial = partial.amax().reshape(1)
            tiles = 1
        _ba_scale[(1,)](partial, bs, tiles, triton.next_power_of_2(tiles), num_warps=8)
        return scaled_fp8_per_tensor_quant(x, scale=bs, output=bq)

    pdl = sm90_pdl_enabled(x.device)
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
        PDL=pdl,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return bq, bs


def _unmodified(module, methods):
    return (
        not module._forward_hooks
        and not module._forward_pre_hooks
        and all(method not in module.__dict__ for method in methods)
    )


def try_project_pair(x, qkvz, ba, *, ba_stream=None, defer_ba_wait=False):
    """Run the existing FP8 GEMMs using each projection's original scale."""
    if not (
        x.ndim == 2
        and x.shape[0] > 0
        and x.numel() <= 2**31 - 1  # Native BA quantization indexes elements with int32.
        and x.shape[1] == 2560
        and x.is_cuda
        and cached_device_capability(x.device) == (9, 0)
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.storage_offset() % 8 == 0
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
    ):
        return None
    quant_kernel = os.environ.get("DSV4_FP8_QUANT_KERNEL", "auto").strip().lower()
    if quant_kernel not in ("auto", "legacy", "v2"):
        return None
    v2 = quant_kernel == "v2" or (
        quant_kernel == "auto" and x.numel() >= FP8_GROUP_QUANT_V2_MIN_ELEMENTS
    )
    backend = (
        qkvz._flashinfer_linear
        if qkvz._should_use_flashinfer(x)
        else qkvz._deepgemm_linear
    )
    if getattr(backend, "scale_ue8m0", False) or not _unmodified(
        backend, ("forward", "quantize_input", "forward_quantized")
    ):
        return None
    q, scales, partial = _quantize_qkvz(x, v2=v2)
    if ba_stream is not None and torch.cuda.is_current_stream_capturing():
        # The BA reduction can run while the QKVZ GEMM consumes its own scales.
        main = torch.cuda.current_stream(x.device)
        ba_stream.wait_stream(main)
        projected_qkvz = backend.forward_quantized(q, scales)
        with torch.cuda.stream(ba_stream):
            ba_q, ba_scale = _quantize_ba(x, partial)
            projected_ba = ba._run_quantized(
                ba_q, ba_scale, x.shape[:-1], x.shape[0]
            ).to(x.dtype)
        if not defer_ba_wait:
            main.wait_stream(ba_stream)
        return projected_qkvz, projected_ba

    if x.shape[0] > 64:
        # Preserve the native submission order on long inputs.
        projected_qkvz = backend.forward_quantized(q, scales)
        ba_q, ba_scale = _quantize_ba(x, partial)
    else:
        ba_q, ba_scale = _quantize_ba(x, partial)
        projected_qkvz = backend.forward_quantized(q, scales)
    projected_ba = ba._run_quantized(
        ba_q, ba_scale, x.shape[:-1], x.shape[0]
    ).to(x.dtype)
    return projected_qkvz, projected_ba
