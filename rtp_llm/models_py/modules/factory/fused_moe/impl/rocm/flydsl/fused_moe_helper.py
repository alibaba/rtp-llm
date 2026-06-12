"""
aiter_compat.py
===============
All aiter dependencies extracted into one file, eliminating direct aiter imports.
All functions/classes/constants are copied from their original definitions without refactoring.
"""

import functools
import inspect
import logging
import argparse
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def get_m_per_expert_avg(B, TOPK, E):
    return math.ceil(B * TOPK / E)


def get_m_align(B, TOPK, E):
    # NOTE: this is an experimental value and may be adjusted later
    m_per_expert_avg = get_m_per_expert_avg(B, TOPK, E)
    align_m_per_expert_avg = None
    if m_per_expert_avg <= 32:
        align_m_per_expert_avg = ((m_per_expert_avg + 31) // 32) * 32
    else:
        align_m_per_expert_avg = ((m_per_expert_avg + 63) // 64) * 64

    # NOTE: limit bucket size to avoid long autotune time
    MAX_BUCKET_M = 1024
    align_m_per_expert_avg = min(align_m_per_expert_avg, MAX_BUCKET_M)

    return math.ceil(align_m_per_expert_avg * E / TOPK)


def get_block_size_m(B, TOPK, E):
    # NOTE: this is an experimental value and may be adjusted later
    m_per_expert_avg = get_m_per_expert_avg(B, TOPK, E)
    return 32 if m_per_expert_avg <= 32 else 64


# ===========================================================================
# flush_cache (from: tensor_tools.flush_cache)
# Falls back to no-op when tensor_tools is not available
# ===========================================================================
try:
    from tensor_tools import flush_cache
except ImportError:

    def flush_cache(size_mb=512):
        pass


# ===========================================================================
# profile_cuda_kernels (from: tensor_tools.profile_cuda_kernels)
# Falls back to direct function call when tensor_tools is not available
# ===========================================================================
try:
    from tensor_tools import profile_cuda_kernels
except ImportError:

    def profile_cuda_kernels(fn, enable_print=False, warmup=5, iters=30, topk=80):
        for _ in range(warmup):
            fn()
        result = None
        for _ in range(iters):
            result = fn()
        torch.cuda.synchronize()
        return result, {}


# ===========================================================================
# logger (from: aiter.logger)
# ===========================================================================
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ===========================================================================
# ActivationType / QuantType (from: aiter.jit.module_aiter_enum, C++ pybind11)
# Re-defined as Python IntEnum, keeping .value and .name consistent
# ===========================================================================
from enum import IntEnum


class ActivationType(IntEnum):
    No = -1
    Silu = 0
    Gelu = 1
    Swiglu = 2


class QuantType(IntEnum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x32 = 3
    per_1x128 = 4
    per_128x128 = 5


# ===========================================================================
# dtypes (from: aiter.utility.dtypes)
# ===========================================================================
_8bit_fallback = torch.uint8

i4x2 = getattr(torch, "int4", _8bit_fallback)
fp4x2 = getattr(torch, "float4_e2m1fn_x2", _8bit_fallback)
# fp8: gfx942 uses e4m3fnuz, gfx950 uses e4m3fn; default to gfx942
fp8 = torch.float8_e4m3fnuz
fp8_e8m0 = getattr(torch, "float8_e8m0fnu", _8bit_fallback)
fp16 = torch.float16
bf16 = torch.bfloat16
fp32 = torch.float32
u32 = torch.uint32
i32 = torch.int32
i16 = torch.int16
i8 = torch.int8

d_dtypes = {
    "fp8": fp8,
    "fp8_e8m0": fp8_e8m0,
    "fp16": fp16,
    "bf16": bf16,
    "fp32": fp32,
    "i4x2": i4x2,
    "fp4x2": fp4x2,
    "u32": u32,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}
dtypes = SimpleNamespace(**d_dtypes)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2tuple(v):
    try:
        parts = v.strip("()").split(",")
        return tuple(int(p.strip()) for p in parts)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid format of input: {v}") from e


# ===========================================================================
# get_torch_quant / get_torch_act (from: aiter.ops.quant)
# ===========================================================================
def get_torch_quant(qType):
    def per_token_quant(x, quant_dtype=fp8):
        amax = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (amax / torch.finfo(quant_dtype).max).to(torch.float32)
        return (x.float() / scale).to(quant_dtype), scale

    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Token: per_token_quant,
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


def get_torch_act(aType):
    tmp = {
        ActivationType.No: lambda *a, **k: a[0],
        ActivationType.Silu: F.silu,
        ActivationType.Gelu: F.gelu,
    }
    return tmp.get(aType, NotImplementedError)


# ===========================================================================
# get_inter_dim / quant_remap (from: aiter.fused_moe)
# ===========================================================================
quant_remap = {QuantType.per_128x128: QuantType.per_1x128}


@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape
    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


# ===========================================================================
# fused_topk (from: aiter.fused_moe.fused_topk)
# Uses aiter C++ topk_softmax when available; falls back to pure torch
# ===========================================================================
_aiter = None
_has_aiter = None


def _get_aiter():
    global _aiter, _has_aiter
    if _has_aiter is None:
        try:
            import aiter as _aiter_mod
        except ImportError:
            _aiter = None
            _has_aiter = False
        else:
            _aiter = _aiter_mod
            _has_aiter = True
    return _aiter if _has_aiter else None


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    topk_ids: torch.Tensor = None,
    topk_weights: torch.Tensor = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    M, _ = hidden_states.shape

    aiter_mod = _get_aiter()
    if aiter_mod is not None:
        expert = gating_output.shape[1]
        token_expert_indicies = torch.empty(
            M, topk, dtype=i32, device=hidden_states.device
        )
        if topk_weights is None:
            topk_weights = torch.empty(M, topk, dtype=fp32, device=hidden_states.device)
        if topk_ids is None:
            topk_ids = torch.empty(M, topk, dtype=i32, device=hidden_states.device)
        aiter_mod.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            gating_output,
            renormalize,
        )
        return topk_weights, topk_ids

    # Fallback: pure torch implementation
    # NOTE: this will increase the final result error for torch.bfloat16
    scores = torch.softmax(gating_output.float(), dim=-1)
    topk_weights_out, topk_ids_out = torch.topk(scores, topk, dim=-1)

    if renormalize:
        topk_weights_out = topk_weights_out / topk_weights_out.sum(dim=-1, keepdim=True)

    topk_weights_out = topk_weights_out.to(torch.float32)
    topk_ids_out = topk_ids_out.to(torch.int32)

    return topk_weights_out, topk_ids_out


# ===========================================================================
# torch_moe_stage1 (from: aiter.fused_moe.torch_moe_stage1)
# ===========================================================================
def torch_moe_stage1(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weight,
    topk_ids,
    dtype=fp16,
    activation=ActivationType.Silu,
    quant_type=QuantType.No,
    # following for quant
    a1_scale=None,  # [token, 1]
    w1_scale=None,  # [expert, inter_dim, 1]
    w1_bias=None,  # [expert, inter_dim, 1]
    doweight=False,
):
    assert doweight is False
    quant_type = quant_remap.get(quant_type, quant_type)
    ctype = fp32  # compute type
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    N = w1.shape[1]
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.No:
        hidden_states = hidden_states.to(ctype)
        w1 = w1.to(ctype)
    else:
        raise NotImplementedError(
            f"quant_type {quant_type} not supported in compat layer"
        )

    hidden_states = hidden_states.view(B, -1, model_dim).repeat(1, topk, 1)

    out = torch.zeros(
        (B, topk, N),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if doweight:
                act_input = act_input * topk_weight[mask].view(-1, 1)
            out[mask] = act_input
            if w1_bias is not None:
                out[mask] = out[mask] + w1_bias[E_id].view(1, -1)
    use_g1u1 = w1.shape[1] == (2 * inter_dim)

    torch_act = get_torch_act(activation)
    if use_g1u1:
        gate, up = out.split([inter_dim, inter_dim], dim=-1)
        out = torch_act(gate) * up
    else:
        out = torch_act(out)
    return out.to(dtype)


# ===========================================================================
# torch_moe_stage2 (from: aiter.fused_moe.torch_moe_stage2)
# ===========================================================================
def torch_moe_stage2(
    hidden_states,
    w1,  # E, inter_dim*2, model_dim
    w2,  # E, model_dim, inter_dim
    topk_weights,
    topk_ids,
    dtype=fp16,
    quant_type=QuantType.No,
    w2_scale=None,
    a2_scale=None,
    w2_bias=None,
    doweight=True,
):
    ctype = fp32  # compute type
    E, model_dim, inter_dim = get_inter_dim(w1.shape, w2.shape)

    if quant_type == QuantType.No:
        hidden_states = hidden_states.to(ctype)
        w2 = w2.to(ctype)
    else:
        raise NotImplementedError(
            f"quant_type {quant_type} not supported in compat layer"
        )

    token_num, topk = topk_ids.shape
    hidden_states = hidden_states.view(token_num, topk, inter_dim)

    out = torch.zeros(
        (token_num, topk, model_dim),
        dtype=ctype,
        device=hidden_states.device,
    )
    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w2[E_id].transpose(0, 1))
            out[mask] = act_input
            if w2_bias is not None:
                out[mask] = out[mask] + w2_bias[E_id].view(1, -1)

    if doweight:
        out = out * topk_weights.view(token_num, -1, 1)
    return out.sum(1).to(dtype)


# ===========================================================================
# shuffle_weight (from: aiter.ops.shuffle)
# Gluon MoE kernels expect weight in (16,16) shuffled layout
# ===========================================================================
def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    return x_


# ===========================================================================
# moe_sorting (calls atrex.api.moe_sorting.moe_sorting_fwd via CK JIT)
# ===========================================================================
def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=64,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
):
    from .moe_sorting import moe_sorting_fwd

    device = topk_ids.device
    M, topk = topk_ids.shape

    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=i32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=fp32, device=device)
    sorted_expert_ids = torch.empty((max_num_m_blocks,), dtype=i32, device=device)
    num_valid_ids = torch.empty((2), dtype=i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    flush_cache(512)
    moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
        dispatch_policy,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


# ===========================================================================
# test helpers (from: aiter.test_common)
# ===========================================================================
def log_args(func, *args, **kwargs):
    callargs = inspect.getcallargs(func, *args, **kwargs)

    prefix = f"calling {func.__name__}("
    blanks = " " * (len(prefix))

    def getTensorInfo(el):
        if isinstance(el, torch.Tensor):
            return f"{el.shape} {el.dtype} {el.device} {hex(el.data_ptr())}"
        elif isinstance(el, tuple):
            viewNum = 5
            if len(el) > viewNum:
                el = list(el[:viewNum]) + ["..."]
            return f'\n{" "*(len(prefix)+31)}'.join(
                ["("] + [f" {getTensorInfo(e)}" for e in el] + [")"]
            )
        return el

    info = [f"{el:<28} = {getTensorInfo(callargs[el])}" for el in callargs]
    info = f",\n{blanks}".join(info)
    logger.info(f"\n{prefix}{info})")
    return callargs


def benchmark():
    def decorator(func):
        def wrapper(*args, **kwargs):
            callargs = log_args(func, *args, **kwargs)
            ret = func(*args, **kwargs)
            if ret is not None:
                callargs.update(ret)
            return callargs

        return wrapper

    return decorator


def checkAllclose(
    a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg="", printNum=8, printLog=True
):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)

    if isClose.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        try:
            mask = ~isClose
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        except RuntimeError:
            mask = ~isClose.to("cpu")
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        if percent > tol_err_ratio:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        else:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[33mwarning!\033[0m] a and b results are not all close"""
            )
        logger.info(
            f"-->max abs delta:{delta.max()}, delta details: {percent:.1%} ({num} of {a.numel()}) elements"
        )
        return percent
