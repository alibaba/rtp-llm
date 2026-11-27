"""Graph-safe gated residual branch mixing for Qwen4."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@dataclass
class MixReduceFp8Bundle:
    x: torch.Tensor
    q: torch.Tensor
    s: torch.Tensor
    partial: torch.Tensor
    mode: str


@dataclass
class MixReduceFp8Request:
    mode: str
    result: Optional[MixReduceFp8Bundle] = None


_mix_reduce_fp8_request: ContextVar[Optional[MixReduceFp8Request]] = ContextVar(
    "_mix_reduce_fp8_request", default=None
)
_prequantized_mix_reduce: ContextVar[Optional[MixReduceFp8Bundle]] = ContextVar(
    "_prequantized_mix_reduce", default=None
)

_FP8_MODE_IDS = {
    "attn_legacy": 0,
    "attn_v2": 1,
    "moe_trt128": 2,
}


def get_prequantized_mix_reduce(
    x: torch.Tensor, expected_mode: str
) -> Optional[MixReduceFp8Bundle]:
    bundle = _prequantized_mix_reduce.get()
    if (
        bundle is not None
        and bundle.mode == expected_mode
        and x.data_ptr() == bundle.x.data_ptr()
        and tuple(x.shape) == tuple(bundle.x.shape)
        and x.stride() == bundle.x.stride()
        and x.device == bundle.x.device
        and x.dtype == bundle.x.dtype
    ):
        return bundle
    return None


@triton.jit
def _mix_reduce_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < HIDDEN
    if PDL:
        base = row * (BRANCHES * HIDDEN) + columns
        n0 = tl.load(normed_ptr + base, mask=mask, other=0).to(tl.float32)
        n1 = tl.load(normed_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        n2 = tl.load(normed_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        n3 = tl.load(normed_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        tl.extra.cuda.gdc_wait()
        l0 = tl.load(logits_ptr + base, mask=mask, other=0).to(tl.float32)
        l1 = tl.load(logits_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        l2 = tl.load(logits_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        l3 = tl.load(logits_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        g0 = (1.0 / (1.0 + tl.exp(-l0))).to(tl.bfloat16).to(tl.float32)
        g1 = (1.0 / (1.0 + tl.exp(-l1))).to(tl.bfloat16).to(tl.float32)
        g2 = (1.0 / (1.0 + tl.exp(-l2))).to(tl.bfloat16).to(tl.float32)
        g3 = (1.0 / (1.0 + tl.exp(-l3))).to(tl.bfloat16).to(tl.float32)
        acc = (
            (g0 * n0).to(tl.bfloat16).to(tl.float32)
            + (g1 * n1).to(tl.bfloat16).to(tl.float32)
            + (g2 * n2).to(tl.bfloat16).to(tl.float32)
            + (g3 * n3).to(tl.bfloat16).to(tl.float32)
        )
        tl.extra.cuda.gdc_launch_dependents()
        tl.store(output_ptr + row * HIDDEN + columns, acc / BRANCHES, mask=mask)
    else:
        acc = tl.full((BLOCK,), 0, tl.float32)
        for branch in tl.static_range(BRANCHES):
            offset = row * (BRANCHES * HIDDEN) + branch * HIDDEN + columns
            logits = tl.load(logits_ptr + offset, mask=mask, other=0).to(tl.float32)
            normed = tl.load(normed_ptr + offset, mask=mask, other=0).to(tl.float32)
            # The eager path materializes both sigmoid and the product in BF16.
            gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
            product = (gate * normed).to(tl.bfloat16).to(tl.float32)
            acc += product
        tl.store(output_ptr + row * HIDDEN + columns, acc / BRANCHES, mask=mask)


def is_supported(logits: torch.Tensor, normed: torch.Tensor, branches: int) -> bool:
    return (
        logits.is_cuda
        and torch.version.hip is None
        and logits.dtype == normed.dtype == torch.bfloat16
        and logits.shape == normed.shape
        and logits.device == normed.device
        and logits.is_contiguous()
        and normed.is_contiguous()
        and logits.dim() >= 2
        and branches == 4
        and logits.shape[-1] % branches == 0
    )


def fused_mix_reduce(
    logits: torch.Tensor, normed: torch.Tensor, branches: int
) -> torch.Tensor:
    if not is_supported(logits, normed, branches):
        raise ValueError("unsupported Qwen4 gated residual mix layout")
    hidden = int(logits.shape[-1]) // branches
    rows = logits.numel() // (branches * hidden)
    result = torch.empty(
        (*logits.shape[:-1], hidden), device=logits.device, dtype=logits.dtype
    )
    block, warps = 128, 4
    if hidden == 2560:
        capability = cached_device_capability(logits.device)
        if capability == (9, 0) and rows >= 4096:
            block, warps = 256, 2
        elif capability == (12, 0):
            # Fewer CTAs for the bandwidth-bound prefill consumer.
            if 4093 <= rows <= 4096:
                block = 2048
            elif rows == 12288:
                block, warps = 256, 2
    pdl = sm90_pdl_enabled(logits.device)
    _mix_reduce_kernel[(rows, triton.cdiv(hidden, block))](
        logits,
        normed,
        result,
        HIDDEN=hidden,
        BRANCHES=branches,
        BLOCK=block,
        PDL=pdl,
        num_warps=warps,
        launch_pdl=pdl,
    )
    return result


@triton.jit(do_not_specialize=["S_ROW_STRIDE", "S_COL_STRIDE"])
def _mix_reduce_fp8_kernel(
    logits_ptr,
    normed_ptr,
    output_ptr,
    q_ptr,
    s_ptr,
    partial_ptr,
    S_ROW_STRIDE,
    S_COL_STRIDE,
    HIDDEN: tl.constexpr,
    BRANCHES: tl.constexpr,
    GROUPS: tl.constexpr,
    MODE: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    columns = group * 128 + tl.arange(0, 128)
    mask = columns < HIDDEN
    base = row * (BRANCHES * HIDDEN) + columns
    if PDL:
        n0 = tl.load(normed_ptr + base, mask=mask, other=0).to(tl.float32)
        n1 = tl.load(normed_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        n2 = tl.load(normed_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        n3 = tl.load(normed_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        tl.extra.cuda.gdc_wait()
        l0 = tl.load(logits_ptr + base, mask=mask, other=0).to(tl.float32)
        l1 = tl.load(logits_ptr + base + HIDDEN, mask=mask, other=0).to(tl.float32)
        l2 = tl.load(logits_ptr + base + 2 * HIDDEN, mask=mask, other=0).to(tl.float32)
        l3 = tl.load(logits_ptr + base + 3 * HIDDEN, mask=mask, other=0).to(tl.float32)
        g0 = (1.0 / (1.0 + tl.exp(-l0))).to(tl.bfloat16).to(tl.float32)
        g1 = (1.0 / (1.0 + tl.exp(-l1))).to(tl.bfloat16).to(tl.float32)
        g2 = (1.0 / (1.0 + tl.exp(-l2))).to(tl.bfloat16).to(tl.float32)
        g3 = (1.0 / (1.0 + tl.exp(-l3))).to(tl.bfloat16).to(tl.float32)
        acc = (
            (g0 * n0).to(tl.bfloat16).to(tl.float32)
            + (g1 * n1).to(tl.bfloat16).to(tl.float32)
            + (g2 * n2).to(tl.bfloat16).to(tl.float32)
            + (g3 * n3).to(tl.bfloat16).to(tl.float32)
        )
    else:
        acc = tl.full((128,), 0, tl.float32)
        for branch in tl.static_range(BRANCHES):
            offset = row * (BRANCHES * HIDDEN) + branch * HIDDEN + columns
            logits = tl.load(logits_ptr + offset, mask=mask, other=0).to(tl.float32)
            normed = tl.load(normed_ptr + offset, mask=mask, other=0).to(tl.float32)
            gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
            product = (gate * normed).to(tl.bfloat16).to(tl.float32)
            acc += product

    mixed_bf16 = (acc / BRANCHES).to(tl.bfloat16)
    val = mixed_bf16.to(tl.float32)
    absolute = tl.abs(val)
    raw_max = tl.max(tl.where(absolute == absolute, absolute, 0.0), 0)
    amax = tl.maximum(raw_max, 1.0e-4)
    if MODE == 1:
        scale = amax * (1.0 / 448.0)
        code = val * tl.div_rn(448.0, amax)
        code = tl.where(code == code, code, -448.0)
        code = tl.minimum(448.0, tl.maximum(-448.0, code)).to(tl.float8e4nv)
    elif MODE == 0:
        scale = tl.div_rn(amax, 448.0)
        code = tl.div_rn(val, scale)
        code = tl.where(code == code, code, -448.0)
        code = tl.minimum(448.0, tl.maximum(-448.0, code)).to(tl.float8e4nv)
    else:
        scale = tl.div_rn(amax, 448.0)
        inv_bf16 = tl.div_rn(1.0, scale).to(tl.bfloat16).to(tl.float32)
        scaled = (val * inv_bf16).to(tl.bfloat16).to(tl.float32)
        scaled = tl.minimum(448.0, tl.maximum(-448.0, scaled))
        code = scaled.to(tl.float8e4nv)

    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    out_offset = row * HIDDEN + columns
    tl.store(output_ptr + out_offset, mixed_bf16, mask=mask)
    tl.store(q_ptr + out_offset, code, mask=mask)
    tl.store(s_ptr + row * S_ROW_STRIDE + group * S_COL_STRIDE, scale)
    tl.store(partial_ptr + row * GROUPS + group, raw_max)


def is_fp8_supported(
    logits: torch.Tensor, normed: torch.Tensor, branches: int
) -> bool:
    return (
        is_supported(logits, normed, branches)
        and logits.ndim == 2
        and logits.shape[1] == 10240
        and 1 <= logits.shape[0] <= 512
        and not logits.requires_grad
        and not normed.requires_grad
        and cached_device_capability(logits.device) == (9, 0)
    )


def fused_mix_reduce_fp8(
    logits: torch.Tensor, normed: torch.Tensor, branches: int, *, mode: str
) -> MixReduceFp8Bundle:
    if not is_fp8_supported(logits, normed, branches):
        raise ValueError("unsupported Qwen4 gated residual FP8 mix layout")
    mode_id = _FP8_MODE_IDS.get(mode)
    if mode_id is None:
        raise ValueError(f"unsupported Qwen4 gated residual FP8 mix mode: {mode}")
    rows = int(logits.shape[0])
    hidden = int(logits.shape[1]) // branches
    groups = hidden // 128
    result = torch.empty((rows, hidden), device=logits.device, dtype=logits.dtype)
    q = torch.empty((rows, hidden), device=logits.device, dtype=torch.float8_e4m3fn)
    if mode_id in (0, 1):
        s = create_per_token_group_quant_fp8_output_scale(
            (rows, hidden), logits.device, 128, True, True, False
        )
    else:
        s = torch.empty((rows, groups), device=logits.device, dtype=torch.float32)
    partial = torch.empty((rows * groups,), device=logits.device, dtype=torch.float32)
    pdl = sm90_pdl_enabled(logits.device)
    _mix_reduce_fp8_kernel[(rows, groups)](
        logits,
        normed,
        result,
        q,
        s,
        partial,
        s.stride(0),
        s.stride(1),
        HIDDEN=hidden,
        BRANCHES=branches,
        GROUPS=groups,
        MODE=mode_id,
        PDL=pdl,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return MixReduceFp8Bundle(x=result, q=q, s=s, partial=partial, mode=mode)
