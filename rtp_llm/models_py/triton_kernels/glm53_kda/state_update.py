"""Experimental pointwise KDA update; preserve separate FP32 rounding."""

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

_ENABLED = os.environ.get("GLM53_KDA_FUSED_STATE_UPDATE", "0") == "1"
_WRITE_CHECKPOINTS = os.environ.get("GLM53_KDA_WRITE_CHECKPOINTS", "0") == "1"


@triton.jit
def _state_update(state, delta, gate, output, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * (BLOCK // 128) + tl.arange(
        0, BLOCK // 128
    ).to(tl.int64)
    cols = tl.arange(0, 128).to(tl.int64)
    i = rows[:, None] * 128 + cols[None, :]
    valid = rows < SIZE // 128
    log_decay = tl.load(gate + rows, valid, 0).to(tl.float32)
    decay = libdevice.exp2(log_decay)
    value = tl.load(state + i, valid[:, None], 0)
    increment = tl.load(delta + i, valid[:, None], 0)
    decayed = libdevice.mul_rn(value, decay[:, None])
    result = libdevice.add_rn(decayed, increment)
    tl.store(output + i, result, valid[:, None])


def kda_state_update(state, delta, log_decay, *, output=None, _block=512):
    """Write an independent FP32 state, or return None outside the supported domain."""
    if not _ENABLED or torch.version.hip is not None:
        return None
    values = (state, delta, log_decay)
    if (
        any(not t.is_cuda or not t.is_contiguous() or t.requires_grad for t in values)
        or any(t.device != state.device for t in values)
        or state.ndim != 3
        or not 1 <= state.shape[0] <= 128
        or state.shape[1:] != (128, 128)
        or delta.shape != state.shape
        or state.dtype != torch.float32
        or delta.dtype != torch.float32
        or log_decay.shape != state.shape[:2]
        or log_decay.dtype not in (torch.float32, torch.bfloat16)
        or _block not in (256, 512, 1024, 2048)
    ):
        return None
    if output is None:
        output = torch.empty_like(state)
    else:
        if (
            output.device != state.device
            or output.dtype != state.dtype
            or output.shape != state.shape
            or not output.is_contiguous()
            or output.requires_grad
        ):
            return None
        begin = output.data_ptr()
        end = begin + output.numel() * output.element_size()
        # Adjacent checkpoint slots can share storage but must not overlap
        # any input. Partial overlap would introduce inter-program races.
        for value in values:
            source = value.data_ptr()
            if begin < source + value.numel() * value.element_size() and source < end:
                return None
    _state_update[(triton.cdiv(state.numel(), _block),)](
        state,
        delta,
        log_decay,
        output,
        state.numel(),
        _block,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return output
