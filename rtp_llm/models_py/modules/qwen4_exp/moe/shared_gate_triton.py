"""Preserve the BF16 sigmoid boundary while fusing the shared-expert gate."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.utils.device_capability import sm90_pdl_enabled


@triton.jit
def _shared_sigmoid_mul(
    gate,
    shared,
    output,
    N: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * H + columns
    valid = columns < H
    logits = tl.load(gate + row).to(tl.float32)
    # torch.sigmoid returns BF16 before the separate BF16 multiply. Keeping
    # that intermediate rounding is required before the TP all-reduce.
    probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    probability = probability.to(tl.bfloat16).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    values = tl.load(shared + offsets, valid, other=0).to(tl.float32)
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(output + offsets, probability * values, valid)


def is_supported(gate, shared, out=None):
    return (
        isinstance(gate, torch.Tensor)
        and isinstance(shared, torch.Tensor)
        and shared.is_cuda
        and torch.version.hip is None
        and gate.device == shared.device
        and gate.dtype == shared.dtype == torch.bfloat16
        and shared.ndim == 2
        and 1 <= shared.shape[0] <= 32768
        and shared.shape[1] == 2560
        and gate.shape == (shared.shape[0], 1)
        and gate.is_contiguous()
        and shared.is_contiguous()
        and (
            out is None
            or (
                isinstance(out, torch.Tensor)
                and out.device == shared.device
                and out.dtype == torch.bfloat16
                and out.shape == shared.shape
                and out.is_contiguous()
            )
        )
        and torch.cuda.get_device_capability(shared.device) == (9, 0)
    )


def try_shared_sigmoid_mul(gate, shared, *, out=None):
    if not is_supported(gate, shared, out=out):
        return None
    output = torch.empty_like(shared) if out is None else out
    pdl = sm90_pdl_enabled(shared.device)
    _shared_sigmoid_mul[(shared.shape[0], triton.cdiv(shared.shape[1], 512))](
        gate,
        shared,
        output,
        N=shared.shape[0],
        H=shared.shape[1],
        BLOCK=512,
        PDL=pdl,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return output
