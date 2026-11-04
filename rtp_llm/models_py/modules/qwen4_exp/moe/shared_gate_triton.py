"""Preserve the BF16 sigmoid boundary while fusing the shared-expert gate."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _shared_sigmoid_mul(
    gate, shared, output, N: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = row * H + columns
    valid = columns < H
    logits = tl.load(gate + row).to(tl.float32)
    values = tl.load(shared + offsets, valid, other=0).to(tl.float32)
    # torch.sigmoid returns BF16 before the separate BF16 multiply. Keeping
    # that intermediate rounding is required before the TP all-reduce.
    probability = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    probability = probability.to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, probability * values, valid)


def is_supported(gate, shared):
    return (
        isinstance(gate, torch.Tensor)
        and isinstance(shared, torch.Tensor)
        and shared.is_cuda
        and torch.version.hip is None
        and gate.device == shared.device
        and gate.dtype == shared.dtype == torch.bfloat16
        and shared.ndim == 2
        and 1 <= shared.shape[0] <= 32
        and shared.shape[1] == 2560
        and gate.shape == (shared.shape[0], 1)
        and gate.is_contiguous()
        and shared.is_contiguous()
        and torch.cuda.get_device_capability(shared.device) == (9, 0)
    )


def try_shared_sigmoid_mul(gate, shared):
    if not is_supported(gate, shared):
        return None
    output = torch.empty_like(shared)
    _shared_sigmoid_mul[(shared.shape[0], triton.cdiv(shared.shape[1], 512))](
        gate,
        shared,
        output,
        N=shared.shape[0],
        H=shared.shape[1],
        BLOCK=512,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output
