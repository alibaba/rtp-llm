# SPDX-License-Identifier: Apache-2.0
# Adapted from sgl-project/sglang, commit 297e54fd105e974cdf401e5b4d20253a22c16f90,
# python/sglang/kernels/ops/elementwise/elementwise.py.
"""Fuse shared gate projection, sigmoid, multiplication and local MoE addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,
    gate_weight_ptr,
    shared_output_ptr,
    output_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    f = tl.load(output_ptr + row_offset + offsets, mask=mask, other=0.0).to(tl.float32)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s + f
    tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def fused_gate_sigmoid_mul_add(hidden_states, gate_weight, shared_output, output):
    rows, hidden = hidden_states.shape
    assert hidden_states.is_contiguous() and gate_weight.is_contiguous()
    assert shared_output.is_contiguous() and output.is_contiguous()
    assert gate_weight.numel() == hidden
    assert shared_output.shape == output.shape == hidden_states.shape
    if rows == 0:
        return
    warps = max(min(triton.next_power_of_2(triton.cdiv(hidden, 256)), 32), 4)
    if rows >= 1024:
        warps = min(warps, 8)
    use_pdl = torch.cuda.get_device_capability(hidden_states.device)[0] >= 9
    _fused_gate_sigmoid_mul_add_kernel[(rows,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        hidden_dim=hidden,
        BLOCK_SIZE=triton.next_power_of_2(hidden),
        USE_PDL=use_pdl,
        num_warps=warps,
        **({"launch_pdl": True} if use_pdl else {}),
    )
