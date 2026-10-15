# Fused sigmoid-gate elementwise multiply for attention output gating.
#
# Contract:
#   out[t, h] = attn_output[t, h] * sigmoid(gate[t, h]), computed in-place on
#   attn_output. Semantics replicate the aten chain
#   ``attn_output * torch.sigmoid(gate)`` bit-exactly: sigmoid is evaluated in
#   fp32 and rounded through the gate dtype (matching aten's bf16/fp16 sigmoid
#   output rounding) before the fp32 elementwise multiply and the single final
#   rounding to the output dtype.
#
# Shapes/dtypes:
#   attn_output: [T, H] (any leading dims flattened by the wrapper), last dim
#                contiguous; bf16/fp16/fp32.
#   gate:        [T, H] with the same H, arbitrary row stride (e.g. a column
#                slice of a fused qkv+gate projection), same dtype family.
#
# Grid follows the moe_gating.py BLOCK_H pattern: (T, ceil(H / BLOCK_H)) with
# BLOCK_H chosen to keep enough programs in flight for small T (decode) and
# coalesced >=256B accesses for large T (prefill).

import torch
import triton
import triton.language as tl

_MIN_TOTAL_PROGRAMS = 512
_MIN_BLOCK_H = 128
_MAX_BLOCK_H = 4096


@triton.jit
def _SigmoidMul_kernel(
    gate_ptr,  # [T, H] gate logits (row stride stride_gate_t)
    out_ptr,  # [T, H] attention output, updated in-place
    T,
    H,
    stride_gate_t,  # gate stride along the token dim (elements)
    stride_out_t,  # out stride along the token dim (elements)
    BLOCK_H: tl.constexpr,
):
    tid = tl.program_id(axis=0)  # token index
    hid = tl.program_id(axis=1)  # hidden block index

    h_offsets = hid * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    gate_base = gate_ptr + tid * stride_gate_t
    out_base = out_ptr + tid * stride_out_t

    g = tl.load(gate_base + h_offsets, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    # aten's bf16/fp16 sigmoid rounds the fp32 result to the gate dtype before
    # the separate elementwise multiply; replicate that rounding so the fused
    # kernel is bit-identical to ``out * torch.sigmoid(gate)``.
    sig = sig.to(gate_base.dtype.element_ty).to(tl.float32)

    a = tl.load(out_base + h_offsets, mask=mask, other=0.0).to(tl.float32)
    res = a * sig
    tl.store(out_base + h_offsets, res.to(out_base.dtype.element_ty), mask=mask)


def _select_block_h(T: int, H: int) -> int:
    """Choose BLOCK_H to balance SM utilisation and memory coalescing."""
    target_h_blocks = max(1, _MIN_TOTAL_PROGRAMS // max(T, 1))
    ideal = triton.next_power_of_2(max(1, H // target_h_blocks))
    return max(_MIN_BLOCK_H, min(_MAX_BLOCK_H, ideal))


def sigmoid_mul_(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """In-place ``attn_output = attn_output * sigmoid(gate)`` (fused, bit-exact).

    Args:
        attn_output: attention output [..., H], last dim contiguous; written
            in-place (the buffer must not alias ``gate``).
        gate: gate logits [..., H]; may be a strided column slice of a fused
            projection output (e.g. ``fused[..., qkv_size:]``).

    Returns:
        The ``attn_output`` tensor (same object, updated in-place).
    """
    H = attn_output.shape[-1]
    out2d = attn_output.view(-1, H)
    gate2d = gate.reshape(-1, gate.shape[-1])
    assert out2d.shape[0] == gate2d.shape[0] and gate2d.shape[1] == H, (
        f"attn_output rows/H must match gate, got {tuple(out2d.shape)} vs {tuple(gate2d.shape)}"
    )
    assert out2d.stride(-1) == 1 and gate2d.stride(-1) == 1
    T = out2d.shape[0]
    BLOCK_H = _select_block_h(T, H)
    grid = (T, triton.cdiv(H, BLOCK_H))
    _SigmoidMul_kernel[grid](
        gate2d,
        out2d,
        T,
        H,
        gate2d.stride(0),
        out2d.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return attn_output
