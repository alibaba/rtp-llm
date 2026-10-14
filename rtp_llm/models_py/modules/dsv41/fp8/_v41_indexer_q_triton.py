# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Alibaba Group Holding Limited
"""V4.1 indexer Q: gather RoPE, MXFP4 quantization and FP32 head weights.

Adapted from vLLM's per-(token, head) MXFP4 Triton fusion in
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py. One launch reads
the original Q and writes packed Q, UE8M0 scales and scaled weights.
Necessary RTP differences: complex64 interleaved frequency storage, the
existing CUDA complex-multiply FMA order, the exact existing quantizer, and
one combined 1/64 weight multiplier (not two separately rounded factors).
Those operations reuse RTP helpers. FP4 block scales remain separate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._v41_fp4_triton import _pack_e2m1_payload, _round_power_of_two_scale


@triton.jit
def _indexer_rope_bf16(q, frequencies, position):
    columns = tl.arange(0, 128)
    values = q.to(tl.float32)
    real = tl.gather(values, columns & ~1, axis=0)
    imag = tl.gather(values, columns | 1, axis=0)
    rotary = columns >= 64
    angle = tl.maximum((columns - 64) // 2, 0)
    cosine = tl.load(frequencies + position * 64 + angle * 2, rotary, other=1)
    sine = tl.load(frequencies + position * 64 + angle * 2 + 1, rotary, other=0)
    even = columns % 2 == 0
    product = tl.where(even, imag, real) * sine
    # PyTorch c10 complex multiplication: fma(a,c,-b*d), fma(b,c,a*d).
    # Triton unary minus can lower to 0-product, losing the negative zero
    # required by FP4's sign bit. Flip the bit instead of subtracting from 0.
    product_bits = product.to(tl.uint32, bitcast=True)
    signed_product = (product_bits ^ tl.where(even, 0x80000000, 0)).to(
        tl.float32, bitcast=True
    )
    rotated = tl.fma(tl.where(even, real, imag), cosine, signed_product)
    return tl.where(rotary, rotated, values).to(tl.bfloat16)


@triton.jit
def _fused_indexer_q_rope_fp4_kernel(
    q_ptr,
    weights_ptr,
    frequencies_ptr,
    positions_ptr,
    payload_ptr,
    sf_ptr,
    weights_out_ptr,
    MAX_POSITION: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    token = row // 32
    position = tl.load(positions_ptr + token).to(tl.int64)
    # Match torch indexing for negative positions; malformed indices must not
    # become out-of-bounds frequency reads in a captured graph.
    position = tl.where(position < 0, position + MAX_POSITION, position)
    if (position < 0) | (position >= MAX_POSITION):
        tl.inline_asm_elementwise(
            "trap; // dummy $0", "=r", [], dtype=tl.int32, is_pure=False, pack=1
        )
        return
    q = tl.load(q_ptr + row * 128 + tl.arange(0, 128))
    # This helper matches eager CUDA complex multiplication, including the
    # explicit FMA and the BF16 rounding before the FP4 block reduction.
    rotated = _indexer_rope_bf16(q, frequencies_ptr, position).to(tl.float32)
    groups = rotated.reshape((4, 32))
    maximum = tl.maximum(tl.max(tl.abs(groups), axis=1), 6.0 * (2.0**-126))
    scale, scale_bytes = _round_power_of_two_scale(maximum, 1.0 / 6.0)
    normalized = tl.minimum(tl.maximum(tl.div_rn(groups, scale[:, None]), -6.0), 6.0)
    payload = _pack_e2m1_payload(normalized, 128)
    tl.store(payload_ptr + row * 64 + tl.arange(0, 64), payload.to(tl.int8))
    sf_bytes = (sf_ptr + row).to(tl.pointer_type(tl.uint8))
    tl.store(sf_bytes + tl.arange(0, 4), scale_bytes)
    weight = tl.load(weights_ptr + row).to(tl.float32)
    tl.store(weights_out_ptr + row, weight * (1.0 / 64.0))


def is_supported(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int = 64,
) -> bool:
    """Only the production BF16/32-head/MXFP4 decode contract is fused."""
    if (
        q.device.type != "cuda"
        or q.ndim != 4
        or q.shape[2:] != (32, 128)
        or q.shape[0] < 1
        or q.shape[1] < 1
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
        or rope_head_dim != 64
    ):
        return False
    if (
        weights.shape != q.shape[:-1]
        or weights.dtype not in (torch.bfloat16, torch.float32)
        or freqs_cis.ndim != 2
        or freqs_cis.shape[0] < 1
        or freqs_cis.shape[1] != 32
        or freqs_cis.dtype != torch.complex64
        or positions.shape != (q.shape[0] * q.shape[1],)
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        return False
    for value in (weights, freqs_cis, positions):
        if value.device != q.device or not value.is_contiguous():
            return False
    if torch.is_grad_enabled() and any(
        value.requires_grad for value in (q, weights, freqs_cis)
    ):
        return False
    return torch.cuda.get_device_capability(q.device)[0] == 10


def try_fused_indexer_q(
    q: torch.Tensor,
    weights: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    rope_head_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return (int8 Q, packed-int32 SF, FP32 weights), or None to fall back.

    Weights are the unscaled BF16/FP32 projection. Frequencies are the complete
    complex64 lookup table, positions contain one index for each B*S token.
    Inputs are never modified. Execution/compilation errors are not swallowed.
    """
    if not is_supported(q, weights, freqs_cis, positions, rope_head_dim):
        return None
    payload = torch.empty(*q.shape[:-1], 64, dtype=torch.int8, device=q.device)
    sf = torch.empty(q.shape[:-1], dtype=torch.int32, device=q.device)
    scaled_weights = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    _fused_indexer_q_rope_fp4_kernel[(q.numel() // 128,)](
        q,
        weights,
        torch.view_as_real(freqs_cis),
        positions,
        payload,
        sf,
        scaled_weights,
        freqs_cis.shape[0],
        num_warps=4,
        enable_fp_fusion=False,
    )
    return payload, sf, scaled_weights
