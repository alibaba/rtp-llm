"""
Fused Scaled Dot Product Gate for MHC

Fuses the following operations:
1. gate = (key_norm * query_norm).sum(dim=-1) / sqrt(hidden_size)
2. gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
3. gate = gate.sigmoid()

Input shapes:
- key_norm: [hc_mult, num_tokens, hidden_size]
- query_norm: [hc_mult, num_tokens, hidden_size]
Output shape:
- gate: [hc_mult, num_tokens]
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def scaled_dot_product_gate_kernel(
    output_ptr,
    key_norm_ptr,
    query_norm_ptr,
    hc_mult: tl.constexpr,
    num_tokens: tl.constexpr,
    hidden_size: tl.constexpr,
    rsqrt_hidden_size,  # 1 / sqrt(hidden_size)
    eps: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # hidden_size 维度的 block size
):
    """
    Fused kernel for scaled dot product gate computation

    Grid: (hc_mult, num_tokens)
    Each block processes one (hc_mult, token) position
    """
    # 2D grid: (hc_mult_id, token_id)
    hc_id = tl.program_id(0)
    token_id = tl.program_id(1)

    # 计算该位置在输入张量中的起始偏移
    # key_norm/query_norm shape: [hc_mult, num_tokens, hidden_size]
    row_offset = hc_id * num_tokens * hidden_size + token_id * hidden_size

    # ========== PASS 1: 计算 dot product ==========
    # dot_product = sum(key_norm * query_norm) over hidden_size dimension
    dot_product = 0.0

    for block_start in range(0, hidden_size, BLOCK_SIZE_H):
        h_offsets = block_start + tl.arange(0, BLOCK_SIZE_H)
        mask = h_offsets < hidden_size

        key_ptrs = key_norm_ptr + row_offset + h_offsets
        query_ptrs = query_norm_ptr + row_offset + h_offsets

        key = tl.load(key_ptrs, mask=mask, other=0.0).to(tl.float32)
        query = tl.load(query_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Element-wise multiplication and accumulate
        dot_product += tl.sum(tl.where(mask, key * query, 0.0))

    # ========== Scaled dot product ==========
    # gate = dot_product / sqrt(hidden_size)
    gate = dot_product * rsqrt_hidden_size

    # ========== Apply abs().clamp_min(eps).sqrt() * sign() ==========
    # sign = 1 if gate >= 0 else -1
    sign = tl.where(gate >= 0.0, 1.0, -1.0)

    # abs_gate = abs(gate)
    abs_gate = tl.abs(gate)

    # clamp_min(eps)
    abs_gate = tl.maximum(abs_gate, eps)

    # sqrt
    abs_gate_sqrt = tl.sqrt(abs_gate)

    # multiply by sign
    gate = abs_gate_sqrt * sign

    # ========== Apply sigmoid ==========
    # sigmoid(x) = 1 / (1 + exp(-x))
    gate = tl.sigmoid(gate)

    # ========== Store result ==========
    # output shape: [hc_mult, num_tokens]
    output_offset = hc_id * num_tokens + token_id
    output_ptr_loc = output_ptr + output_offset
    tl.store(output_ptr_loc, gate)


def scaled_dot_product_gate(
    key_norm: torch.Tensor,
    query_norm: torch.Tensor,
    hidden_size: int,
    eps: float = 1e-6,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused scaled dot product gate computation

    Computes:
        gate = (key_norm * query_norm).sum(dim=-1) / sqrt(hidden_size)
        gate = gate.abs().clamp_min(eps).sqrt() * gate.sign()
        gate = gate.sigmoid()

    Args:
        key_norm: [hc_mult, num_tokens, hidden_size]
        query_norm: [hc_mult, num_tokens, hidden_size]
        hidden_size: size of hidden dimension
        eps: minimum value for clamping (default: 1e-6)
        output: optional pre-allocated output tensor [hc_mult, num_tokens]

    Returns:
        gate: [hc_mult, num_tokens]
    """
    assert key_norm.dim() == 3, f"Expected 3D key_norm, got {key_norm.dim()}D"
    assert query_norm.dim() == 3, f"Expected 3D query_norm, got {query_norm.dim()}D"
    assert (
        key_norm.shape == query_norm.shape
    ), f"key_norm shape {key_norm.shape} != query_norm shape {query_norm.shape}"

    hc_mult, num_tokens, h_size = key_norm.shape
    assert (
        h_size == hidden_size
    ), f"Input hidden_size {h_size} doesn't match specified hidden_size {hidden_size}"

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (hc_mult, num_tokens), dtype=key_norm.dtype, device=key_norm.device
        )
    else:
        assert output.shape == (
            hc_mult,
            num_tokens,
        ), f"Output shape {output.shape} doesn't match expected ({hc_mult}, {num_tokens})"

    # Pre-compute 1 / sqrt(hidden_size)
    rsqrt_hidden_size = 1.0 / (hidden_size**0.5)

    if hidden_size <= 256:
        BLOCK_SIZE_H = 256
    elif hidden_size <= 512:
        BLOCK_SIZE_H = 512
    elif hidden_size <= 1024:
        BLOCK_SIZE_H = 512
    elif hidden_size <= 2048:
        BLOCK_SIZE_H = 1024
    elif hidden_size <= 4096:
        BLOCK_SIZE_H = 1024
    elif hidden_size <= 8192:
        BLOCK_SIZE_H = 2048
    else:
        BLOCK_SIZE_H = 2048

    # Grid: (hc_mult, num_tokens)
    grid = (hc_mult, num_tokens)

    scaled_dot_product_gate_kernel[grid](
        output,
        key_norm,
        query_norm,
        hc_mult,
        num_tokens,
        hidden_size,
        rsqrt_hidden_size,
        eps,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )

    return output
