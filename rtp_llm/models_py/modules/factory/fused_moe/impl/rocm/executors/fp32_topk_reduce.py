"""Fixed-order expert reduction for gfx942 per-channel FP8 MoE."""

import torch
import triton
import triton.language as tl
from aiter.fused_moe import fused_moe


@triton.jit
def _quantize_expand(
    inputs, quantized, scales, WIDTH: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    values = tl.load(inputs + token * WIDTH + offsets, offsets < WIDTH, 0).to(tl.float32)
    amax = tl.max(tl.abs(values), 0)
    # Match AITER's float32 multiplication by 1/240 and amdgcn_rcpf.
    # Replacing these with division changes rounding at FP8 bin boundaries.
    scale = amax * tl.full((), 0.004166666883975267, tl.float32)
    inverse = tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        constraints="=v,v",
        args=[scale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    quantized_values = tl.where(amax == 0, 0.0, values * inverse).to(
        quantized.dtype.element_ty
    )
    # Read and quantize each input row once; materialize only FP8 copies.
    for expert_slot in range(TOPK):
        row = token * TOPK + expert_slot
        tl.store(quantized + row * WIDTH + offsets, quantized_values, offsets < WIDTH)
        tl.store(scales + row, scale)


@triton.jit
def _reduce_topk(
    inputs, output, WIDTH: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    accumulator = tl.full((BLOCK,), 0, tl.float32)
    for expert_slot in range(TOPK):
        contribution = tl.load(
            inputs + (token * TOPK + expert_slot) * WIDTH + offsets,
            offsets < WIDTH,
            0,
        ).to(tl.float32)
        accumulator = accumulator + contribution
    # Convert once, without an intermediate float32 tensor or a cast kernel.
    tl.store(output + token * WIDTH + offsets, accumulator, offsets < WIDTH)


def fp8_moe_fp32_topk_reduce(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Give each expert its own output row before fixed-order FP32 reduction.

    The caller restricts this to gfx942, BF16 input, FNUZ FP8 weights and EP=1.
    AITER still computes and weights each expert contribution in its existing
    kernel. Only the cross-expert accumulation and input staging change.
    """
    tokens, width = hidden_states.shape
    topk = topk_ids.shape[1]
    if tokens == 0:
        return hidden_states.new_empty((0, w2.shape[1]))
    with torch.cuda.device(hidden_states.device):
        expanded = torch.empty(
            (tokens * topk, width), device=hidden_states.device, dtype=w1.dtype
        )
        scales = torch.empty(
            (tokens * topk, 1), device=hidden_states.device, dtype=torch.float32
        )
        _quantize_expand[(tokens,)](
            hidden_states, expanded, scales, width, topk, triton.next_power_of_2(width)
        )
        contributions = fused_moe(
            expanded,
            w1,
            w2,
            topk_weights.reshape(-1, 1).contiguous(),
            topk_ids.reshape(-1, 1).contiguous(),
            a1_scale=scales,
            dtype=hidden_states.dtype,
            **kwargs,
        )
        result = hidden_states.new_empty((tokens, contributions.shape[-1]))
        _reduce_topk[(tokens, triton.cdiv(result.shape[-1], 256))](
            contributions, result, result.shape[-1], topk, 256
        )
    return result
