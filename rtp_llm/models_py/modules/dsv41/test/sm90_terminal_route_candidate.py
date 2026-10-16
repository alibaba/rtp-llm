"""Test-only exact terminal-offset MoE candidate; NOT a production backend.

The current installed Humming grouped scheduler derives the final expert's
end from shape_m, not offsets[E]. Therefore this helper alone does not skip
dummy GEMM rows. It supplies a correctness harness for a backend that honors
the device terminal count, including safe uninitialized-output boundaries.
The extra torch sum/copy overhead is included in any whole-chain timing.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.moe._sm90_route_kernels import (
    prepare_static_routes,
    reduce_static_routes,
)


@triton.jit
def _terminal_swiglu_kernel(
    gate_up,
    weights,
    slots,
    valid,
    output,
    D: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row)
    keep = tl.load(valid + slot)
    # A terminal-count backend leaves gate_up[V:capacity] unwritten. Mask
    # before loading, not after multiplying by zero: NaN * 0 remains NaN.
    g = tl.load(gate_up + row * ROW_STRIDE + col, keep & (col < D), other=0).to(
        tl.float32
    )
    u = tl.load(gate_up + row * ROW_STRIDE + D + col, keep & (col < D), other=0).to(
        tl.float32
    )
    if CLAMP_LIMIT > 0:
        u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
        u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
        g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)
    # Preserve routed_swiglu's operation order, FP32 math and weighted-BF16
    # boundary BEFORE the existing w2 G128 activation quantization.
    s = g * tl.sigmoid(g)
    hidden = s * u
    weight = tl.load(weights + slot, keep, other=0).to(tl.float32)
    weighted = hidden * weight
    # Valid NaNs must propagate. Only invalid rows are replaced with zeros.
    tl.store(output + row * D + col, tl.where(keep, weighted, 0), col < D)


def masked_terminal_swiglu(
    gate_up, weights, slots, valid, *, clamp_limit, output_dtype
):
    """Independent test seam for poison-tail and valid-NaN coverage."""
    if (
        gate_up.ndim != 2
        or gate_up.shape[1] % 2
        or gate_up.stride(1) != 1
        or gate_up.dtype not in (torch.bfloat16, torch.float32)
        or slots.shape != (gate_up.shape[0],)
        or valid.shape != slots.shape
        or valid.dtype != torch.bool
        or not slots.is_contiguous()
        or not valid.is_contiguous()
        or output_dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("Unsupported terminal SwiGLU test layout")
    rows, width = gate_up.shape
    dim = width // 2
    output = torch.empty((rows, dim), device=gate_up.device, dtype=output_dtype)
    if rows and dim:
        block = min(1024, triton.next_power_of_2(dim))
        _terminal_swiglu_kernel[(rows, triton.cdiv(dim, block))](
            gate_up,
            weights.reshape(-1),
            slots,
            valid,
            output,
            dim,
            gate_up.stride(0),
            float(clamp_limit),
            block,
            num_warps=4,
            num_stages=2,
        )
    return output


def forward_local_terminal(
    local,
    x,
    weights,
    indices,
    start,
    end,
    *,
    output_dtype=torch.float32,
):
    """Fixed-capacity route tensors, dynamic device terminal count, native A8.

    No host read of routes/counts, dynamic tensor shape, alternate quantizer,
    repacked weights or changed expert-reduction order is introduced.
    """
    if (
        not local.native_a8
        or any(linear.promotion_group_size != 32 for linear in (local.w13, local.w2))
        or indices.numel() > 256
        or not 0 < end - start <= 256
        or indices.shape != weights.shape
        or indices.shape[0] != x.shape[0]
        or not 1 <= indices.shape[1] <= 8
        or output_dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError(
            "Terminal route candidate requires bounded native-A8/G32 SM90 routes"
        )
    if x.shape[0] == 0:
        return torch.zeros(x.shape, device=x.device, dtype=output_dtype)
    slots, offsets, valid, inverse, grouped_x, clean_weights = prepare_static_routes(
        x,
        weights,
        indices,
        start,
        end,
    )
    # The original route kernel already computes exact prefixes [0:E]. Only
    # its capacity sentinel is changed; route permutation and inverse stay
    # identical. This test-only two-op update is not claimed to be optimal.
    offsets[-1:].copy_(valid.sum(dtype=torch.int32).reshape(1))
    gate_up = local.w13(grouped_x, offsets)
    hidden = masked_terminal_swiglu(
        gate_up,
        clean_weights,
        slots,
        valid,
        clamp_limit=local.cfg.swiglu_limit,
        output_dtype=x.dtype,
    )
    down = local.w2(hidden, offsets)
    # Existing reduction masks invalid original slots BEFORE loading down;
    # unwritten dummy output rows are never consumed, even for V=0.
    return reduce_static_routes(
        down,
        inverse,
        valid,
        x.shape[0],
        indices.shape[1],
        output_dtype=output_dtype,
    )
