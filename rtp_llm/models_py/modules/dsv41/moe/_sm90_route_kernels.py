"""Exact elementwise/route reduction glue for SM90 grouped experts.

No GEMM or quantizer is changed. SwiGLU retains the existing split kernel's
FP32 math, route weights are applied before the original BF16/down quantizer
boundary. The inverse-route reduction reproduces ATen CUDA Reduce.cuh's
four-accumulator sum for a short, non-innermost reduction dimension.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _static_route_metadata_kernel(
    indices,
    slots,
    offsets,
    valid_out,
    inverse,
    TOPK: tl.constexpr,
    CAPACITY: tl.constexpr,
    IDX_ROW_STRIDE: tl.constexpr,
    IDX_COL_STRIDE: tl.constexpr,
    LOCAL_START: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
    EXPERT_BLOCK: tl.constexpr,
):
    route = tl.arange(0, BLOCK)
    expert = (
        tl.load(
            indices
            + (route // TOPK) * IDX_ROW_STRIDE
            + (route % TOPK) * IDX_COL_STRIDE,
            route < CAPACITY,
            other=-1,
        ).to(tl.int64)
        - LOCAL_START
    )
    valid = (route < CAPACITY) & (expert >= 0) & (expert < EXPERTS)
    # Slot is part of the key, so sorting is stable even for repeated experts.
    key = tl.where(valid, expert, EXPERTS) * (CAPACITY + 1) + route
    key = tl.where(route < CAPACITY, key, 0x7FFFFFFFFFFFFFFF)
    ordered = tl.sort(key, descending=False)
    slot = ordered % (CAPACITY + 1)
    tl.store(slots + route, slot, route < CAPACITY)
    tl.store(inverse + slot, route, route < CAPACITY)
    tl.store(valid_out + route, valid, route < CAPACITY)
    boundary = tl.arange(0, EXPERT_BLOCK)
    prefix = tl.sum(
        (valid[None, :] & (expert[None, :] < boundary[:, None])).to(tl.int32),
        axis=1,
    )
    # All invalid physical rows belong to the final expert's capacity.
    prefix = tl.where(boundary == EXPERTS, CAPACITY, prefix)
    tl.store(offsets + boundary, prefix, boundary <= EXPERTS)


def static_route_metadata(indices, local_start, local_end):
    """Bounded device-only stable routes; inverse includes invalid dummy slots.

    No route values are inspected on the host. This helper intentionally
    supports at most 256 physical slots; larger eager/prefill routing stays
    on its original implementation. Every returned tensor owns fresh storage.
    """
    if (
        not indices.is_cuda
        or indices.ndim != 2
        or indices.dtype not in (torch.int32, torch.int64)
        or not 1 <= indices.shape[1] <= 8
        or local_end <= local_start
        or local_end - local_start > 256
        or indices.numel() > 256
    ):
        raise ValueError("unsupported bounded SM90 static route metadata")
    capacity = indices.numel()
    experts = local_end - local_start
    slots = torch.empty(capacity, device=indices.device, dtype=torch.int64)
    offsets = torch.empty(experts + 1, device=indices.device, dtype=torch.int32)
    valid = torch.empty(capacity, device=indices.device, dtype=torch.bool)
    inverse = torch.empty(capacity, device=indices.device, dtype=torch.int32)
    if capacity:
        _static_route_metadata_kernel[(1,)](
            indices,
            slots,
            offsets,
            valid,
            inverse,
            indices.shape[1],
            capacity,
            *indices.stride(),
            local_start,
            experts,
            triton.next_power_of_2(capacity),
            triton.next_power_of_2(experts + 1),
            num_warps=4,
        )
    else:
        offsets.zero_()
    return slots, offsets, valid, inverse


@triton.jit
def _static_gather_kernel(
    x,
    weights,
    slots,
    valid,
    grouped_x,
    clean_weights,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr,
    X_COL_STRIDE: tl.constexpr,
    W_ROW_STRIDE: tl.constexpr,
    W_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + route)
    keep = tl.load(valid + slot)
    row = slot // TOPK
    value = tl.load(
        x + row * X_ROW_STRIDE + col * X_COL_STRIDE,
        keep & (col < D),
        other=0,
    )
    tl.store(grouped_x + route * D + col, value, col < D)
    if tl.program_id(1) == 0:
        weight = tl.load(
            weights + row * W_ROW_STRIDE + (slot % TOPK) * W_COL_STRIDE,
            keep,
            other=0,
        )
        # Keep original-slot layout for the unchanged weighted SwiGLU.
        tl.store(clean_weights + slot, weight)


def prepare_static_routes(x, weights, indices, local_start, local_end):
    """Metadata plus NaN-safe gather/weight cleanup in two GPU launches."""
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[1] <= 1
        or x.dtype not in (torch.bfloat16, torch.float32)
        or weights.device != x.device
        or indices.device != x.device
        or weights.shape != indices.shape
        or indices.ndim != 2
        or indices.shape[0] != x.shape[0]
        or weights.dtype not in (torch.float32, torch.bfloat16)
    ):
        raise ValueError("unsupported SM90 static gather layout")
    slots, offsets, valid, inverse = static_route_metadata(
        indices, local_start, local_end
    )
    grouped_x = torch.empty(
        (indices.numel(), x.shape[1]), device=x.device, dtype=x.dtype
    )
    clean_weights = torch.empty(weights.shape, device=x.device, dtype=weights.dtype)
    if indices.numel():
        _static_gather_kernel[(indices.numel(), triton.cdiv(x.shape[1], 256))](
            x,
            weights,
            slots,
            valid,
            grouped_x,
            clean_weights,
            x.shape[1],
            indices.shape[1],
            *x.stride(),
            *weights.stride(),
            256,
            num_warps=4,
        )
    return slots, offsets, valid, inverse, grouped_x, clean_weights


@triton.jit
def _routed_swiglu_kernel(
    gate_up,
    weights,
    slots,
    output,
    D: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    CLAMP_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(gate_up + row * ROW_STRIDE + col, col < D, other=0).to(tl.float32)
    u = tl.load(gate_up + row * ROW_STRIDE + D + col, col < D, other=0).to(tl.float32)
    if CLAMP_LIMIT > 0:
        u = tl.where(u > CLAMP_LIMIT, CLAMP_LIMIT, u)
        u = tl.where(u < -CLAMP_LIMIT, -CLAMP_LIMIT, u)
        g = tl.where(g > CLAMP_LIMIT, CLAMP_LIMIT, g)
    # Same arithmetic and FP32 intermediate boundaries as silu_mul_split.
    s = g * tl.sigmoid(g)
    hidden = s * u
    slot = tl.load(slots + row)
    weight = tl.load(weights + slot).to(tl.float32)
    weighted = hidden * weight
    tl.store(output + row * D + col, weighted, col < D)


def routed_swiglu(gate_up, weights, slots, *, clamp_limit, output_dtype):
    """BF16/FP32 grouped gate/up -> original weighted down-GEMM input."""
    if (
        not gate_up.is_cuda
        or gate_up.ndim != 2
        or gate_up.shape[1] % 2
        or gate_up.stride(1) != 1
        or gate_up.dtype not in (torch.bfloat16, torch.float32)
        or output_dtype not in (torch.bfloat16, torch.float32)
        or slots.shape != (gate_up.shape[0],)
        or slots.device != gate_up.device
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
        or weights.device != gate_up.device
    ):
        raise ValueError("unsupported SM90 routed SwiGLU layout")
    rows, double_dim = gate_up.shape
    dim = double_dim // 2
    output = torch.empty((rows, dim), device=gate_up.device, dtype=output_dtype)
    if rows and dim:
        block = min(1024, triton.next_power_of_2(dim))
        _routed_swiglu_kernel[(rows, triton.cdiv(dim, block))](
            gate_up,
            weights.reshape(-1),
            slots,
            output,
            dim,
            gate_up.stride(0),
            float(clamp_limit),
            block,
            num_warps=4,
            num_stages=2,
        )
    return output


@triton.jit(do_not_specialize=["ROUTES"])
def _inverse_route_kernel(slots, inverse, ROUTES, BLOCK: tl.constexpr):
    route = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + route, route < ROUTES, other=0)
    # group_local_routes produces unique original slot IDs, even when one
    # token contains the same expert twice. Therefore no atomic is needed.
    tl.store(inverse + slot, route, route < ROUTES)


@triton.jit
def _ordered_route_sum_kernel(
    down,
    inverse,
    output,
    D: tl.constexpr,
    DOWN_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    valid=None,
    MASK_INVALID: tl.constexpr = False,
):
    token = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    # ATen CUDA sum on contiguous [M, TOPK, D], D>1, TOPK<=8:
    # thread_reduce_impl uses vt0=4 and does not split a short reduction
    # across warps. Preserve its four streams and left-associative combine,
    # rather than tl.sum's different tree or any unordered floating atomics.
    a0 = tl.full((BLOCK,), 0, tl.float32)
    a1 = tl.full((BLOCK,), 0, tl.float32)
    a2 = tl.full((BLOCK,), 0, tl.float32)
    a3 = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(TOPK):
        route = tl.load(inverse + token * TOPK + slot).to(tl.int64)
        keep = route >= 0
        if MASK_INVALID:
            keep = keep & tl.load(valid + token * TOPK + slot)
        value = tl.load(
            down + route * DOWN_STRIDE + col,
            keep & (col < D),
            other=0,
        ).to(tl.float32)
        if slot % 4 == 0:
            a0 = a0 + value
        elif slot % 4 == 1:
            a1 = a1 + value
        elif slot % 4 == 2:
            a2 = a2 + value
        else:
            a3 = a3 + value
    result = ((a0 + a1) + a2) + a3
    tl.store(output + token * D + col, result, col < D)


def reduce_routes(down, slots, rows, topk):
    """Restore original route slots and exactly sum without [M,K,D] scratch.

    ``slots`` must be the unique, valid grouped route permutation from
    group_local_routes. Missing/nonlocal/padded slots contribute FP32 zero.
    Scratch is 4*M*TOPK bytes, independent of hidden dimension.
    """
    if (
        not down.is_cuda
        or down.ndim != 2
        or down.shape[1] <= 1
        or down.stride(1) != 1
        or down.dtype not in (torch.bfloat16, torch.float32)
        or not 1 <= topk <= 8
        or rows < 0
        or slots.shape != (down.shape[0],)
        or slots.device != down.device
        or slots.dtype not in (torch.int32, torch.int64)
        or not slots.is_contiguous()
    ):
        raise ValueError("unsupported SM90 ordered route reduction layout")
    dim = down.shape[1]
    inverse = torch.full((rows, topk), -1, device=down.device, dtype=torch.int32)
    output = torch.empty((rows, dim), device=down.device, dtype=torch.float32)
    if slots.numel():
        _inverse_route_kernel[(triton.cdiv(slots.numel(), 256),)](
            slots, inverse, slots.numel(), 256
        )
    if rows:
        _ordered_route_sum_kernel[(rows, triton.cdiv(dim, 256))](
            down,
            inverse,
            output,
            dim,
            down.stride(0),
            topk,
            256,
            enable_fp_fusion=False,
        )
    return output


def reduce_static_routes(
    down, inverse, valid, rows, topk, *, output_dtype=torch.float32
):
    """Same ordered FP32 sum; mask dummy NaNs and optionally store BF16 once.

    The BF16 store is precisely normal DeepEP's original FP32-to-BF16 combine
    boundary. It is not an earlier cast of individual expert contributions.
    ``inverse``/``valid`` must come from static_route_metadata for this call.
    """
    if (
        not down.is_cuda
        or down.ndim != 2
        or down.shape[1] <= 1
        or down.stride(1) != 1
        or down.dtype not in (torch.bfloat16, torch.float32)
        or not 1 <= topk <= 8
        or rows < 0
        or down.shape[0] != rows * topk
        or inverse.shape != (rows * topk,)
        or inverse.dtype != torch.int32
        or valid.shape != inverse.shape
        or valid.dtype != torch.bool
        or inverse.device != down.device
        or valid.device != down.device
        or not inverse.is_contiguous()
        or not valid.is_contiguous()
        or output_dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("unsupported SM90 static route reduction layout")
    output = torch.empty((rows, down.shape[1]), device=down.device, dtype=output_dtype)
    if rows:
        _ordered_route_sum_kernel[(rows, triton.cdiv(down.shape[1], 256))](
            down,
            inverse,
            output,
            down.shape[1],
            down.stride(0),
            topk,
            256,
            valid,
            True,
            enable_fp_fusion=False,
        )
    return output
