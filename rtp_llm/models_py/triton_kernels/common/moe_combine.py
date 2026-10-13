"""MoE deferred-combine kernels (TP>1 / EP paths).

The EP-shared-allreduce branch of GenericMoeLayer and the pure-TP router
finalize otherwise run the deferred top-k reduction and the shared-expert
gate broadcast as *unfused* aten ops (the TP1 / ``skip_tp_allreduce`` path
already folds them into moe_gating.SigmoidGateScaleAdd):

  * ``aten::sum(dim=1)`` over the unsummed routed output ``[M, topk, K]``
    (pure-TP router finalize, before the TP all-reduce)
        -> :func:`topk_sum_triton`
  * ``torch.sigmoid(gate) * shared`` broadcast multiply
    (``[T, 1]`` gate x ``[T, H]`` shared, pre-all-reduce)
        -> :func:`sigmoid_mul_triton`

Both kernels follow bit-exact protocols verified with ``torch.equal``
against the aten chains they replace: the top-k sum replicates aten's
reduce_kernel ILP-4 accumulation order (four fp32 streams over k mod 4,
combined left-associatively; verified for topk <= 32), and every
intermediate that aten materializes in a reduced dtype is rounded at the
same point (the sigmoid output in the gate dtype, the top-k sum in the
experts dtype).
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.moe_gating import _select_block_h

# aten's reduce_kernel switches its accumulation layout for reductions
# longer than this; torch.equal was verified for 1..32 (K=64 mismatches).
_MAX_VERIFIED_TOPK = 32

# The 4-stream protocol holds four fp32 accumulators of BLOCK_H lanes per
# program, so the block size is capped below moe_gating's single-accumulator
# heuristic to keep register pressure comparable (4 * 1024 fp32 lanes = 32
# registers per thread at the default 4 warps).
_TOPK_SUM_MAX_BLOCK_H = 1024

# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _TopkSum_kernel(
    experts_ptr,  # [M, TOPK, H] unsummed routed experts output
    output_ptr,  # [M, H] summed result (fresh tensor)
    H,
    stride_experts_m,  # experts stride along the token dim (elements)
    stride_experts_k,  # experts stride along the top-k dim (elements)
    stride_out_m,  # output stride along the token dim (elements)
    TOPK: tl.constexpr,  # top-k slots folded into the sum
    BLOCK_H: tl.constexpr,
):
    """output[m, :] = sum_k experts[m, k, :]

    The accumulation replicates aten's reduce_kernel order exactly (verified
    torch.equal against torch.sum(dim=1) for TOPK <= 32): four fp32
    accumulators split the top-k slots interleaved — stream j accumulates
    the k = j, j+4, j+8, ... slices in ascending order — and the streams
    combine left-associatively ((s0 + s1) + s2) + s3.  A single sequential
    ascending loop does NOT match (aten's ILP-4 unrolled reduction differs
    in the fp32 rounding points; ~2/108M bf16 elements at [53140, 8, 2048]).
    The fp32 result is rounded to the experts dtype exactly where aten's
    sum materializes its output.

    Grid: (M, ceil(H / BLOCK_H))
      - axis-0: token index
      - axis-1: hidden-dimension block index
    """
    pid_m = tl.program_id(axis=0)  # token index
    pid_h = tl.program_id(axis=1)  # hidden block index
    pid_m64 = pid_m.to(tl.int64)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    experts_base = experts_ptr + pid_m64 * stride_experts_m
    zeros = tl.zeros([BLOCK_H], dtype=tl.float32)

    # stream 0: k = 0, 4, 8, ...
    acc0 = tl.load(experts_base + h_offsets, mask=mask, other=0.0).to(tl.float32)
    for k in range(4, TOPK, 4):
        acc0 += tl.load(
            experts_base + k * stride_experts_k + h_offsets, mask=mask, other=0.0
        ).to(tl.float32)
    # stream 1: k = 1, 5, 9, ...
    if TOPK > 1:
        acc1 = tl.load(
            experts_base + stride_experts_k + h_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        for k in range(5, TOPK, 4):
            acc1 += tl.load(
                experts_base + k * stride_experts_k + h_offsets, mask=mask, other=0.0
            ).to(tl.float32)
    else:
        acc1 = zeros
    # stream 2: k = 2, 6, 10, ...
    if TOPK > 2:
        acc2 = tl.load(
            experts_base + 2 * stride_experts_k + h_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        for k in range(6, TOPK, 4):
            acc2 += tl.load(
                experts_base + k * stride_experts_k + h_offsets, mask=mask, other=0.0
            ).to(tl.float32)
    else:
        acc2 = zeros
    # stream 3: k = 3, 7, 11, ...
    if TOPK > 3:
        acc3 = tl.load(
            experts_base + 3 * stride_experts_k + h_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        for k in range(7, TOPK, 4):
            acc3 += tl.load(
                experts_base + k * stride_experts_k + h_offsets, mask=mask, other=0.0
            ).to(tl.float32)
    else:
        acc3 = zeros

    accumulator = ((acc0 + acc1) + acc2) + acc3
    tl.store(
        output_ptr + pid_m64 * stride_out_m + h_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _SigmoidBroadcastMul_kernel(
    gate_ptr,  # [T, 1] scalar gate output from shared_expert_gate linear
    shared_ptr,  # [T, H] shared expert MLP output
    output_ptr,  # [T, H] result (fresh tensor)
    H,
    stride_gate_t,  # gate stride along token dim (elements)
    stride_shared_t,  # shared stride along token dim (elements)
    stride_out_t,  # output stride along token dim (elements)
    BLOCK_H: tl.constexpr,
):
    """output[t, :] = sigmoid(gate[t, 0]) * shared[t, :]

    Replaces the aten broadcast chain ``torch.sigmoid(gate) * shared``
    ([T, 1] x [T, H]): the sigmoid is computed in fp32 (as aten's opmath
    does) and rounded to the gate dtype exactly where aten materializes
    ``torch.sigmoid(gate)``'s output, and the multiply runs in fp32 with a
    single rounding to the shared dtype — bit-identical to the two-op aten
    chain.  The [T, 1] gate is read directly with a strided scalar load, so
    the broadcast never materializes a [T, H] operand.

    Grid: (T, ceil(H / BLOCK_H))
      - axis-0: token index
      - axis-1: hidden-dimension block index
    """
    pid_t = tl.program_id(axis=0)  # token index
    pid_h = tl.program_id(axis=1)  # hidden block index
    pid_t64 = pid_t.to(tl.int64)

    gate_val = tl.load(gate_ptr + pid_t64 * stride_gate_t).to(tl.float32)
    gate_val = tl.sigmoid(gate_val)
    # torch.sigmoid writes its output in the gate's dtype; the aten multiply
    # then reads that rounded value back and upcasts it.  Round here at the
    # same point so the fused kernel is bit-exact vs the two-op chain.
    gate_val = gate_val.to(gate_ptr.dtype.element_ty).to(tl.float32)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    shared_vec = tl.load(
        shared_ptr + pid_t64 * stride_shared_t + h_offsets, mask=mask, other=0.0
    )
    result = gate_val * shared_vec.to(tl.float32)
    tl.store(
        output_ptr + pid_t64 * stride_out_t + h_offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Supported-layout predicates (callers fall back to the aten chain when
# these return False, e.g. non-CUDA tensors, exotic dtypes or layouts)
# ---------------------------------------------------------------------------

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def topk_sum_supported(experts: torch.Tensor) -> bool:
    """Whether :func:`topk_sum_triton` can fold this unsummed routed output.

    Restricted to the contiguous [M, topk, H] layout the executors produce
    (``out.view(M, top_k, K)`` of a contiguous [M * top_k, K] buffer) and the
    top-k range the aten-order replication was verified for; anything else
    falls back to the aten sum.
    """
    return (
        experts.dim() == 3
        and experts.is_cuda
        and experts.dtype in _SUPPORTED_DTYPES
        and experts.is_contiguous()
        and 0 < experts.size(1) <= _MAX_VERIFIED_TOPK
        and experts.size(0) > 0
        and experts.size(2) > 0
    )


def sigmoid_mul_supported(gate: torch.Tensor, shared: torch.Tensor) -> bool:
    """Whether :func:`sigmoid_mul_triton` can gate this shared-expert output."""
    return (
        gate.dim() == 2
        and gate.size(1) == 1
        and shared.dim() == 2
        and gate.is_cuda
        and shared.is_cuda
        and gate.dtype == shared.dtype
        and shared.dtype in _SUPPORTED_DTYPES
        and shared.stride(1) == 1
        and gate.stride(1) == 1
        and shared.size(0) > 0
        and shared.size(1) > 0
    )


# ---------------------------------------------------------------------------
# Low-level functional entry points
# ---------------------------------------------------------------------------


def topk_sum_triton(experts: torch.Tensor) -> torch.Tensor:
    """Fold an unsummed routed output [M, topk, H] to [M, H] in one kernel.

    Computes ``out = experts.sum(dim=1)`` bit-identically to torch.sum
    (fp32 accumulation in aten's ILP-4 stream order, a single rounding of
    the fp32 result to the experts dtype), replacing aten's reduce_kernel
    launch (measured 913us -> 534us at 3.67 TB/s for [53140, 8, 2048] bf16
    on H20) and keeping the [M, H] output contiguous for the TP all-reduce
    that follows in the router finalize path.

    Args:
        experts: [M, topk, H] routed experts output, last dim contiguous,
                 1 <= topk <= 32 (see :func:`topk_sum_supported`).

    Returns:
        A fresh contiguous [M, H] tensor (torch.sum allocated the same
        intermediate, so the allocation count is unchanged).
    """
    M, topk, H = experts.shape
    output = torch.empty((M, H), dtype=experts.dtype, device=experts.device)
    BLOCK_H = min(_select_block_h(M, H), _TOPK_SUM_MAX_BLOCK_H)
    grid = (M, triton.cdiv(H, BLOCK_H))
    _TopkSum_kernel[grid](
        experts,
        output,
        H,
        experts.stride(0),
        experts.stride(1),
        output.stride(0),
        TOPK=topk,
        BLOCK_H=BLOCK_H,
    )
    return output


def sigmoid_mul_triton(gate: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """Fused sigmoid gate broadcast multiply: sigmoid(gate) * shared.

    Computes ``out[t, :] = sigmoid(gate[t, 0]) * shared[t, :]`` in one
    kernel, bit-identical to the aten chain ``torch.sigmoid(gate) * shared``
    (fp32 opmath, the sigmoid rounded to the gate dtype where aten
    materializes it, single rounding of the product to the shared dtype).
    The strided [T, 1] gate load replaces aten's broadcast multiply over a
    strided [T, H] view (measured 442us -> 123us at 3.53 TB/s for
    [53140, 1] x [53140, 2048] bf16 on H20) and skips the separate
    sigmoid kernel launch.

    Args:
        gate:    [T, 1] scalar gate from the shared_expert_gate linear.
        shared:  [T, H] shared expert MLP output, last dim contiguous.

    Returns:
        A fresh contiguous [T, H] tensor (the aten multiply allocated the
        same [T, H] output, so the allocation count is unchanged; the
        separate [T, 1] sigmoid tensor is no longer materialized).
    """
    T, H = shared.shape
    output = torch.empty((T, H), dtype=shared.dtype, device=shared.device)
    BLOCK_H = _select_block_h(T, H)
    grid = (T, triton.cdiv(H, BLOCK_H))
    _SigmoidBroadcastMul_kernel[grid](
        gate,
        shared,
        output,
        H,
        gate.stride(0),
        shared.stride(0),
        output.stride(0),
        BLOCK_H=BLOCK_H,
    )
    return output
