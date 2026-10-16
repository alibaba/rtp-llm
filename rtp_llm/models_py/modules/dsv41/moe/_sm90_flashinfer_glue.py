"""Exact FlashInfer input/output glue for the explicit SM90 candidate backend.

No routing, quantizer, or GEMM is changed. Physical top-k slots retain their
order. Every call owns fresh outputs and recomputes validity on the device,
including CUDA graph replay. Invalid memory is masked before loading rather
than multiplied by zero. This does not change the backend's numerical contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_kernel(
    x,
    weights,
    indices,
    safe_x,
    safe_weights,
    safe_indices,
    live_rows,
    K: tl.constexpr,
    TOPK: tl.constexpr,
    EXPERTS: tl.constexpr,
    ROUTE_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.arange(0, ROUTE_BLOCK)
    ids = tl.load(indices + row * TOPK + slot, slot < TOPK, other=-1)
    # Compare in the original INT64 domain: narrowing first could wrap a large
    # invalid ID into a real expert. Zero route weights do not make IDs invalid.
    valid = (slot < TOPK) & (ids >= 0) & (ids < EXPERTS)
    live = tl.sum(valid.to(tl.int32), axis=0) > 0
    if tile == 0:
        weight_bits = tl.load(
            weights.to(tl.pointer_type(tl.int32)) + row * TOPK + slot,
            valid,
            other=0,
        )
        tl.store(
            safe_weights.to(tl.pointer_type(tl.int32)) + row * TOPK + slot,
            weight_bits,
            slot < TOPK,
        )
        cleaned_ids = tl.where(valid, ids, EXPERTS).to(tl.int32)
        tl.store(safe_indices + row * TOPK + slot, cleaned_ids, slot < TOPK)
        tl.store(live_rows + row, live)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    x_bits = tl.load(
        x.to(tl.pointer_type(tl.int16)) + row * K + col,
        live & (col < K),
        other=0,
    )
    tl.store(
        safe_x.to(tl.pointer_type(tl.int16)) + row * K + col,
        x_bits,
        col < K,
    )


@triton.jit
def _finish_kernel(
    output,
    live_rows,
    result,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    WIDEN: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = tl.load(live_rows + row)
    bits = tl.load(
        output.to(tl.pointer_type(tl.uint16)) + row * K + col,
        live & (col < K),
        other=0,
    )
    if WIDEN:
        # BF16 -> FP32 is an exact bit expansion. Preserve NaN payload/sign and
        # signed zero without invoking an arithmetic NaN canonicalization.
        wide_bits = bits.to(tl.uint32) << 16
        tl.store(
            result.to(tl.pointer_type(tl.uint32)) + row * K + col,
            wide_bits,
            col < K,
        )
    else:
        tl.store(
            result.to(tl.pointer_type(tl.uint16)) + row * K + col,
            bits,
            col < K,
        )


def _check_cuda_contiguous(tensor, dtype, device):
    if tensor.dtype != dtype or tensor.device != device or not tensor.is_cuda:
        raise TypeError("expected matching CUDA device and declared input dtype")
    if not tensor.is_contiguous():
        raise ValueError("FlashInfer glue requires contiguous tensors")


def prepare_flashinfer_inputs(x, weights, indices, num_experts):
    """Return fresh (BF16 x, FP32 weights, INT32 IDs, BOOL [M,1] live).

    Inputs are contiguous CUDA BF16 [M,K], FP32 [M,T], INT64 [M,T].
    T is 1..8, K is positive, M may be zero. Invalid IDs become sentinel E;
    their weights become +0. All-invalid input rows become +0. Valid payloads
    are copied bitwise, including NaNs and signed zero. No host route inspection.
    """
    if (
        not isinstance(num_experts, int)
        or isinstance(num_experts, bool)
        or not 1 <= num_experts <= 0x7FFFFFFF
    ):
        raise ValueError("num_experts must be a positive INT32-representable int")
    if x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError("expected x with shape [M,K], K > 0")
    if (
        indices.ndim != 2
        or weights.shape != indices.shape
        or indices.shape[0] != x.shape[0]
        or not 1 <= indices.shape[1] <= 8
    ):
        raise ValueError("expected matching routes [M,T], T in [1,8]")
    for tensor, dtype in (
        (x, torch.bfloat16),
        (weights, torch.float32),
        (indices, torch.int64),
    ):
        _check_cuda_contiguous(tensor, dtype, x.device)
    rows, k = x.shape
    topk = indices.shape[1]
    safe_x = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    safe_weights = torch.empty(weights.shape, device=x.device, dtype=torch.float32)
    safe_indices = torch.empty(indices.shape, device=x.device, dtype=torch.int32)
    live_rows = torch.empty((rows, 1), device=x.device, dtype=torch.bool)
    if rows:
        _prepare_kernel[(rows, triton.cdiv(k, 512))](
            x,
            weights,
            indices,
            safe_x,
            safe_weights,
            safe_indices,
            live_rows,
            k,
            topk,
            num_experts,
            triton.next_power_of_2(topk),
            512,
            num_warps=4,
        )
    return safe_x, safe_weights, safe_indices, live_rows


def finish_flashinfer_output(output, live_rows, output_dtype=torch.float32):
    """Fresh BF16/FP32 output; all-invalid rows are +0 without reading them."""
    if output.ndim != 2 or output.shape[1] <= 0:
        raise ValueError("expected output with shape [M,K], K > 0")
    if live_rows.shape != (output.shape[0], 1):
        raise ValueError("live_rows must have shape [M,1]")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output dtype must be BF16 or FP32")
    _check_cuda_contiguous(output, torch.bfloat16, output.device)
    _check_cuda_contiguous(live_rows, torch.bool, output.device)
    result = torch.empty(output.shape, device=output.device, dtype=output_dtype)
    rows, k = output.shape
    if rows:
        _finish_kernel[(rows, triton.cdiv(k, 512))](
            output,
            live_rows,
            result,
            k,
            512,
            output_dtype == torch.float32,
            num_warps=4,
        )
    return result
