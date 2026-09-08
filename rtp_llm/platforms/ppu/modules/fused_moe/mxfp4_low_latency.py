"""PPU MXFP4 compute and buffer ownership for an engine-owned DeepEP group."""

import torch

from rtp_llm.platforms.ppu.kernels.ppu_mxfp4_masked import (
    mxfp4_experts_masked,
    tensor_spans_overlap,
)
from rtp_llm.platforms.ppu.kernels.ppu_topk_padding import pad_inactive_routes


def pad_topk(indices, weights):
    """Pad unsupported widths (including V4's six) with inactive routes."""
    width = indices.shape[1]
    supported = (2, 4, 8, 16)
    if width in supported:
        return indices, weights
    target = next((n for n in supported if n > width), None)
    if target is None or width <= 0:
        raise ValueError("DeepEP topk must be in [1,16]")
    return pad_inactive_routes(indices, weights, target)


def low_latency_mxfp4_moe(
    buffer,
    x,
    weights,
    indices,
    weight13,
    weight2,
    *,
    num_experts,
    max_dispatch_tokens,
    expected_m,
    swiglu_limit=None,
    output_dtype=torch.float32,
):
    """Dispatch and combine every valid row, with no compact-prefix capacity.

    This adapter borrows the engine's existing buffer/communicator. The handle
    and payload stay alive through combine. Padded/inactive routes remain -1.
    BF16 output preserves the combine result for consumers that promote in
    registers; the FP32 default retains the existing adapter contract.
    """
    if output_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("PPU routed output must be FP32 or BF16")
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or indices.ndim != 2
        or weights.shape != indices.shape
        or x.shape[0] != indices.shape[0]
        or weights.dtype != torch.float32
        or indices.dtype != torch.int64
        or weights.device != x.device
        or indices.device != x.device
        or x.shape[0] > max_dispatch_tokens
    ):
        raise ValueError("Invalid DeepEP LL BF16 tokens / FP32 weights / int64 routes")
    indices, weights = pad_topk(indices, weights)
    indices, weights = indices.contiguous(), weights.contiguous()
    expert_x, counts, handle, _, _ = buffer.low_latency_dispatch(
        x=x.contiguous(),
        topk_idx=indices,
        num_max_dispatch_tokens_per_rank=max_dispatch_tokens,
        num_experts=num_experts,
        use_fp8=False,
        use_mxfp4=True,
        mxfp4_scale_row_major=False,
        quant_size=32,
        async_finish=False,
        return_recv_hook=False,
    )
    if not isinstance(expert_x, tuple) or len(expert_x) != 2:
        raise RuntimeError("DeepEP LL MXFP4 dispatch must return (data, scale)")
    combine_slot = buffer.get_next_low_latency_combine_buffer(handle)
    # DeepEP versions may share arenas between dispatch and combine. Only
    # write w2 directly when the candidate cannot alias any live input.
    live_inputs = (*expert_x, *weight13, *weight2, counts)
    direct = not any(tensor_spans_overlap(combine_slot, t) for t in live_inputs)
    result = mxfp4_experts_masked(
        expert_x,
        weight13,
        weight2,
        counts,
        expected_m=expected_m,
        swiglu_limit=swiglu_limit,
        out=combine_slot if direct else None,
    )
    if not direct:
        if result.shape != combine_slot.shape:
            raise ValueError("DeepEP combine slot differs from the full expert output")
        combine_slot.copy_(result)
    combined, _, _ = buffer.low_latency_combine(
        x=combine_slot,
        topk_idx=indices,
        topk_weights=weights,
        handle=handle,
        zero_copy=True,
        async_finish=False,
        return_recv_hook=False,
    )
    if combined.dtype != torch.bfloat16:
        raise RuntimeError("PPU DeepEP LL combine must return BF16")
    return combined.to(output_dtype)
