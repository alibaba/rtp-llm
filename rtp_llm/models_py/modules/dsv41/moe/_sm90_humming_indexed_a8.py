"""Bounded native A8/K32 indexed experts, with device-only route metadata.

This local executor does not own dispatch/combine or EP participation. It reuses
the existing native-A8 packed weights and g128 quantizer. The indexed GEMMs
write original top-k slots; routing weights are rounded into BF16 hidden BEFORE
down-input quantization. No A16 decode, custom tuning or precision fallback.
Warm up each shape on the execution device before CUDA Graph capture.
"""

import json

import torch


_COMPUTE = {
    "gemm_type": "indexed",
    "use_f16_accum": False,
    "use_batch_invariant": True,
}


def _settings(linear, slots):
    """Only static heuristic metadata is cached; never cache live route values."""
    from humming.tune import get_heuristics_config

    key = (linear.config.to_str(), linear.weight.device.index, slots)
    cache = getattr(linear, "_sm90_indexed_a8_settings", None)
    if cache is None:
        cache = {}
        linear._sm90_indexed_a8_settings = cache
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up native A8 indexed shape before capture")
        tuning = get_heuristics_config(
            linear.config, shape_m=slots, device=linear.weight.device, **_COMPUTE,
        )
        if tuning["use_stream_k"] or tuning["block_shape"][2] != tuning["warp_shape"][2]:
            raise RuntimeError("Native A8 indexed requires invariant no-Stream-K traversal")
        cache[key] = (tuning["block_shape"][0], json.dumps(tuning))
    return cache[key]


def _identity_slots(local, x, topk):
    key = (x.device.index, x.shape[0], topk)
    cache = getattr(local, "_sm90_indexed_a8_identity", None)
    if cache is None:
        cache = {}
        local._sm90_indexed_a8_identity = cache
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up native A8 indexed identity before capture")
        cache[key] = torch.arange(x.shape[0] * topk, device=x.device, dtype=torch.int64)
    return cache[key]


def _gemm(linear, q, scales, output, metadata, topk, tuning):
    from humming.forward import humming_forward

    slots, experts, padded_count = metadata
    return humming_forward(
        linear.config, inputs=q, input_scale=scales, outputs=output,
        weight=linear.weight, weight_scale=linear.weight_scale,
        weight_scale_2=linear.weight_scale_2, locks=linear.locks,
        sorted_ids=slots, expert_ids=experts, num_tokens_padded=padded_count,
        top_k=topk, valid_shape_m=q.shape[0],
        compute_config=_COMPUTE, tuning_config=tuning,
    )


def forward_local_indexed_a8(local, x, weights, indices, local_start, local_end):
    """FP32 local routed sum; supports local IDs with start=0 or global IDs.

    Scope: BF16 contiguous SM90 [R,K], FP32 [R,topk] weights, contiguous int32/
    int64 IDs, 1<=topk<=8, E<=64, R*topk<1024. Invalid IDs are omitted, including
    arbitrary int64 sentinels. Zero local rows are legal here, but callers MUST
    still participate in EP dispatch/combine. Unsupported inputs raise instead
    of silently entering an A16 or non-graph-safe fallback.
    """
    from rtp_llm.models_py.kernels.cuda.sm90_moe_align import align_small_batch
    from ._sm90_route_kernels import reduce_routes, routed_swiglu

    if not getattr(local, "native_a8", False):
        raise ValueError("Indexed graph experts require native A8")
    first, second = local.w13, local.w2
    if any(linear.promotion_group_size != 32 for linear in (first, second)):
        raise ValueError("Indexed graph experts require unchanged K32 promotion")
    if (
        not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2
        or not x.is_contiguous() or indices.ndim != 2
        or weights.shape != indices.shape or indices.shape[0] != x.shape[0]
        or weights.dtype != torch.float32
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_contiguous() or not weights.is_contiguous()
        or indices.device != x.device or weights.device != x.device
        or not 1 <= indices.shape[1] <= 8 or indices.numel() >= 1024
        or type(local_start) is not int or type(local_end) is not int
        or not 0 <= local_start <= 2**31 - 1 - (local_end - local_start)
        or not 1 <= local_end - local_start <= 64
    ):
        raise ValueError("Unsupported bounded native A8 indexed route layout")
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError("Native A8 indexed experts require SM90")
    if (
        any(linear.weight.device != x.device for linear in (first, second))
        or any(linear.config.num_experts != local_end - local_start for linear in (first, second))
        or first.in_features != x.shape[1]
        or first.out_features != 2 * second.in_features
        or second.out_features != x.shape[1]
        or first.in_features % 128 or second.in_features % 128
    ):
        raise ValueError("Native A8 indexed expert projections do not match routes/input")
    if x.shape[0] == 0:
        return torch.zeros_like(x, dtype=torch.float32)

    topk, slots = indices.shape[1], indices.numel()
    first_bm, first_tuning = _settings(first, slots)
    second_bm, second_tuning = _settings(second, slots)
    identity = _identity_slots(local, x, topk)
    first_meta = align_small_batch(
        indices, local_end - local_start, first_bm, local_start=local_start,
    )
    second_meta = first_meta if first_bm == second_bm else align_small_batch(
        indices, local_end - local_start, second_bm, local_start=local_start,
    )
    if first_meta is None or second_meta is None:
        raise RuntimeError("Native alignment rejected bounded A8 indexed input")

    valid = (indices >= local_start) & (indices < local_end)
    # DeepEP receive padding may contain NaNs. Selection is required: NaN*0
    # would poison g128 amax and the weighted hidden intermediate.
    clean_x = torch.where(valid.any(dim=1)[:, None], x, 0).contiguous()
    clean_weights = torch.where(valid, weights, 0).contiguous()
    gate_up = torch.zeros((slots, first.out_features), device=x.device, dtype=x.dtype)
    _gemm(first, *first.quantize_input(clean_x), gate_up, first_meta, topk, first_tuning)
    hidden = routed_swiglu(
        gate_up, clean_weights, identity,
        clamp_limit=local.cfg.swiglu_limit, output_dtype=torch.bfloat16,
    )
    down = torch.zeros((slots, second.out_features), device=x.device, dtype=x.dtype)
    _gemm(second, *second.quantize_input(hidden), down, second_meta, 1, second_tuning)
    # Both indexed outputs are pre-zeroed; invalid routes stay exactly zero.
    # Reuse the original four-lane FP32 summation tree, without float atomics.
    return reduce_routes(down, identity, x.shape[0], topk)
