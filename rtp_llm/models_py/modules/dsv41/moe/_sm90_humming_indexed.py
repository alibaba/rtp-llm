"""Opt-in, bounded SM90 indexed experts over existing Humming weight buffers.

No dependency upgrade, repacking, quantizer substitution, or route-weight move.
Unsupported requests return None to the caller's existing grouped fallback.
Kernel failures deliberately propagate rather than silently changing arithmetic.
"""

import json
import os

import torch

from ._sm90_humming_linear import decode_quantized_activation
from ._sm90_route_kernels import reduce_routes, routed_swiglu


_COMPUTE = {
    "gemm_type": "indexed",
    "use_f16_accum": False,
    "use_batch_invariant": True,
}


def _settings(linear, routed_slots):
    """Cache only static metadata, never a second packed weight or route layout."""
    from humming.tune import get_heuristics_config

    key = (linear.config.to_str(), linear.weight.device.index, routed_slots)
    cache = getattr(linear, "_sm90_indexed_settings", None)
    if cache is None:
        cache = {}
        linear._sm90_indexed_settings = cache
    if key not in cache:
        config = get_heuristics_config(
            linear.config, shape_m=routed_slots, device=linear.weight.device,
            **_COMPUTE,
        )
        if config["use_stream_k"] or config["block_shape"][2] != config["warp_shape"][2]:
            raise RuntimeError("Indexed invariant candidate requires unchanged no-Stream-K K traversal")
        cache[key] = (config["block_shape"][0], json.dumps(config))
    return cache[key]


def _decode(x):
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    payload, scales = sgl_per_token_group_quant_fp8(
        x, group_size=128, eps=1e-4, column_major_scales=True,
        scale_tma_aligned=True, scale_ue8m0=True,
    )
    return decode_quantized_activation(payload, scales)


def _gemm(linear, inputs, output, metadata, topk, tuning):
    from humming.forward import humming_forward

    sorted_slots, block_experts, padded_count = metadata
    return humming_forward(
        linear.config, inputs=inputs, outputs=output,
        weight=linear.weight, weight_scale=linear.weight_scale,
        weight_scale_2=getattr(linear, "weight_scale_2", None),
        sorted_ids=sorted_slots, expert_ids=block_experts,
        num_tokens_padded=padded_count, top_k=topk,
        valid_shape_m=inputs.shape[0], locks=linear.locks,
        compute_config=_COMPUTE, tuning_config=tuning,
    )


def try_forward_indexed(local, x, weights, globalids, start, end):
    """Return FP32 local routed sum, or None for the existing grouped fallback.

    Bounded to BF16 SM90, E<=64, top-k1..8 and fewer than1024 route slots.
    Graph/capture, other dtypes and arbitrary strides remain unsupported here.
    IDs and routing weights retain their original request/top-k slot order.
    """
    if os.environ.get("DSV41_SM90_INDEXED_MOE", "0") != "1":
        return None
    if (
        not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2
        or not x.is_contiguous() or globalids.ndim != 2 or weights.ndim != 2
        or globalids.dtype not in (torch.int32, torch.int64)
        or weights.dtype != torch.float32 or not globalids.is_contiguous()
        or not weights.is_contiguous() or globalids.shape != weights.shape
        or globalids.shape[0] != x.shape[0] or not 1 <= globalids.shape[1] <= 8
        or globalids.numel() >= 1024 or globalids.device != x.device
        or weights.device != x.device or not isinstance(start, int) or not isinstance(end, int)
        or not 1 <= end - start <= 64 or start < 0
    ):
        return None
    if torch.cuda.get_device_capability(x.device)[0] != 9 or torch.cuda.is_current_stream_capturing():
        return None
    first, second = local.w13, local.w2
    if (
        first.weight.device != x.device or second.weight.device != x.device
        or first.config.num_experts != end - start or second.config.num_experts != end - start
        or first.in_features != x.shape[1] or first.out_features != 2 * second.in_features
        or second.out_features != x.shape[1] or x.shape[1] % 128 or second.in_features % 128
    ):
        return None
    if x.shape[0] == 0:
        return torch.zeros(x.shape, dtype=torch.float32, device=x.device)

    from rtp_llm.models_py.kernels.cuda.sm90_moe_align import align_small_batch

    routed_slots = globalids.numel()
    first_bm, first_tuning = _settings(first, routed_slots)
    second_bm, second_tuning = _settings(second, routed_slots)
    first_meta = align_small_batch(globalids, end - start, first_bm, local_start=start)
    if first_meta is None:
        return None
    second_meta = first_meta if second_bm == first_bm else align_small_batch(
        globalids, end - start, second_bm, local_start=start,
    )
    if second_meta is None:
        return None

    # Indexed writes only valid routes. Initialize BOTH outputs: zeroing after
    # multiplication is not safe because uninitialized/NaN values times0 areNaN.
    gate_up = torch.zeros((routed_slots, first.out_features), device=x.device, dtype=x.dtype)
    _gemm(first, _decode(x), gate_up, first_meta, globalids.shape[1], first_tuning)
    identity_slots = torch.arange(routed_slots, device=x.device, dtype=torch.int64)
    hidden = routed_swiglu(
        gate_up, weights, identity_slots,
        clamp_limit=local.cfg.swiglu_limit, output_dtype=x.dtype,
    )
    down = torch.zeros((routed_slots, second.out_features), device=x.device, dtype=x.dtype)
    _gemm(second, _decode(hidden), down, second_meta, 1, second_tuning)
    # Invalid slots remain initialized zeros; valid outputs are in original slot
    # order regardless of native align's within-expert scheduling order.
    return reduce_routes(down, identity_slots, x.shape[0], globalids.shape[1])
