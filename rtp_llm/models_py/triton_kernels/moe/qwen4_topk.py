"""Fused FP32 softmax and ten-expert routing for the Qwen4 MoE geometry."""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_topk(logits, ids, weights, STRIDE: tl.constexpr):
    row = tl.program_id(0)
    experts = tl.arange(0, 512)
    values = tl.load(logits + row * STRIDE + experts).to(tl.float32)
    probabilities = tl.exp(values - tl.max(values, 0))
    probabilities = probabilities / tl.sum(probabilities, 0)
    invalid = tl.sum((probabilities != probabilities).to(tl.int32), 0) > 0
    slots = tl.arange(0, 16)
    chosen = tl.full((16,), 0, tl.float32)
    norm = tl.full((), 0, tl.float32)
    for i in tl.static_range(10):
        # Softmax probabilities are nonnegative; their signed integer bits
        # preserve ordering above the -1 sentinel. Both warp reductions can
        # use hardware instructions, without changing softmax rounding ties.
        # Invalid softmax rows contain only NaNs and retain the override below.
        best = tl.max(probabilities.to(tl.int32, bitcast=True), 0).to(
            tl.float32, bitcast=True
        )
        expert = tl.min(tl.where(probabilities == best, experts, 2147483647), 0)
        chosen = tl.where(slots == i, best, chosen)
        norm = norm + best
        # The native 512-expert router scans two 256-element chunks.
        # Its NaN ArgMax picks 256 first, then 0; retain NaN weights so an
        # invalid activation cannot become an apparently valid route.
        native_invalid_id = 256 if i == 0 else 0
        tl.store(ids + row * 10 + i, tl.where(invalid, native_invalid_id, expert))
        probabilities = tl.where(experts == expert, -1.0, probabilities)
    tl.store(weights + row * 10 + slots, chosen * (1.0 / norm), slots < 10)


def is_supported(logits, ids, weights):
    return (
        isinstance(logits, torch.Tensor)
        and logits.is_cuda
        and torch.version.hip is None
        and logits.ndim == 2
        and logits.shape[1] == 512
        and 0 < logits.shape[0] <= 49152
        and logits.dtype in (torch.bfloat16, torch.float32)
        and logits.stride(-1) == 1
        and logits.stride(0) > 0
        and ids.shape == weights.shape == (logits.shape[0], 10)
        and ids.dtype == torch.int32
        and weights.dtype == torch.float32
        and ids.device == weights.device == logits.device
        and ids.is_contiguous()
        and weights.is_contiguous()
        and torch.cuda.get_device_capability(logits.device)[0] >= 8
    )
