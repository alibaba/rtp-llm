"""Fused FP32 softmax and ten-expert routing for the Qwen4 MoE geometry."""

import torch
import triton
import triton.language as tl
from torch import nn


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
        best = tl.max(probabilities, 0)
        # CUB ArgMax in the native router chooses the lower ID on a tie.
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
        and 0 < logits.shape[0] <= 8192
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


class Qwen4ExpMoeTopk(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        config = native.config
        self.geometry_supported = (
            config.expert_num == 512
            and config.moe_k == 10
            and bool(config.has_moe_norm)
        )

    def forward(self, router_logits, topk_ids, topk_weights):
        if not self.geometry_supported or not is_supported(
            router_logits, topk_ids, topk_weights
        ):
            return self.native(router_logits, topk_ids, topk_weights)
        _moe_topk[(router_logits.shape[0],)](
            router_logits,
            topk_ids,
            topk_weights,
            STRIDE=router_logits.stride(0),
            num_warps=4,
            enable_fp_fusion=False,
        )
