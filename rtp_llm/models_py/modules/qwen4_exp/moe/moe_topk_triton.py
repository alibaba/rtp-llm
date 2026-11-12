"""Fused FP32 softmax and ten-expert routing for the Qwen4 MoE geometry."""

import os

import torch
import triton
import triton.language as tl
from torch import nn
from triton.language.extra.cuda import libdevice


@triton.jit
def _native_softmax_512(values):
    # Reproduce moeSoftmax<256>: add the two values per thread, reduce
    # each warp with offsets 1, 2, 4, 8, 16, then sum the eight warp
    # aggregates in warp order. The intermediate FP32 rounding affects
    # expert weights after the TopK renormalization.
    exp_values = libdevice.exp(values - tl.max(values, 0))
    low, high = tl.split(tl.reshape(exp_values, (2, 256)).trans(1, 0))
    partial = tl.reshape(low + high, (8, 32))
    for step in tl.static_range(5):
        left, right = tl.split(tl.reshape(partial, (8, 16 >> step, 2)))
        partial = left + right
    warp_sums = tl.reshape(partial, (8,))
    total = tl.full((), 0, tl.float32)
    for warp in tl.static_range(8):
        total += tl.sum(tl.gather(warp_sums, tl.full((1,), warp, tl.int32), axis=0), 0)
    return exp_values * tl.div_rn(1.0, total)


@triton.jit
def _moe_topk(logits, ids, weights, STRIDE: tl.constexpr, NATIVE_MATH: tl.constexpr = False):
    row = tl.program_id(0)
    experts = tl.arange(0, 512)
    values = tl.load(logits + row * STRIDE + experts).to(tl.float32)
    if NATIVE_MATH:
        probabilities = _native_softmax_512(values)
    else:
        probabilities = tl.exp(values - tl.max(values, 0))
        probabilities = probabilities / tl.sum(probabilities, 0)
    probabilities = tl.where(probabilities < 1.1754943508222875e-38, 0.0, probabilities)
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
    scale = tl.div_rn(1.0, norm) if NATIVE_MATH else 1.0 / norm
    tl.store(weights + row * 10 + slots, chosen * scale, slots < 10)


def is_supported(logits, ids, weights):
    if not (
        isinstance(logits, torch.Tensor)
        and logits.is_cuda
        and torch.version.hip is None
        and logits.ndim == 2
        and logits.shape[1] == 512
        and 0 < logits.shape[0] <= 32768
        and logits.dtype in (torch.bfloat16, torch.float32)
        and logits.stride(-1) == 1
        and logits.stride(0) > 0
        and ids.shape == weights.shape == (logits.shape[0], 10)
        and (
            ids.dtype == torch.int32
            or (
                ids.dtype == torch.int64
                and os.environ.get("QWEN_MOE_TOPK_INT64_FUSION", "0") == "1"
            )
        )
        and weights.dtype == torch.float32
        and ids.device == weights.device == logits.device
        and ids.is_contiguous()
        and weights.is_contiguous()
    ):
        return False
    capability = torch.cuda.get_device_capability(logits.device)
    if logits.shape[0] <= 8192:
        return capability[0] >= 8
    return capability == (9, 0) and "H20" in torch.cuda.get_device_name(logits.device)


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
        rows = router_logits.shape[0]
        native_math = rows > 8192
        _moe_topk[(rows,)](
            router_logits,
            topk_ids,
            topk_weights,
            STRIDE=router_logits.stride(0),
            NATIVE_MATH=native_math,
            num_warps=1 if native_math else 4,
            enable_fp_fusion=False,
        )
