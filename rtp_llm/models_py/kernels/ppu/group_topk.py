"""Exact small-batch GLM one-group routing on PPU."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _group_topk_kernel(LOGITS, BIAS, IDS, WEIGHTS):
    row = tl.program_id(0)
    index = tl.arange(0, 256)
    logits = tl.load(LOGITS + row * 256 + index)
    # Match Torch's FP32 sigmoid, including correctly rounded division. The
    # approximate tl.sigmoid/reciprocal can change both routes and their weights.
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    corrected = score + tl.load(BIAS + index)
    finite = (corrected == corrected) & (tl.abs(corrected) != float("inf"))
    maximum = tl.max(tl.where(finite, corrected, -float("inf")), 0)
    count = tl.sum((finite & (corrected == maximum)).to(tl.int32), 0)
    second = tl.where(
        count > 1,
        maximum,
        tl.max(tl.where(finite & (corrected < maximum), corrected, -float("inf")), 0),
    )
    group_score = maximum + second
    # Native GroupTopK rejects nonfinite groups and uses lowest<float> as its
    # invalid-group sentinel, even when that sentinel is a finite group sum.
    lowest = -3.4028234663852886e38
    valid_group = (
        (group_score == group_score)
        & (tl.abs(group_score) != float("inf"))
        & (group_score != lowest)
    )
    choice = tl.where(finite, corrected, lowest)
    bits = choice.to(tl.int32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    # Native WarpSelect is stable: smaller expert IDs win equal FP32 scores.
    keys = tl.sort(ordered * 512 + (255 - index), descending=True)
    lane = tl.arange(0, 8)
    selected = (255 - (tl.gather(keys, lane, 0) & 511)).to(tl.int32)
    value = tl.gather(score, selected, 0)
    valid_value = (value == value) & (tl.abs(value) != float("inf"))
    valid = valid_group & (tl.sum(valid_value.to(tl.int32), 0) == 8)
    # This reduction matches the native 32-lane sum of eight selected scores
    # followed by zero lanes. Keep division and scaling as separate operations.
    total = tl.sum(value, 0) + 1e-20
    weight = tl.div_rn(value, total) * 2.5
    tl.store(IDS + row * 8 + lane, tl.where(valid, selected, lane))
    tl.store(WEIGHTS + row * 8 + lane, tl.where(valid, weight, 0.125))


def try_fused_group_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scores: torch.Tensor,
    correction_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> bool:
    """Run only the qualified layout/configuration; otherwise keep native routing."""
    if (
        n_group != 1
        or topk_group != 1
        or topk != 8
        or not renormalize
        or routed_scaling_factor != 2.5
        or scores.ndim != 2
        or scores.shape[1] != 256
        or scores.shape[0] > 1024
        or scores.dtype != torch.float32
        or not scores.is_cuda
        or not scores.is_contiguous()
        or correction_bias.shape != (256,)
        or correction_bias.dtype != torch.float32
        or not correction_bias.is_contiguous()
        or topk_weights.shape != (scores.shape[0], 8)
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or topk_ids.shape != topk_weights.shape
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_ids.is_contiguous()
        or correction_bias.device != scores.device
        or topk_weights.device != scores.device
        or topk_ids.device != scores.device
    ):
        return False
    if scores.shape[0]:
        _group_topk_kernel[(scores.shape[0],)](
            scores,
            correction_bias,
            topk_ids,
            topk_weights,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return True
