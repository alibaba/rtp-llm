# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from numbers import Integral

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _count_survival_at_least(
    survival_ptr, num_slots, threshold, BLOCK_SIZE: tl.constexpr
):
    count = 0
    for start in range(0, num_slots, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        bits = tl.load(survival_ptr + offsets, mask=offsets < num_slots, other=-1)
        count += tl.sum((bits >= threshold).to(tl.int32))
    return count


@triton.jit(do_not_specialize=["num_reqs", "draft_budget"])
def _assign_draft_token_budget_kernel(
    # [max_num_reqs, num_steps]
    confidence_probs_ptr,
    confidence_probs_stride,
    # [num_reqs]
    idx_mapping_ptr,
    # [num_reqs]
    capacities_ptr,
    # [max_num_reqs * num_steps]
    survival_ptr,
    num_reqs,
    draft_budget,
    NUM_STEPS: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Survival probabilities are non-negative, so their float bits order like
    # the values; ineligible slots store -1 to rank below every real draft.
    for start in range(0, num_reqs, BLOCK_REQS):
        reqs = start + tl.arange(0, BLOCK_REQS)
        mask = reqs < num_reqs
        req_state_idx = tl.load(idx_mapping_ptr + reqs, mask=mask, other=0)
        capacity = tl.load(capacities_ptr + reqs, mask=mask, other=0)
        survival = tl.full([BLOCK_REQS], 1.0, tl.float32)
        for step in tl.static_range(NUM_STEPS):
            prob = tl.load(
                confidence_probs_ptr + req_state_idx * confidence_probs_stride + step,
                mask=mask,
                other=0.0,
            )
            survival *= prob
            bits = tl.where(step < capacity, survival.to(tl.int32, bitcast=True), -1)
            tl.store(survival_ptr + reqs * NUM_STEPS + step, bits, mask=mask)
    tl.debug_barrier()

    # Binary search the largest threshold that still admits draft_budget slots.
    num_slots = num_reqs * NUM_STEPS
    lo = 0
    hi = 0x7F800000
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2
        if (
            _count_survival_at_least(survival_ptr, num_slots, mid, BLOCK_SIZE)
            >= draft_budget
        ):
            lo = mid
        else:
            hi = mid - 1
    ties_left = draft_budget - _count_survival_at_least(
        survival_ptr, num_slots, lo + 1, BLOCK_SIZE
    )

    # Admit every slot above the threshold and hand out ties in request order.
    for start in range(0, num_reqs, BLOCK_REQS):
        reqs = start + tl.arange(0, BLOCK_REQS)
        mask = reqs < num_reqs
        num_above = tl.zeros([BLOCK_REQS], tl.int32)
        num_tied = tl.zeros([BLOCK_REQS], tl.int32)
        for step in tl.static_range(NUM_STEPS):
            bits = tl.load(survival_ptr + reqs * NUM_STEPS + step, mask=mask, other=-1)
            num_above += (bits > lo).to(tl.int32)
            num_tied += (bits == lo).to(tl.int32)
        tied_before = tl.cumsum(num_tied, axis=0) - num_tied
        num_taken = tl.minimum(num_tied, tl.maximum(ties_left - tied_before, 0))
        tl.store(capacities_ptr + reqs, num_above + num_taken, mask=mask)
        ties_left -= tl.sum(num_tied)


def assign_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    survival: torch.Tensor,
) -> None:
    """Admit the globally best `draft_budget` draft slots, in place.

    Every (request, step) slot is scored by its survival probability, the running
    product of that request's per-position confidences, and the highest scores win.
    Survival only decreases along a request, so the winners are always a prefix of
    each request's steps and only per-request counts are needed.

    `capacities` enters holding each request's scheduled draft count (which bounds its
    eligible slots) and leaves holding the admitted count. The caller only calls this
    when `draft_budget < capacities.sum()`, so every winner is a real draft slot.
    `survival` is int32 scratch with room for `num_reqs * num_steps` slots.
    """
    num_steps = confidence_probs.shape[1]
    _assign_draft_token_budget_kernel[(1,)](
        confidence_probs,
        confidence_probs.stride(0),
        idx_mapping,
        capacities,
        survival,
        idx_mapping.shape[0],
        draft_budget,
        NUM_STEPS=num_steps,
        BLOCK_REQS=256,
        BLOCK_SIZE=1024,
        num_warps=8,
    )


class AdaptiveBudgetInputError(ValueError):
    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


def select_draft_token_budget(
    confidence_probs: np.ndarray,
    capacities: np.ndarray,
    *,
    draft_cost_ms: np.ndarray,
    verify_cost_ms: np.ndarray,
    num_non_draft_tokens: int,
    num_sampling_requests: int,
    max_draft_budget: int | None = None,
) -> int:
    """Choose the smallest optimal budget using measured per-batch millisecond costs.

    Draft costs are indexed by request count; verify costs by total token count.
    Confidence rows and capacities are in batch order, including zero capacities.
    """
    if confidence_probs.ndim != 2 or confidence_probs.dtype.kind not in "fiu":
        raise AdaptiveBudgetInputError("confidence_probs", "expected a numeric matrix")
    if not np.isfinite(confidence_probs).all() or not (
        (confidence_probs >= 0) & (confidence_probs <= 1)
    ).all():
        raise AdaptiveBudgetInputError("confidence_probs", "must be finite in [0, 1]")
    num_reqs, num_steps = confidence_probs.shape
    if capacities.shape != (num_reqs,) or capacities.dtype.kind not in "iu":
        raise AdaptiveBudgetInputError("capacities", "expected one integer per request")
    if not ((capacities >= 0) & (capacities <= num_steps)).all():
        raise AdaptiveBudgetInputError("capacities", "must lie in [0, num_steps]")
    for name, value in (
        ("num_non_draft_tokens", num_non_draft_tokens),
        ("num_sampling_requests", num_sampling_requests),
        ("max_draft_budget", max_draft_budget),
    ):
        if name == "max_draft_budget" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise AdaptiveBudgetInputError(name, "expected a nonnegative integer")
    if num_sampling_requests > min(num_reqs, num_non_draft_tokens):
        raise AdaptiveBudgetInputError(
            "num_sampling_requests", "exceeds requests or non-draft tokens"
        )
    for name, costs in (
        ("draft_cost_ms", draft_cost_ms),
        ("verify_cost_ms", verify_cost_ms),
    ):
        if costs.ndim != 1 or costs.dtype.kind not in "fiu":
            raise AdaptiveBudgetInputError(name, "expected a numeric vector")
    available = sum(int(capacity) for capacity in capacities)
    limit = available if max_draft_budget is None else min(available, max_draft_budget)
    if len(draft_cost_ms) <= num_reqs:
        raise AdaptiveBudgetInputError("draft_cost_ms", "request count is not measured")
    if len(verify_cost_ms) <= num_non_draft_tokens + limit:
        raise AdaptiveBudgetInputError(
            "verify_cost_ms", "token counts are not measured"
        )
    draft_ms = np.float64(draft_cost_ms[num_reqs])
    verify_ms = verify_cost_ms[
        num_non_draft_tokens : num_non_draft_tokens + limit + 1
    ].astype(np.float64)
    if not np.isfinite(draft_ms) or draft_ms < 0:
        raise AdaptiveBudgetInputError("draft_cost_ms", "must be finite and nonnegative")
    if not np.isfinite(verify_ms).all() or (verify_ms <= 0).any():
        raise AdaptiveBudgetInputError("verify_cost_ms", "must be finite and positive")
    with np.errstate(over="ignore"):
        costs = draft_ms + verify_ms
    if not np.isfinite(costs).all() or (costs <= 0).any():
        raise AdaptiveBudgetInputError("total_cost_ms", "must be finite and positive")
    survival = np.cumprod(confidence_probs.astype(np.float64), axis=1)
    valid = np.arange(num_steps)[None, :] < capacities[:, None]
    scores = np.sort(survival[valid])[::-1][:limit]
    expected_tokens = np.concatenate(
        ([num_sampling_requests], num_sampling_requests + np.cumsum(scores))
    )
    return int(np.argmax(expected_tokens / costs))


def allocate_draft_token_budget(
    confidence_probs: torch.Tensor,
    idx_mapping: torch.Tensor,
    capacities: torch.Tensor,
    draft_budget: int,
    survival: torch.Tensor,
) -> None:
    """Validate eagerly; CUDA synchronization prevents capture of this wrapper."""
    if confidence_probs.ndim != 2 or confidence_probs.dtype != torch.float32:
        raise AdaptiveBudgetInputError("confidence_probs", "expected a float32 matrix")
    if not confidence_probs.is_cuda or confidence_probs.stride(1) != 1:
        raise AdaptiveBudgetInputError(
            "confidence_probs", "requires contiguous CUDA rows"
        )
    num_reqs, num_steps = idx_mapping.numel(), confidence_probs.shape[1]
    for name, tensor in (
        ("idx_mapping", idx_mapping),
        ("capacities", capacities),
        ("survival", survival),
    ):
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise AdaptiveBudgetInputError(name, "expected a contiguous vector")
        if tensor.device != confidence_probs.device:
            raise AdaptiveBudgetInputError(name, "must share the confidence device")
    if idx_mapping.dtype not in (torch.int32, torch.int64):
        raise AdaptiveBudgetInputError("idx_mapping", "expected integer request slots")
    if capacities.dtype != torch.int32 or capacities.numel() != num_reqs:
        raise AdaptiveBudgetInputError("capacities", "expected one int32 per request")
    if survival.dtype != torch.int32 or survival.numel() < num_reqs * num_steps:
        raise AdaptiveBudgetInputError(
            "survival", "insufficient contiguous int32 scratch"
        )
    if ((idx_mapping < 0) | (idx_mapping >= confidence_probs.shape[0])).any().item():
        raise AdaptiveBudgetInputError(
            "idx_mapping", "padding and invalid slots unsupported"
        )
    if ((capacities < 0) | (capacities > num_steps)).any().item():
        raise AdaptiveBudgetInputError("capacities", "must lie in [0, num_steps]")
    if (
        ~torch.isfinite(confidence_probs)
        | (confidence_probs < 0)
        | (confidence_probs > 1)
    ).any().item():
        raise AdaptiveBudgetInputError("confidence_probs", "must be finite in [0, 1]")
    if isinstance(draft_budget, bool) or not isinstance(draft_budget, Integral):
        raise AdaptiveBudgetInputError("draft_budget", "expected an integer")
    available = int(capacities.sum().item())
    if draft_budget < 0 or draft_budget > available:
        raise AdaptiveBudgetInputError(
            "draft_budget", "must lie in [0, total capacity]"
        )
    if draft_budget == 0:
        capacities.zero_()
    elif draft_budget < available:
        assign_draft_token_budget(
            confidence_probs, idx_mapping, capacities, int(draft_budget), survival
        )
