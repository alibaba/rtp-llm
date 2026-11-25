# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from numbers import Integral

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _pack_verification_rows_kernel(
    caps_ptr,
    ends_ptr,
    output_ptr,
    num_rows,
    CAPS_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    dense = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    request = dense // WIDTH
    position = dense % WIDTH
    valid = dense < num_rows
    cap = tl.load(caps_ptr + request * CAPS_STRIDE, valid, other=0)
    end = tl.load(ends_ptr + request, valid, other=0)
    output_row = end - cap - 1 + position
    tl.store(output_ptr + output_row, dense.to(tl.int64), valid & (position <= cap))


@triton.jit
def _pack_ep_tail_rows(
    scores_ptr,
    fallback_ptr,
    capacities_ptr,
    output_ptr,
    BATCH: tl.constexpr,
    CUT: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
):
    request = tl.arange(0, REQUEST_BLOCK)
    hop = tl.arange(0, 4)
    active = request < BATCH
    p = tl.load(
        scores_ptr + hop[:, None] * BATCH + request[None, :],
        mask=active[None, :],
        other=0.0,
    )
    valid = active & (tl.sum(((p >= 0.0) & (p <= 1.0)).to(tl.int32), axis=0) == 4)
    confidence = tl.where(valid, tl.sum(p, axis=0), float("inf"))
    capacities = tl.full([REQUEST_BLOCK], 5, tl.int32)
    for choice in tl.static_range((CUT + 1) // 2):
        weakest = tl.argmin(confidence, axis=0, tie_break_left=True)
        trim = tl.minimum(2, CUT - 2 * choice)
        capacities = tl.where(request == weakest, 5 - trim, capacities)
        confidence = tl.where(request == weakest, float("inf"), confidence)
    # Keep the original host plan when there aren't enough usable scores.
    fallback = tl.load(fallback_ptr + request, mask=active, other=0)
    enough_scores = tl.sum(valid.to(tl.int32), axis=0) >= (CUT + 1) // 2
    capacities = tl.where(enough_scores, capacities, fallback)
    tl.store(capacities_ptr + request, capacities, mask=active)
    widths = tl.where(active, capacities + 1, 0)
    starts = tl.cumsum(widths, axis=0) - widths
    position = tl.arange(0, 8)
    valid_row = active[:, None] & (position[None, :] <= capacities[:, None])
    tl.store(
        output_ptr + starts[:, None] + position[None, :],
        (request[:, None] * 6 + position[None, :]).to(tl.int64),
        mask=valid_row,
    )


def pack_ep_tail_rows(
    scores: torch.Tensor,
    fallback: torch.Tensor,
    total_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep MTP5 prefixes and the host-chosen physical shape, without D2H.

    The host fallback owns the budget and cap histogram. Trim up to two
    suffix positions per low-confidence request, concentrating cuts enough
    to eliminate a four-query Indexer tile when possible.
    """
    batch = fallback.numel()
    cut = batch * 6 - total_tokens
    if (
        not fallback.is_cuda
        or fallback.shape != (batch,)
        or fallback.dtype != torch.int32
        or not fallback.is_contiguous()
        or scores.device != fallback.device
        or scores.dtype != torch.float32
        or scores.shape != (4, batch)
        or not scores.is_contiguous()
        or not 0 < cut <= batch
    ):
        raise AdaptiveBudgetInputError(
            "scores", "invalid MTP5 tail budget or score layout"
        )
    capacities = torch.empty_like(fallback)
    rows = torch.empty(total_tokens, dtype=torch.int64, device=fallback.device)
    _pack_ep_tail_rows[(1,)](
        scores,
        fallback,
        capacities,
        rows,
        BATCH=batch,
        CUT=cut,
        REQUEST_BLOCK=triton.next_power_of_2(batch),
        num_warps=4,
    )
    return capacities, rows


# Fit to earlier full-width B14/MTP5 feedback; used only for the opt-in local
# margin experiment. Inference ranks log odds, so no sigmoid need be launched.
_MARGIN_SURVIVAL_COEFFICIENTS = (
    (-0.172850355, 0.819581561, 0.353639385),
    (-0.511972034, 0.797561552, 0.626072572),
    (-0.729625770, 0.801363478, 0.774889372),
    (-0.859486665, 0.818958801, 0.837569505),
    (-1.001554279, 0.805196673, 0.913103953),
)


# B14/MTP5 full-width Draft probe, genuine training ordinals 18–65.
# Columns: bias, historical prefix log-odds, and the four current Draft
# log1p Top2 gaps (proposals 2–5). Held-out ordinals 76–99 were never fit.
_FOUR_GAP_SURVIVAL_COEFFICIENTS = (
    (-1.054171831, 0.601229492, 0.458830186, 0.220959986, 0.161713496, 0.035412151),
    (-2.404322125, 0.551051488, 1.084158682, 0.198121441, 0.197105499, 0.082224571),
    (-3.505655225, 0.530519687, 0.878252113, 0.764493974, 0.275120148, 0.115853759),
    (-4.252901583, 0.531189599, 0.779063643, 0.660519435, 0.728659147, 0.143737013),
    (-4.934907450, 0.510239267, 0.704770974, 0.636047476, 0.620509142, 0.599176707),
)


@triton.jit
def _smaller(left, right):
    return tl.minimum(left, right)


@triton.jit(do_not_specialize=["optional"])
def _pack_ep_margin_rows(
    gaps_ptr,
    prior_ptr,
    fallback_ptr,
    caps_ptr,
    rows_ptr,
    optional,
    BATCH: tl.constexpr,
    REQUEST_BLOCK: tl.constexpr,
    COEFFICIENTS: tl.constexpr,
    FOUR_GAPS: tl.constexpr,
):
    request = tl.arange(0, REQUEST_BLOCK)
    hop = tl.arange(0, 8)
    active = request < BATCH

    gap_index = tl.arange(0, 4)
    gaps = tl.load(
        gaps_ptr + gap_index[None, :] * BATCH + request[:, None],
        mask=active[:, None],
        other=0.0,
    )
    finite = (gaps >= 0.0) & (gaps < float("inf"))
    valid = tl.sum((~finite & active[:, None]).to(tl.int32)) == 0
    safe_gaps = tl.where(finite, gaps, 0.0)
    if FOUR_GAPS:
        statistics = tl.log(1.0 + tl.minimum(safe_gaps, 32.0))
    else:
        statistic = tl.log(1.0 + tl.minimum(tl.min(safe_gaps, axis=1), 32.0))

    prior = tl.load(
        prior_ptr + request[:, None] * 5 + hop[None, :],
        mask=active[:, None] & (hop[None, :] < 5),
        other=0.5,
    )
    prior_valid = (prior >= 0.0) & (prior <= 1.0)
    valid = valid & (tl.sum((~prior_valid).to(tl.int32)) == 0)
    prior = tl.where(prior_valid, prior, 0.5)
    prior = tl.minimum(0.9999, tl.maximum(0.0001, prior))
    log_odds = tl.log(prior / (1.0 - prior))
    calibrated = tl.full((REQUEST_BLOCK, 8), 0.0, tl.float32)
    for depth in tl.static_range(5):
        if FOUR_GAPS:
            weights = tl.where(
                gap_index == 0,
                COEFFICIENTS[depth][2],
                tl.where(
                    gap_index == 1,
                    COEFFICIENTS[depth][3],
                    tl.where(
                        gap_index == 2, COEFFICIENTS[depth][4], COEFFICIENTS[depth][5]
                    ),
                ),
            )
            margin_score = tl.sum(statistics * weights[None, :], axis=1)
        else:
            margin_score = COEFFICIENTS[depth][2] * statistic
        score = (
            COEFFICIENTS[depth][0]
            + COEFFICIENTS[depth][1] * log_odds
            + margin_score[:, None]
        )
        calibrated = tl.where(hop[None, :] == depth, score, calibrated)
    scores = tl.associative_scan(calibrated, 1, _smaller)
    eligible = active[:, None] & (hop[None, :] >= 1) & (hop[None, :] < 5)
    flat_scores = tl.reshape(
        tl.where(eligible, scores, -float("inf")), (REQUEST_BLOCK * 8,)
    )
    order = tl.arange(0, REQUEST_BLOCK * 8)
    sorted_scores = tl.sort(flat_scores, descending=True)
    cutoff = tl.sum(tl.where(order == optional - 1, sorted_scores, 0.0))
    higher = flat_scores > cutoff
    tied = flat_scores == cutoff
    left = optional - tl.sum(higher.to(tl.int32))
    # Original row order breaks ties, so a retained depth always owns its prefix.
    keep = higher | (tied & (tl.cumsum(tied.to(tl.int32)) <= left))
    caps = 1 + tl.sum(tl.reshape(keep.to(tl.int32), (REQUEST_BLOCK, 8)), axis=1)
    fallback = tl.load(fallback_ptr + request, mask=active, other=0)
    caps = tl.where(valid, caps, fallback)
    tl.store(caps_ptr + request, caps, mask=active)
    widths = tl.where(active, caps + 1, 0)
    starts = tl.cumsum(widths) - widths
    tl.store(
        rows_ptr + starts[:, None] + hop[None, :],
        (request[:, None] * 6 + hop[None, :]).to(tl.int64),
        active[:, None] & (hop[None, :] <= caps[:, None]),
    )


def pack_ep_margin_rows(
    gaps: torch.Tensor,
    prior: torch.Tensor,
    fallback: torch.Tensor,
    total_tokens: int,
    *,
    four_gaps: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder admitted prefixes within a fixed rank-local row budget."""
    if (
        fallback.dim() != 1
        or not fallback.is_cuda
        or fallback.dtype != torch.int32
        or not fallback.is_contiguous()
        or not 2 * fallback.numel() < total_tokens < 6 * fallback.numel()
        or not 1 <= fallback.numel() <= 32
    ):
        raise AdaptiveBudgetInputError("fallback", "invalid MTP5 margin allocation")
    batch = fallback.numel()
    if (
        gaps.shape != (4, batch)
        or prior.shape != (batch, 5)
        or any(
            tensor.device != fallback.device
            or tensor.dtype != torch.float32
            or not tensor.is_contiguous()
            for tensor in (gaps, prior)
        )
    ):
        raise AdaptiveBudgetInputError(
            "gaps", "invalid margins or historical probability layout"
        )
    caps = torch.empty_like(fallback)
    rows = torch.empty(total_tokens, dtype=torch.int64, device=fallback.device)
    _pack_ep_margin_rows[(1,)](
        gaps,
        prior,
        fallback,
        caps,
        rows,
        total_tokens - 2 * batch,
        BATCH=batch,
        REQUEST_BLOCK=triton.next_power_of_2(batch),
        COEFFICIENTS=(
            _FOUR_GAP_SURVIVAL_COEFFICIENTS
            if four_gaps
            else _MARGIN_SURVIVAL_COEFFICIENTS
        ),
        FOUR_GAPS=four_gaps,
        num_warps=4,
    )
    return caps, rows


def pack_verification_rows(
    capacities: torch.Tensor, num_steps: int, total_tokens: int
) -> torch.Tensor:
    """Pack validated per-request prefixes without sorting or device readback.

    The controller owns the invariant total_tokens == sum(capacities + 1),
    with every capacity in [0, num_steps]. Use its already-known CPU budget;
    recomputing it from the device would introduce a synchronization point.
    """
    if (
        capacities.ndim != 1
        or not capacities.is_cuda
        or capacities.dtype not in (torch.int32, torch.int64)
    ):
        raise AdaptiveBudgetInputError("capacities", "expected a CUDA integer vector")
    if (
        isinstance(num_steps, bool)
        or not isinstance(num_steps, Integral)
        or num_steps < 0
    ):
        raise AdaptiveBudgetInputError("num_steps", "expected a nonnegative integer")
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, Integral):
        raise AdaptiveBudgetInputError("total_tokens", "expected an integer budget")
    width = int(num_steps) + 1
    dense_rows = capacities.numel() * width
    if not capacities.numel() <= total_tokens <= dense_rows:
        raise AdaptiveBudgetInputError("total_tokens", "outside prefix bounds")
    if total_tokens == dense_rows:
        return torch.arange(dense_rows, dtype=torch.int64, device=capacities.device)
    if total_tokens == capacities.numel():
        return torch.arange(
            0, dense_rows, width, dtype=torch.int64, device=capacities.device
        )
    output = torch.empty(total_tokens, dtype=torch.int64, device=capacities.device)
    if capacities.numel() == 0:
        return output
    ends = torch.cumsum(capacities + 1, dim=0, dtype=torch.int64)
    _pack_verification_rows_kernel[(triton.cdiv(dense_rows, 256),)](
        capacities,
        ends,
        output,
        dense_rows,
        CAPS_STRIDE=capacities.stride(0),
        WIDTH=width,
        BLOCK=256,
    )
    return output


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
    known_ptr,
    num_reqs,
    draft_budget,
    HAS_KNOWN: tl.constexpr,
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
        known = tl.full([BLOCK_REQS], True, tl.int1)
        if HAS_KNOWN:
            known = tl.load(known_ptr + reqs, mask=mask, other=False)
        survival = tl.full([BLOCK_REQS], 1.0, tl.float32)
        for step in tl.static_range(NUM_STEPS):
            prob = tl.load(
                confidence_probs_ptr + req_state_idx * confidence_probs_stride + step,
                mask=mask,
                other=0.0,
            )
            # Unknown predictions are optimistic; sanitize in the ranking kernel
            # rather than launching gather/where/isfinite kernels per round.
            prob = tl.where(known & (prob >= 0.0) & (prob <= 1.0), prob, 1.0)
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
    known: torch.Tensor | None = None,
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
    if known is not None and (
        known.shape != idx_mapping.shape
        or known.dtype != torch.bool
        or known.device != confidence_probs.device
        or not known.is_contiguous()
    ):
        raise ValueError("known must be a contiguous boolean mask in request order")
    num_steps = confidence_probs.shape[1]
    _assign_draft_token_budget_kernel[(1,)](
        confidence_probs,
        confidence_probs.stride(0),
        idx_mapping,
        capacities,
        survival,
        known,
        idx_mapping.shape[0],
        draft_budget,
        HAS_KNOWN=known is not None,
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
    if (
        not np.isfinite(confidence_probs).all()
        or not ((confidence_probs >= 0) & (confidence_probs <= 1)).all()
    ):
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
        raise AdaptiveBudgetInputError(
            "draft_cost_ms", "must be finite and nonnegative"
        )
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
        (
            ~torch.isfinite(confidence_probs)
            | (confidence_probs < 0)
            | (confidence_probs > 1)
        )
        .any()
        .item()
    ):
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
