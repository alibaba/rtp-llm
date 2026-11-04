#!/usr/bin/env python3
"""Evaluate admission-time active decode progress as a latency feature.

The request lifecycle audit does not log every request's output length at every
step. For non-preempted requests, this reconstructs a causal estimate from the
successful decode steps completed after first output. The estimate only uses
events visible before each admission; terminal output length is never used.
"""

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import gzip
import heapq
import json
import math
from pathlib import Path
import re

import numpy as np

from AnalyzeLatencyFeatureNeighbors import FEATURES, pressure_phases, read_samples
from AnalyzeLatencyPhaseLeastSquares import (
    ACTIVE,
    LOWER,
    PARAMETER_SCALES,
    input_matrix,
    shape,
    unpack,
)
from AnalyzeVllmRequestLifecycles import read_events


PARAMETERS = re.compile(
    r"\[navi-latency-training-update\].*?"
    r"parameters_before=\[([^]]*)\] parameters_after=\[([^]]*)\]"
)
GENERATED_TOKEN_SCALE = 100_000.0
CORRECTION_SEGMENTS = (
    "held_all",
    "held_high_waiting",
    "held_low_waiting",
    "held_rising",
    "held_falling",
    "held_flat",
)
RIDGE_STRENGTHS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
ROBUST_CORRECTION_VARIANTS = (
    "decode_count",
    "generated_tokens",
    "decode_count_and_generated_tokens",
    "mean_generated_tokens",
    "decode_progress_pressure",
    "decode_progress_phase",
)


def _decode_step_prefix(events):
    steps = []
    exact_steps = 0
    for event in events:
        if event.get("event") != "execution_step" or not event.get("success", False):
            continue
        decode_requests = int(event.get("decode_requests", 0))
        decode_tokens = int(event.get("decode_tokens", 0))
        completed_ns = event.get("completed_perf_ns")
        if decode_requests <= 0 or decode_tokens <= 0 or completed_ns is None:
            continue
        steps.append((int(completed_ns), decode_tokens / decode_requests))
        exact_steps += int(decode_tokens == decode_requests)
    steps.sort()
    times = []
    prefix = [0.0]
    for completed_ns, tokens_per_request in steps:
        times.append(completed_ns)
        prefix.append(prefix[-1] + tokens_per_request)
    return times, prefix, exact_steps


def _prefix_at(times, prefix, monotonic_ns):
    return prefix[bisect_right(times, monotonic_ns)]


def reconstruct_decode_progress(events):
    """Return admission-time decode count and generated-token estimate by batch."""
    sessions = {event["session"] for event in events}
    if len(sessions) != 1:
        raise ValueError(f"expected one audit session, found {sorted(sessions)}")
    ordered = sorted(
        (event for event in events if "sequence" in event),
        key=lambda event: event["sequence"],
    )
    times, prefix, exact_steps = _decode_step_prefix(events)
    requests = {}
    active_slots = {}
    admissions = {}
    batch_requests = {}
    preempted_requests = set()

    def request_state(request_id):
        return requests.setdefault(
            request_id, {"slots": [], "first_output": None, "terminal": None}
        )

    for event in ordered:
        kind = event.get("event")
        request_id = str(event.get("request_id"))
        if kind == "admitted":
            batch_id = int(event["batch_id"])
            admissions.setdefault(batch_id, int(event["monotonic_ns"]))
            batch_requests.setdefault(batch_id, []).append(request_id)
            request_state(request_id)
        elif kind == "slot_acquire":
            state = request_state(request_id)
            if event.get("resumed") or int(event.get("num_preemptions", 0)) > 0:
                preempted_requests.add(request_id)
            active_slots[request_id] = event
            state["last_acquire"] = event
        elif kind == "first_output":
            request_state(request_id)["first_output"] = event
        elif kind == "slot_release":
            state = request_state(request_id)
            acquired = active_slots.pop(request_id, None)
            if acquired is not None:
                state["slots"].append((acquired, event))
            state.pop("last_acquire", None)
        elif kind == "terminal":
            request_state(request_id)["terminal"] = event

    intervals = []
    skipped_preempted_intervals = 0
    for request_id, state in requests.items():
        first_output = state["first_output"]
        if first_output is None:
            continue
        slots = list(state["slots"])
        acquired = active_slots.get(request_id)
        if acquired is not None:
            slots.append((acquired, None))
        if request_id in preempted_requests:
            skipped_preempted_intervals += len(slots)
            continue
        first_ns = int(first_output["monotonic_ns"])
        first_tokens = max(1, int(first_output.get("output_tokens", 1)))
        offset = first_tokens - _prefix_at(times, prefix, first_ns)
        for acquired, released in slots:
            start_ns = max(first_ns, int(acquired["monotonic_ns"]))
            end_ns = (
                int(released["monotonic_ns"])
                if released is not None
                else math.inf
            )
            if start_ns < end_ns:
                intervals.append((start_ns, end_ns, offset))

    intervals.sort()
    active = []
    active_offsets = 0.0
    interval_index = 0
    result = {}
    for batch_id, admission_ns in sorted(admissions.items(), key=lambda item: item[1]):
        while interval_index < len(intervals) and intervals[interval_index][0] <= admission_ns:
            _, end_ns, offset = intervals[interval_index]
            heapq.heappush(active, (end_ns, offset))
            active_offsets += offset
            interval_index += 1
        while active and active[0][0] <= admission_ns:
            _, offset = heapq.heappop(active)
            active_offsets -= offset
        decode_count = len(active)
        generated_tokens = (
            decode_count * _prefix_at(times, prefix, admission_ns) + active_offsets
        )
        target_requests = batch_requests.get(batch_id, ())
        target_aborted = any(
            requests[request_id].get("terminal", {}).get("status")
            == "FINISHED_ABORTED"
            for request_id in target_requests
        )
        result[batch_id] = {
            "active_decode_count": decode_count,
            "generated_decode_tokens": max(0.0, generated_tokens),
            "mean_generated_decode_tokens": (
                max(0.0, generated_tokens) / decode_count if decode_count else 0.0
            ),
            "target_aborted": target_aborted,
        }
    diagnostics = {
        "decode_steps": len(times),
        "unit_decode_steps": exact_steps,
        "unit_decode_step_ratio": exact_steps / len(times) if times else None,
        "preempted_requests_skipped": len(preempted_requests),
        "preempted_intervals_skipped": skipped_preempted_intervals,
    }
    return result, diagnostics


def _metrics(target, prediction):
    error = np.abs(prediction - target)
    return {
        "n": int(len(target)),
        "mae_ms": float(np.mean(error)),
        "p95_abs_ms": float(np.percentile(error, 95)),
        "mean_relative_error": float(
            np.mean(error / np.maximum(target, 100.0))
        ),
        "signed_error_ms": float(np.mean(prediction - target)),
    }


def read_parameter_starts(path):
    states = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line in stream:
            match = PARAMETERS.search(line)
            if not match:
                continue
            before = [float(value) for value in match.group(1).split(",")]
            after = [float(value) for value in match.group(2).split(",")]
            if len(before) != 17 or len(after) != 17:
                raise ValueError("expected v21 parameter count")
            if not states:
                states.append(before)
            states.append(after)
    if not states:
        raise ValueError("no latency training parameter states")
    # This reader accepts v21 physical coefficients, while fitting uses coordinates.
    return [(np.asarray(state) / PARAMETER_SCALES[:17]).tolist()
            for state in (states[0], states[len(states) // 2], states[-1])]


def _augmented_prediction_and_jacobian(inputs, active, generated_tokens):
    parameters = unpack(active) * PARAMETER_SCALES[:17]
    generated = generated_tokens / GENERATED_TOKEN_SCALE
    linear = (
        inputs[:, :7] @ parameters[:7]
        + inputs[:, 7:12] @ parameters[10:15]
        + inputs[:, 13] * parameters[16]
        + generated * active[-1]
    ) / 320.0
    zero_linear = parameters[0] / 320.0
    curvature = parameters[8]
    root = np.sqrt(linear * linear + curvature)
    zero_root = np.sqrt(zero_linear * zero_linear + curvature)
    total_prompt = inputs[:, 2] + inputs[:, 3] + inputs[:, 8] + inputs[:, 9]
    raw = (
        parameters[7]
        + 1500.0
        * (shape(linear, curvature) - shape(zero_linear, curvature))
        + parameters[9] * total_prompt
        + parameters[15] * inputs[:, 12]
    )
    jacobian = np.zeros((len(inputs), 18))
    slope = 1500.0 * (1.0 + linear / root) / 320.0
    jacobian[:, :7] = slope[:, None] * inputs[:, :7]
    jacobian[:, 0] -= (
        1500.0 * (1.0 + zero_linear / zero_root) / 320.0
    )
    jacobian[:, 7] = 1.0
    jacobian[:, 8] = 750.0 * (1.0 / root - 1.0 / zero_root)
    jacobian[:, 9] = total_prompt
    jacobian[:, 10:15] = slope[:, None] * inputs[:, 7:12]
    jacobian[:, 15] = inputs[:, 12]
    jacobian[:, 16] = slope * inputs[:, 13]
    jacobian[:, :17] *= PARAMETER_SCALES[:17]
    jacobian[:, 17] = slope * generated
    jacobian[raw < 0.0] = 0.0
    indices = list(ACTIVE) + [17]
    return np.maximum(raw, 0.0), jacobian[:, indices]


def _base_prediction_and_jacobian(inputs, active):
    generated = np.zeros(len(inputs))
    augmented = np.concatenate((np.asarray(active, dtype=float), [0.0]))
    prediction, jacobian = _augmented_prediction_and_jacobian(
        inputs, augmented, generated
    )
    return prediction, jacobian[:, :-1]


def _fit_formula(inputs, generated, target, starts, augmented):
    from scipy.optimize import least_squares

    best = None
    for initial in starts:
        values = np.asarray(initial, dtype=float)[ACTIVE]
        lower = LOWER
        if augmented:
            values = np.concatenate((values, [0.0]))
            lower = np.concatenate((LOWER, [-np.inf]))
        values = np.maximum(values, lower + 1e-9)
        cache = [None, None, None]

        def evaluate(candidate):
            if cache[0] is None or not np.array_equal(candidate, cache[0]):
                if augmented:
                    prediction, jacobian = _augmented_prediction_and_jacobian(
                        inputs, candidate, generated
                    )
                else:
                    prediction, jacobian = _base_prediction_and_jacobian(
                        inputs, candidate
                    )
                cache[:] = [candidate.copy(), prediction, jacobian]
            return cache[1], cache[2]

        def residual(candidate):
            prediction = evaluate(candidate)[0]
            return np.log1p(prediction) - np.log1p(target)

        def jacobian(candidate):
            prediction, derivative = evaluate(candidate)
            return derivative / (1.0 + prediction[:, None])

        result = least_squares(
            residual,
            values,
            jac=jacobian,
            bounds=(lower, np.full(len(values), np.inf)),
            x_scale="jac",
            max_nfev=1200,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        if best is None or result.cost < best.cost:
            best = result
    return best


def fit_static_formula_ablation(rows, progress, starts):
    matched = [
        row
        for row in rows
        if row["batch_id"] in progress
        and not progress[row["batch_id"]]["target_aborted"]
    ]
    if len(matched) < 40:
        return None
    raw_features = [
        [row["batch_count"], *[row[name] for name in FEATURES]]
        for row in matched
    ]
    inputs = input_matrix(raw_features)
    generated = np.asarray(
        [progress[row["batch_id"]]["generated_decode_tokens"] for row in matched],
        dtype=float,
    )
    target = np.asarray([row["target_ms"] for row in matched], dtype=float)
    event_time = np.asarray([row["event_time_us"] for row in matched], dtype=np.int64)
    held = (event_time // 120_000_000) % 4 == 3
    if np.sum(~held) < 40 or not np.any(held):
        return None
    baseline_fit = _fit_formula(
        inputs[~held], generated[~held], target[~held], starts, False
    )
    augmented_fit = _fit_formula(
        inputs[~held], generated[~held], target[~held], starts, True
    )
    baseline = _base_prediction_and_jacobian(inputs, baseline_fit.x)[0]
    augmented = _augmented_prediction_and_jacobian(
        inputs, augmented_fit.x, generated
    )[0]
    high = np.asarray([row["waiting_batch_size"] >= 33 for row in matched])
    evaluations = {}
    for name, mask in (
        ("train_all", ~held),
        ("held_all", held),
        ("held_high_waiting", held & high),
        ("held_low_waiting", held & ~high),
    ):
        evaluations[name] = (
            {
                "baseline": _metrics(target[mask], baseline[mask]),
                "augmented": _metrics(target[mask], augmented[mask]),
            }
            if np.any(mask)
            else None
        )
    return {
        "matched_samples": len(matched),
        "train_samples": int(np.sum(~held)),
        "held_samples": int(np.sum(held)),
        "baseline_success": bool(baseline_fit.success),
        "augmented_success": bool(augmented_fit.success),
        "baseline_nfev": int(baseline_fit.nfev),
        "augmented_nfev": int(augmented_fit.nfev),
        "generated_token_parameter": float(augmented_fit.x[-1]),
        "evaluations": evaluations,
    }


def fit_logged_prediction_correction(rows, progress, feature_names):
    matched = [
        row
        for row in rows
        if row["batch_id"] in progress
        and not progress[row["batch_id"]]["target_aborted"]
        and row.get("prediction_before_ms", -1) >= 0
    ]
    if len(matched) < 20:
        return None
    event_time = np.asarray([row["event_time_us"] for row in matched], dtype=np.int64)
    held = (event_time // 120_000_000) % 4 == 3
    if np.sum(~held) <= len(feature_names) + 1 or not np.any(held):
        return None
    target = np.asarray([row["target_ms"] for row in matched], dtype=float)
    baseline = np.asarray(
        [row["prediction_before_ms"] for row in matched], dtype=float
    )
    raw = np.asarray(
        [
            [progress[row["batch_id"]][name] for name in feature_names]
            for row in matched
        ],
        dtype=float,
    )
    mean = np.mean(raw[~held], axis=0)
    scale = np.std(raw[~held], axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    residual = np.log1p(target) - np.log1p(baseline)
    coefficient = np.linalg.lstsq(design[~held], residual[~held], rcond=None)[0]
    corrected = np.maximum(
        0.0, np.expm1(np.log1p(baseline) + design @ coefficient)
    )
    high = np.asarray([row["waiting_batch_size"] >= 33 for row in matched])
    evaluations = {}
    for name, mask in (
        ("held_all", held),
        ("held_high_waiting", held & high),
        ("held_low_waiting", held & ~high),
    ):
        if not np.any(mask):
            evaluations[name] = None
            continue
        evaluations[name] = {
            "baseline": _metrics(target[mask], baseline[mask]),
            "corrected": _metrics(target[mask], corrected[mask]),
        }
    return {
        "matched_samples": len(matched),
        "train_samples": int(np.sum(~held)),
        "held_samples": int(np.sum(held)),
        "features": list(feature_names),
        "train_mean": mean.tolist(),
        "train_scale": scale.tolist(),
        "coefficients": coefficient.tolist(),
        "evaluations": evaluations,
    }


def _decode_progress_design(matched, progress, variant, phases):
    count = np.asarray(
        [progress[row["batch_id"]]["active_decode_count"] for row in matched],
        dtype=float,
    )
    generated = np.asarray(
        [progress[row["batch_id"]]["generated_decode_tokens"] for row in matched],
        dtype=float,
    )
    mean_generated = np.divide(
        generated,
        count,
        out=np.zeros_like(generated),
        where=count > 0,
    )
    waiting = np.asarray(
        [row["waiting_batch_size"] for row in matched], dtype=float
    )
    running = np.asarray(
        [row["running_batch_size"] for row in matched], dtype=float
    )
    phase = np.asarray(phases)
    columns = {
        "decode_count": (("active_decode_count",), (count,)),
        "generated_tokens": (("generated_decode_tokens",), (generated,)),
        "decode_count_and_generated_tokens": (
            ("active_decode_count", "generated_decode_tokens"),
            (count, generated),
        ),
        "mean_generated_tokens": (
            ("mean_generated_decode_tokens",),
            (mean_generated,),
        ),
        "decode_progress_pressure": (
            (
                "active_decode_count",
                "generated_decode_tokens",
                "mean_generated_decode_tokens",
                "active_decode_count_x_waiting",
                "generated_decode_tokens_x_waiting",
                "mean_generated_decode_tokens_x_waiting",
                "active_decode_count_x_running",
                "generated_decode_tokens_x_running",
            ),
            (
                count,
                generated,
                mean_generated,
                count * waiting,
                generated * waiting,
                mean_generated * waiting,
                count * running,
                generated * running,
            ),
        ),
        "decode_progress_phase": (
            (
                "active_decode_count",
                "generated_decode_tokens",
                "active_decode_count_x_rising",
                "generated_decode_tokens_x_rising",
                "active_decode_count_x_falling",
                "generated_decode_tokens_x_falling",
            ),
            (
                count,
                generated,
                count * (phase == "rising"),
                generated * (phase == "rising"),
                count * (phase == "falling"),
                generated * (phase == "falling"),
            ),
        ),
    }
    if variant not in columns:
        raise ValueError(f"unknown decode progress variant: {variant}")
    names, values = columns[variant]
    return names, np.column_stack(values)


def _ridge_fit(design, target, strength):
    penalty = np.zeros((design.shape[1], design.shape[1]))
    penalty[1:, 1:] = np.eye(design.shape[1] - 1) * math.sqrt(strength)
    return np.linalg.lstsq(
        np.vstack((design, penalty)),
        np.concatenate((target, np.zeros(design.shape[1]))),
        rcond=None,
    )[0]


def _log_error(target, prediction):
    return float(
        np.mean(np.abs(np.log1p(prediction) - np.log1p(target)))
    )


def fit_robust_decode_progress_correction(rows, progress, variant):
    phase_records = pressure_phases(rows)
    phase_by_batch = {
        row["batch_id"]: phase["phase"]
        for row, phase in zip(rows, phase_records)
    }
    matched = [
        row
        for row in rows
        if row["batch_id"] in progress
        and not progress[row["batch_id"]]["target_aborted"]
        and row.get("prediction_before_ms", -1) >= 0
    ]
    if len(matched) < 40:
        return None
    phases = [phase_by_batch[row["batch_id"]] for row in matched]
    feature_names, raw = _decode_progress_design(
        matched, progress, variant, phases
    )
    event_time = np.asarray(
        [row["event_time_us"] for row in matched], dtype=np.int64
    )
    block = (event_time // 120_000_000) % 4
    held = block == 3
    validation = block == 2
    fit = ~held & ~validation
    if (
        np.sum(fit) <= raw.shape[1] + 1
        or not np.any(validation)
        or not np.any(held)
    ):
        return None
    mean = np.mean(raw[fit], axis=0)
    scale = np.std(raw[fit], axis=0)
    scale[scale == 0.0] = 1.0
    design = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    target = np.asarray([row["target_ms"] for row in matched], dtype=float)
    baseline = np.asarray(
        [row["prediction_before_ms"] for row in matched], dtype=float
    )
    residual = np.log1p(target) - np.log1p(baseline)
    validation_candidates = [
        {
            "ridge_strength": None,
            "log_error": _log_error(target[validation], baseline[validation]),
        }
    ]
    for strength in RIDGE_STRENGTHS:
        coefficient = _ridge_fit(design[fit], residual[fit], strength)
        corrected = np.maximum(
            0.0,
            np.expm1(np.log1p(baseline) + design @ coefficient),
        )
        validation_candidates.append(
            {
                "ridge_strength": strength,
                "log_error": _log_error(
                    target[validation], corrected[validation]
                ),
            }
        )
    selected = min(validation_candidates, key=lambda item: item["log_error"])
    train = ~held
    if selected["ridge_strength"] is None:
        coefficient = np.zeros(design.shape[1])
        corrected = baseline.copy()
    else:
        # Recompute normalization and coefficients with all non-held blocks.
        mean = np.mean(raw[train], axis=0)
        scale = np.std(raw[train], axis=0)
        scale[scale == 0.0] = 1.0
        design = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
        coefficient = _ridge_fit(
            design[train], residual[train], selected["ridge_strength"]
        )
        corrected = np.maximum(
            0.0,
            np.expm1(np.log1p(baseline) + design @ coefficient),
        )
    phase = np.asarray(phases)
    high = np.asarray([row["waiting_batch_size"] >= 33 for row in matched])
    masks = {
        "held_all": held,
        "held_high_waiting": held & high,
        "held_low_waiting": held & ~high,
        "held_rising": held & (phase == "rising"),
        "held_falling": held & (phase == "falling"),
        "held_flat": held & (phase == "flat"),
    }
    evaluations = {
        name: (
            {
                "baseline": _metrics(target[mask], baseline[mask]),
                "corrected": _metrics(target[mask], corrected[mask]),
            }
            if np.any(mask)
            else None
        )
        for name, mask in masks.items()
    }
    return {
        "matched_samples": len(matched),
        "fit_samples": int(np.sum(fit)),
        "validation_samples": int(np.sum(validation)),
        "held_samples": int(np.sum(held)),
        "features": list(feature_names),
        "selected_ridge_strength": selected["ridge_strength"],
        "validation_candidates": validation_candidates,
        "train_mean": mean.tolist(),
        "train_scale": scale.tolist(),
        "coefficients": coefficient.tolist(),
        "evaluations": evaluations,
    }


def evaluate_pairs(pairs, progress):
    analyzed = []
    for pair in pairs:
        if any(side.get("crosses_abort_wave") for side in pair["sides"]):
            continue
        sides = []
        for side in pair["sides"]:
            feature = progress.get(side["batch_id"])
            if feature is None or feature["target_aborted"]:
                break
            sides.append({**side, **feature})
        if len(sides) != 2:
            continue
        slower = max(sides, key=lambda side: side["target_ms"])
        faster = min(sides, key=lambda side: side["target_ms"])
        generated_delta = (
            slower["generated_decode_tokens"]
            - faster["generated_decode_tokens"]
        )
        count_delta = (
            slower["active_decode_count"] - faster["active_decode_count"]
        )
        generated_relative = abs(generated_delta) / max(
            1.0,
            slower["generated_decode_tokens"],
            faster["generated_decode_tokens"],
        )
        analyzed.append(
            {
                "batches": pair["batches"],
                "target_gap_ms": pair["target_gap_ms"],
                "slower_batch": slower["batch_id"],
                "active_decode_count_delta": count_delta,
                "generated_decode_tokens_delta": generated_delta,
                "generated_decode_tokens_relative_difference": generated_relative,
                "slower": {
                    key: slower[key]
                    for key in (
                        "batch_id",
                        "target_ms",
                        "active_decode_count",
                        "generated_decode_tokens",
                        "mean_generated_decode_tokens",
                    )
                },
                "faster": {
                    key: faster[key]
                    for key in (
                        "batch_id",
                        "target_ms",
                        "active_decode_count",
                        "generated_decode_tokens",
                        "mean_generated_decode_tokens",
                    )
                },
            }
        )
    return {
        "pair_count": len(analyzed),
        "slower_has_fewer_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] < 0 for pair in analyzed
        ),
        "slower_has_more_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] > 0 for pair in analyzed
        ),
        "same_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] == 0 for pair in analyzed
        ),
        "generated_difference_over_5pct": sum(
            pair["generated_decode_tokens_relative_difference"] > 0.05
            for pair in analyzed
        ),
        "active_decode_count_difference_over_1": sum(
            abs(pair["active_decode_count_delta"]) > 1 for pair in analyzed
        ),
        "examples": sorted(
            analyzed, key=lambda pair: pair["target_gap_ms"], reverse=True
        )[:5],
    }


def _pair_stats(pairs):
    return {
        "pair_count": len(pairs),
        "slower_has_fewer_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] < 0 for pair in pairs
        ),
        "slower_has_more_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] > 0 for pair in pairs
        ),
        "same_generated_tokens": sum(
            pair["generated_decode_tokens_delta"] == 0 for pair in pairs
        ),
        "generated_difference_over_5pct": sum(
            pair["generated_decode_tokens_relative_difference"] > 0.05
            for pair in pairs
        ),
        "active_decode_count_difference_over_1": sum(
            abs(pair["active_decode_count_delta"]) > 1 for pair in pairs
        ),
    }


def evaluate_all_similar_pairs(rows, progress, definition):
    matched = [row for row in rows if row["batch_id"] in progress]
    features = np.asarray(
        [[row[name] for name in FEATURES] for row in matched], dtype=float
    )
    finish = np.asarray([row["event_time_us"] for row in matched], dtype=np.int64)
    ready = finish - np.asarray(
        [row["target_ms"] * 1000.0 for row in matched], dtype=np.int64
    )
    target = np.asarray([row["target_ms"] for row in matched], dtype=float)
    scale = np.std(features, axis=0)
    scale[scale == 0.0] = 1.0
    minimum_separation_us = int(
        definition["min_finish_and_ready_separation_s"] * 1_000_000
    )
    abort_start_us, abort_end_us = definition["abort_wave_event_time_us"]
    crosses_abort = (ready <= abort_end_us) & (finish >= abort_start_us)
    found = []
    for left in range(len(matched)):
        right = np.arange(left + 1, len(matched))
        if not len(right):
            continue
        separated = (
            np.abs(finish[right] - finish[left]) >= minimum_separation_us
        ) & (np.abs(ready[right] - ready[left]) >= minimum_separation_us)
        gap = np.abs(target[right] - target[left])
        relative = np.max(
            np.abs(features[right] - features[left])
            / np.maximum(
                1.0,
                np.maximum(np.abs(features[right]), np.abs(features[left])),
            ),
            axis=1,
        )
        distance = np.linalg.norm(
            (features[right] - features[left]) / scale, axis=1
        )
        selected = right[
            separated
            & (gap >= definition["min_target_gap_ms"])
            & (relative <= definition["max_relative_difference"])
            & (distance <= definition["distance_limit"])
        ]
        for other in selected:
            if crosses_abort[left] or crosses_abort[other]:
                continue
            left_progress = progress[matched[left]["batch_id"]]
            right_progress = progress[matched[other]["batch_id"]]
            if left_progress["target_aborted"] or right_progress["target_aborted"]:
                continue
            slower, faster = (
                (left, other) if target[left] > target[other] else (other, left)
            )
            slower_progress = progress[matched[slower]["batch_id"]]
            faster_progress = progress[matched[faster]["batch_id"]]
            generated_delta = (
                slower_progress["generated_decode_tokens"]
                - faster_progress["generated_decode_tokens"]
            )
            generated_relative = abs(generated_delta) / max(
                1.0,
                slower_progress["generated_decode_tokens"],
                faster_progress["generated_decode_tokens"],
            )
            found.append(
                {
                    "indexes": (left, int(other)),
                    "batches": (
                        matched[left]["batch_id"], matched[other]["batch_id"]
                    ),
                    "target_gap_ms": abs(target[left] - target[other]),
                    "active_decode_count_delta": (
                        slower_progress["active_decode_count"]
                        - faster_progress["active_decode_count"]
                    ),
                    "generated_decode_tokens_delta": generated_delta,
                    "generated_decode_tokens_relative_difference": (
                        generated_relative
                    ),
                }
            )
    used = set()
    disjoint = []
    for pair in sorted(found, key=lambda item: item["target_gap_ms"], reverse=True):
        if any(index in used for index in pair["indexes"]):
            continue
        used.update(pair["indexes"])
        disjoint.append(pair)
    return {
        "all_no_abort": _pair_stats(found),
        "greedy_disjoint_no_abort": _pair_stats(disjoint),
        "examples": [
            {key: value for key, value in pair.items() if key != "indexes"}
            for pair in sorted(
                disjoint, key=lambda item: item["target_gap_ms"], reverse=True
            )[:5]
        ],
    }


def evaluate_node(job):
    node, deployment, log_path, pairs, definition = job
    events, malformed = read_events([Path(log_path)])
    progress, diagnostics = reconstruct_decode_progress(events)
    rows = read_samples(Path(log_path))
    starts = read_parameter_starts(Path(log_path))
    correction = {
        "decode_count": fit_logged_prediction_correction(
            rows, progress, ("active_decode_count",)
        ),
        "generated_tokens": fit_logged_prediction_correction(
            rows, progress, ("generated_decode_tokens",)
        ),
        "decode_count_and_generated_tokens": fit_logged_prediction_correction(
            rows,
            progress,
            ("active_decode_count", "generated_decode_tokens"),
        ),
    }
    robust_correction = {
        variant: fit_robust_decode_progress_correction(
            rows, progress, variant
        )
        for variant in ROBUST_CORRECTION_VARIANTS
    }
    return {
        "node": node,
        "deployment": deployment,
        "log": str(log_path),
        "malformed_audit_lines": malformed,
        "progress_batch_count": len(progress),
        "diagnostics": diagnostics,
        "pairs": evaluate_pairs(pairs, progress),
        "all_similar_pairs": evaluate_all_similar_pairs(
            rows, progress, definition
        ),
        "held_out_correction": correction,
        "robust_held_out_correction": robust_correction,
        "static_formula_ablation": fit_static_formula_ablation(
            rows, progress, starts
        ),
    }


def _weighted_metrics(nodes, variant, segment, predictor):
    records = []
    for node in nodes:
        fit = node["held_out_correction"].get(variant)
        evaluation = fit and fit["evaluations"].get(segment)
        if evaluation is not None:
            records.append(evaluation[predictor])
    total = sum(record["n"] for record in records)
    if not total:
        return None
    return {
        "n": total,
        **{
            key: sum(record[key] * record["n"] for record in records) / total
            for key in (
                "mae_ms",
                "p95_abs_ms",
                "mean_relative_error",
                "signed_error_ms",
            )
        },
    }


def aggregate(nodes):
    pair_count = sum(node["pairs"]["pair_count"] for node in nodes)
    pair_fields = (
        "slower_has_fewer_generated_tokens",
        "slower_has_more_generated_tokens",
        "same_generated_tokens",
        "generated_difference_over_5pct",
        "active_decode_count_difference_over_1",
    )
    pairs = {"pair_count": pair_count}
    pairs.update(
        {
            field: sum(node["pairs"][field] for node in nodes)
            for field in pair_fields
        }
    )
    held_out = {}
    for variant in (
        "decode_count",
        "generated_tokens",
        "decode_count_and_generated_tokens",
    ):
        held_out[variant] = {}
        for segment in ("held_all", "held_high_waiting", "held_low_waiting"):
            held_out[variant][segment] = {
                predictor: _weighted_metrics(
                    nodes, variant, segment, predictor
                )
                for predictor in ("baseline", "corrected")
            }
    robust_held_out = {}
    for variant in ROBUST_CORRECTION_VARIANTS:
        robust_held_out[variant] = {}
        for segment in CORRECTION_SEGMENTS:
            records_by_predictor = {"baseline": [], "corrected": []}
            selected = {}
            for node in nodes:
                fit = node["robust_held_out_correction"].get(variant)
                if fit is None:
                    continue
                strength = fit["selected_ridge_strength"]
                key = "baseline" if strength is None else str(strength)
                selected[key] = selected.get(key, 0) + 1
                evaluation = fit["evaluations"].get(segment)
                if evaluation is None:
                    continue
                for predictor in records_by_predictor:
                    records_by_predictor[predictor].append(
                        evaluation[predictor]
                    )
            robust_held_out[variant][segment] = {
                predictor: (
                    {
                        "n": sum(record["n"] for record in records),
                        **{
                            key: sum(
                                record[key] * record["n"]
                                for record in records
                            )
                            / sum(record["n"] for record in records)
                            for key in (
                                "mae_ms",
                                "p95_abs_ms",
                                "mean_relative_error",
                                "signed_error_ms",
                            )
                        },
                    }
                    if records
                    else None
                )
                for predictor, records in records_by_predictor.items()
            }
            robust_held_out[variant][segment][
                "selected_ridge_strength_node_counts"
            ] = selected
    static_formula = {}
    for segment in (
        "train_all",
        "held_all",
        "held_high_waiting",
        "held_low_waiting",
    ):
        static_formula[segment] = {}
        for predictor in ("baseline", "augmented"):
            records = []
            for node in nodes:
                ablation = node.get("static_formula_ablation")
                evaluation = ablation and ablation["evaluations"].get(segment)
                if evaluation is not None:
                    records.append(evaluation[predictor])
            total = sum(record["n"] for record in records)
            static_formula[segment][predictor] = (
                {
                    "n": total,
                    **{
                        key: sum(
                            record[key] * record["n"] for record in records
                        )
                        / total
                        for key in (
                            "mae_ms",
                            "p95_abs_ms",
                            "mean_relative_error",
                            "signed_error_ms",
                        )
                    },
                }
                if total
                else None
            )
    all_similar_pairs = {}
    for group in ("all_no_abort", "greedy_disjoint_no_abort"):
        fields = (
            "pair_count",
            "slower_has_fewer_generated_tokens",
            "slower_has_more_generated_tokens",
            "same_generated_tokens",
            "generated_difference_over_5pct",
            "active_decode_count_difference_over_1",
        )
        all_similar_pairs[group] = {
            field: sum(
                node["all_similar_pairs"][group][field] for node in nodes
            )
            for field in fields
        }
    return {
        "pairs": pairs,
        "all_similar_pairs": all_similar_pairs,
        "held_out_correction": held_out,
        "robust_held_out_correction": robust_held_out,
        "static_formula_ablation": static_formula,
    }


def _log_path(log_dir, node):
    path = log_dir / f"lifecycle_{node}.log.gz"
    return path if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("pair_report", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    report = json.loads(args.pair_report.read_text())
    manifest_path = args.log_dir / "manifest.json"
    deployment_by_node = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        deployment_by_node = {
            entry["ip"]: entry.get("deployment", "unknown")
            for entry in manifest.get("nodes", ())
        }
    jobs = []
    for node in report["nodes"]:
        log_path = _log_path(args.log_dir, node["node"])
        if log_path is None:
            continue
        jobs.append(
            (
                node["node"],
                deployment_by_node.get(node["node"], "unknown"),
                log_path,
                node["top_disjoint_pairs"],
                report["definition"],
            )
        )
    if args.workers == 1:
        nodes = [evaluate_node(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            nodes = list(executor.map(evaluate_node, jobs))
    nodes.sort(key=lambda node: node["node"])
    output = {
        "definition": {
            "active_decode_count": (
                "slot-holding requests that emitted first output before admission"
            ),
            "generated_decode_tokens": (
                "first output plus successful decode tokens per active request "
                "completed before admission; preempted requests excluded"
            ),
            "held_blocks": "two-minute event-time blocks where block_index % 4 == 3",
            "correction": (
                "least-squares correction of logged final prediction in log1p space"
            ),
            "static_formula_ablation": (
                "v21 formula fitted on non-held samples, with generated token "
                "count added inside the nonlinear linear workload term"
            ),
            "robust_correction": (
                "outer held blocks use block_index % 4 == 3; ridge is selected "
                "on block_index % 4 == 2, then refitted on all non-held blocks; "
                "the unchanged logged prediction is also a selectable candidate"
            ),
            "pressure_phase": (
                "causal preceding_request_count trend against admissions 15-30 "
                "seconds earlier"
            ),
        },
        "aggregate": aggregate(nodes),
        "by_deployment": {
            deployment: aggregate(
                [node for node in nodes if node["deployment"] == deployment]
            )
            for deployment in sorted({node["deployment"] for node in nodes})
        },
        "nodes": nodes,
    }
    print(json.dumps(output, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
