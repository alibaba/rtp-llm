#!/usr/bin/env python3
"""Reconstruct admission-visible vLLM decode scheduler state.

The aggregate decode count and generated-token count do not preserve either
the active requests' prompt/context lengths or configured output-budget mix.
This tool reconstructs those values from the bounded lifecycle audit and joins
them to NaviSched latency samples by local batch ID. Values computed as
``max_tokens - generated_tokens`` are diagnostics, not true remaining output.
"""

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import math
from pathlib import Path

import numpy as np

from AnalyzeLatencyFeatureNeighbors import FEATURES, read_samples
from AnalyzeVllmRequestLifecycles import read_events
from EvaluateDecodeProgressFeature import _decode_step_prefix, _prefix_at


EXISTING_FEATURES = (
    "batch_count",
    *FEATURES,
    "running_decode_request_count",
    "running_decode_generated_tokens",
)
CAUSAL_CANDIDATES = (
    "running_decode_prompt_tokens",
    "running_decode_context_tokens",
)
RETROSPECTIVE_FIELDS = (
    "active_release_5s_count",
    "active_release_10s_count",
    "active_release_30s_count",
)


def _percentile(values, percentile):
    if not values:
        return 0.0
    return float(np.percentile(values, percentile))


def reconstruct_decode_scheduler_state(events):
    """Return decode cohort state at each batch admission.

    Generated progress is reconstructed only from successful execution steps
    completed before admission. Slot-release timestamps are reported separately
    as retrospective validation and never used as an admission-time feature.
    """
    sessions = {event["session"] for event in events}
    if len(sessions) != 1:
        raise ValueError(f"expected one audit session, found {sorted(sessions)}")
    times, prefix, unit_steps = _decode_step_prefix(events)
    requests = {}
    active_slots = {}
    admissions = {}
    batch_requests = {}
    preempted = set()

    def state_for(request_id):
        return requests.setdefault(
            request_id,
            {
                "admission": None,
                "first_output": None,
                "terminal": None,
                "acquires": [],
                "slots": [],
            },
        )

    ordered = sorted(
        (event for event in events if "sequence" in event),
        key=lambda event: event["sequence"],
    )
    for event in ordered:
        kind = event.get("event")
        request_id = str(event.get("request_id"))
        if kind == "admitted":
            batch_id = int(event["batch_id"])
            admissions.setdefault(batch_id, int(event["monotonic_ns"]))
            batch_requests.setdefault(batch_id, []).append(request_id)
            state_for(request_id)["admission"] = event
        elif kind == "slot_acquire":
            if event.get("resumed") or int(event.get("num_preemptions", 0)) > 0:
                preempted.add(request_id)
            active_slots[request_id] = event
            state_for(request_id)["acquires"].append(event)
        elif kind == "first_output":
            state_for(request_id)["first_output"] = event
        elif kind == "slot_release":
            acquired = active_slots.pop(request_id, None)
            if acquired is not None:
                state_for(request_id)["slots"].append((acquired, event))
        elif kind == "terminal":
            state_for(request_id)["terminal"] = event

    intervals_by_request = {}
    finish_statuses = Counter()
    for request_id, state in requests.items():
        terminal = state["terminal"]
        if terminal is not None:
            finish_statuses[str(terminal.get("status"))] += 1
        admission = state["admission"]
        first_output = state["first_output"]
        if (
            admission is None
            or first_output is None
            or request_id in preempted
        ):
            continue
        slots = list(state["slots"])
        acquired = active_slots.get(request_id)
        if acquired is not None:
            slots.append((acquired, None))
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
            if start_ns >= end_ns:
                continue
            intervals_by_request.setdefault(request_id, []).append(
                {
                    "request_id": request_id,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "offset": offset,
                    "prompt_tokens": max(
                        0, int(admission.get("prompt_tokens", 0))
                    ),
                    "max_tokens": max(0, int(admission.get("max_tokens", 0))),
                }
            )

    active_decode_ids = set()
    admission_snapshots = {}
    for event in ordered:
        kind = event.get("event")
        request_id = str(event.get("request_id"))
        if kind == "admitted":
            batch_id = int(event["batch_id"])
            admission_snapshots.setdefault(
                batch_id,
                (int(event["monotonic_ns"]), tuple(active_decode_ids)),
            )
        elif kind == "first_output" and request_id not in preempted:
            active_decode_ids.add(request_id)
        elif kind == "slot_release":
            active_decode_ids.discard(request_id)

    result = {}
    for batch_id, (admission_ns, active_ids) in sorted(
        admission_snapshots.items(), key=lambda item: item[1][0]
    ):
        progress_prefix = _prefix_at(times, prefix, admission_ns)
        prompt_tokens = []
        generated_tokens = []
        remaining_tokens = []
        release_after_s = []
        active_requests = []
        for request_id in active_ids:
            matching = [
                interval
                for interval in intervals_by_request.get(request_id, ())
                if interval["start_ns"] <= admission_ns < interval["end_ns"]
            ]
            if len(matching) != 1:
                continue
            interval = matching[0]
            generated = max(1.0, progress_prefix + interval["offset"])
            generated = min(float(interval["max_tokens"]), generated)
            remaining = max(0.0, interval["max_tokens"] - generated)
            prompt_tokens.append(interval["prompt_tokens"])
            generated_tokens.append(generated)
            remaining_tokens.append(remaining)
            release_after_s.append(
                (interval["end_ns"] - admission_ns) / 1_000_000_000.0
            )
            active_requests.append(
                {
                    "request_id": interval["request_id"],
                    "prompt_tokens": interval["prompt_tokens"],
                    "max_tokens": interval["max_tokens"],
                    "generated_tokens": generated,
                    "remaining_tokens": remaining,
                    "release_after_s": release_after_s[-1],
                }
            )

        target_ids = batch_requests.get(batch_id, ())
        target_aborted = any(
            (requests[request_id].get("terminal") or {}).get("status")
            == "FINISHED_ABORTED"
            for request_id in target_ids
        )
        first_slot_ns = min(
            (
                int(acquire["monotonic_ns"])
                for request_id in target_ids
                for acquire in requests[request_id]["acquires"]
            ),
            default=None,
        )
        result[batch_id] = {
            "active_decode_count": len(active_requests),
            "generated_decode_tokens": float(sum(generated_tokens)),
            "running_decode_remaining_token_budget": float(
                sum(remaining_tokens)
            ),
            "running_decode_prompt_tokens": float(sum(prompt_tokens)),
            "running_decode_context_tokens": float(
                sum(prompt_tokens) + sum(generated_tokens)
            ),
            "remaining_token_min": float(min(remaining_tokens, default=0.0)),
            "remaining_token_p10": _percentile(remaining_tokens, 10),
            "remaining_token_p25": _percentile(remaining_tokens, 25),
            "remaining_token_p50": _percentile(remaining_tokens, 50),
            "remaining_token_square_sum": float(
                sum(value * value for value in remaining_tokens)
            ),
            **{
                f"remaining_le_{limit}_count": sum(
                    value <= limit for value in remaining_tokens
                )
                for limit in (8, 16, 32, 64)
            },
            **{
                f"active_release_{seconds}s_count": sum(
                    value <= seconds for value in release_after_s
                )
                for seconds in (5, 10, 30)
            },
            "queue_to_first_slot_ms": (
                (first_slot_ns - admission_ns) / 1_000_000.0
                if first_slot_ns is not None
                else None
            ),
            "target_aborted": target_aborted,
            "decode_template_counts": {
                f"prompt_{prompt}_max_{maximum}": count
                for (prompt, maximum), count in Counter(
                    (request["prompt_tokens"], request["max_tokens"])
                    for request in active_requests
                ).items()
            },
            "nearest_release_requests": sorted(
                active_requests, key=lambda request: request["release_after_s"]
            )[:10],
        }

    terminal_count = sum(finish_statuses.values())
    diagnostics = {
        "decode_steps": len(times),
        "unit_decode_steps": unit_steps,
        "unit_decode_step_ratio": unit_steps / len(times) if times else None,
        "preempted_request_count": len(preempted),
        "terminal_count": terminal_count,
        "length_capped_fraction": (
            finish_statuses["FINISHED_LENGTH_CAPPED"] / terminal_count
            if terminal_count
            else None
        ),
        "finish_statuses": dict(finish_statuses),
    }
    return result, diagnostics


def reconstruct_sample_scheduler_state(events, rows, max_callback_lag=8):
    """Reconstruct decode state at the workload snapshot used by each sample.

    The forwarding thread captures this snapshot before request preprocessing
    and EngineCore insertion, so it precedes the scheduler's ``admitted``
    lifecycle event. Its wall time is sample completion minus the measured
    admission-to-prefill latency.

    Async scheduling can complete an execute-model callback before
    ``update_from_output`` commits its token to scheduler request state. The
    recorded generated-token aggregate selects that causal callback lag;
    latency and future release times do not participate in the selection.
    """
    if max_callback_lag < 0:
        raise ValueError("max callback lag must be nonnegative")
    sessions = {event["session"] for event in events}
    if len(sessions) != 1:
        raise ValueError(f"expected one audit session, found {sorted(sessions)}")

    callback_times = sorted(
        int(event["event_time_us"])
        for event in events
        if event.get("event") == "execution_step"
        and event.get("success", False)
        and int(event.get("decode_requests", 0)) > 0
        and int(event.get("decode_tokens", 0))
        == int(event.get("decode_requests", 0))
    )

    requests = {}
    active_slots = {}
    batch_requests = {}
    preempted = set()

    def state_for(request_id):
        return requests.setdefault(
            request_id,
            {
                "admission": None,
                "first_output": None,
                "terminal": None,
                "acquires": [],
                "slots": [],
            },
        )

    ordered = sorted(
        (event for event in events if "sequence" in event),
        key=lambda event: event["sequence"],
    )
    for event in ordered:
        kind = event.get("event")
        request_id = str(event.get("request_id"))
        state = state_for(request_id)
        if kind == "admitted":
            state["admission"] = event
            batch_requests.setdefault(int(event["batch_id"]), []).append(request_id)
        elif kind == "slot_acquire":
            if event.get("resumed") or int(event.get("num_preemptions", 0)) > 0:
                preempted.add(request_id)
            active_slots[request_id] = event
            state["acquires"].append(event)
        elif kind == "first_output":
            state["first_output"] = event
        elif kind == "slot_release":
            acquired = active_slots.pop(request_id, None)
            if acquired is not None:
                state["slots"].append((acquired, event))
        elif kind == "terminal":
            state["terminal"] = event

    intervals = []
    for request_id, state in requests.items():
        admission = state["admission"]
        first_output = state["first_output"]
        if admission is None or first_output is None or request_id in preempted:
            continue
        slots = list(state["slots"])
        acquired = active_slots.get(request_id)
        if acquired is not None:
            slots.append((acquired, None))
        first_us = int(first_output["event_time_us"])
        first_tokens = max(1, int(first_output.get("output_tokens", 1)))
        offset = first_tokens - bisect_right(callback_times, first_us)
        for acquired, released in slots:
            start_us = max(first_us, int(acquired["event_time_us"]))
            end_us = (
                int(released["event_time_us"])
                if released is not None
                else math.inf
            )
            if start_us >= end_us:
                continue
            intervals.append(
                {
                    "request_id": request_id,
                    "start_us": start_us,
                    "end_us": end_us,
                    "offset": offset,
                    "prompt_tokens": max(
                        0, int(admission.get("prompt_tokens", 0))
                    ),
                    "max_tokens": max(0, int(admission.get("max_tokens", 0))),
                }
            )
    intervals.sort(key=lambda interval: interval["start_us"])

    queries = sorted(
        (
            float(row["event_time_us"] - row["target_ms"] * 1000.0),
            row,
        )
        for row in rows
    )
    active = {}
    releases = []
    interval_index = 0
    result = {}
    count_mismatches = []
    progress_mismatches = []
    selected_lags = Counter()

    for snapshot_us, row in queries:
        while (
            interval_index < len(intervals)
            and intervals[interval_index]["start_us"] <= snapshot_us
        ):
            interval = intervals[interval_index]
            active[interval["request_id"]] = interval
            heapq.heappush(
                releases, (interval["end_us"], interval["request_id"])
            )
            interval_index += 1
        while releases and releases[0][0] <= snapshot_us:
            end_us, request_id = heapq.heappop(releases)
            current = active.get(request_id)
            if current is not None and current["end_us"] == end_us:
                del active[request_id]

        expected_count = int(row["running_decode_request_count"])
        if len(active) != expected_count:
            count_mismatches.append(
                {
                    "batch_id": int(row["batch_id"]),
                    "expected": expected_count,
                    "reconstructed": len(active),
                }
            )
            continue

        callback_prefix = bisect_right(callback_times, snapshot_us)
        expected_generated = int(row["running_decode_generated_tokens"])
        best = None
        for lag in range(max_callback_lag + 1):
            committed_prefix = callback_prefix - lag
            generated = [
                min(
                    interval["max_tokens"],
                    max(1, committed_prefix + interval["offset"]),
                )
                for interval in active.values()
            ]
            difference = abs(sum(generated) - expected_generated)
            candidate = (difference, lag, committed_prefix, generated)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        difference, lag, _, generated_tokens = best
        if difference != 0:
            progress_mismatches.append(
                {
                    "batch_id": int(row["batch_id"]),
                    "expected": expected_generated,
                    "nearest": int(sum(generated_tokens)),
                    "callback_prefix": callback_prefix,
                }
            )
            continue
        selected_lags[lag] += 1

        active_requests = []
        prompt_tokens = []
        remaining_tokens = []
        release_after_s = []
        for interval, generated in zip(active.values(), generated_tokens):
            remaining = max(0.0, interval["max_tokens"] - generated)
            release_s = (interval["end_us"] - snapshot_us) / 1_000_000.0
            prompt_tokens.append(interval["prompt_tokens"])
            remaining_tokens.append(remaining)
            release_after_s.append(release_s)
            active_requests.append(
                {
                    "request_id": interval["request_id"],
                    "prompt_tokens": interval["prompt_tokens"],
                    "max_tokens": interval["max_tokens"],
                    "generated_tokens": float(generated),
                    "remaining_tokens": float(remaining),
                    "release_after_s": release_s,
                }
            )

        batch_id = int(row["batch_id"])
        target_ids = batch_requests.get(batch_id, ())
        target_aborted = any(
            (requests[request_id].get("terminal") or {}).get("status")
            == "FINISHED_ABORTED"
            for request_id in target_ids
        )
        first_slot_us = min(
            (
                int(acquire["event_time_us"])
                for request_id in target_ids
                for acquire in requests[request_id]["acquires"]
            ),
            default=None,
        )
        result[batch_id] = {
            "active_decode_count": len(active_requests),
            "generated_decode_tokens": float(sum(generated_tokens)),
            "running_decode_remaining_token_budget": float(
                sum(remaining_tokens)
            ),
            "running_decode_prompt_tokens": float(sum(prompt_tokens)),
            "running_decode_context_tokens": float(
                sum(prompt_tokens) + sum(generated_tokens)
            ),
            "remaining_token_min": float(min(remaining_tokens, default=0.0)),
            "remaining_token_p10": _percentile(remaining_tokens, 10),
            "remaining_token_p25": _percentile(remaining_tokens, 25),
            "remaining_token_p50": _percentile(remaining_tokens, 50),
            "remaining_token_square_sum": float(
                sum(value * value for value in remaining_tokens)
            ),
            **{
                f"remaining_le_{limit}_count": sum(
                    value <= limit for value in remaining_tokens
                )
                for limit in (8, 16, 32, 64)
            },
            **{
                f"active_release_{seconds}s_count": sum(
                    value <= seconds for value in release_after_s
                )
                for seconds in (5, 10, 30)
            },
            "queue_to_first_slot_ms": (
                (first_slot_us - snapshot_us) / 1000.0
                if first_slot_us is not None
                else None
            ),
            "target_aborted": target_aborted,
            "decode_template_counts": {
                f"prompt_{prompt}_max_{maximum}": count
                for (prompt, maximum), count in Counter(
                    (request["prompt_tokens"], request["max_tokens"])
                    for request in active_requests
                ).items()
            },
            "nearest_release_requests": sorted(
                active_requests, key=lambda request: request["release_after_s"]
            )[:10],
        }

    diagnostics = {
        "sample_snapshot_count": len(rows),
        "reconstructed_snapshot_count": len(result),
        "count_mismatch_count": len(count_mismatches),
        "progress_mismatch_count": len(progress_mismatches),
        "selected_callback_lags": {
            str(lag): count for lag, count in sorted(selected_lags.items())
        },
        "count_mismatch_examples": count_mismatches[:10],
        "progress_mismatch_examples": progress_mismatches[:10],
    }
    return result, diagnostics


def _relative_difference(left, right):
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _correlation(left, right):
    if len(left) < 3 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def join_samples(rows, states):
    joined = []
    for row in rows:
        state = states.get(row["batch_id"])
        if state is None or state["target_aborted"]:
            continue
        copy = dict(row)
        copy.update(
            {
                key: state[key]
                for key in (*CAUSAL_CANDIDATES, *RETROSPECTIVE_FIELDS)
            }
        )
        copy["queue_to_first_slot_ms"] = state["queue_to_first_slot_ms"]
        copy["decode_template_counts"] = state["decode_template_counts"]
        copy["nearest_release_requests"] = state["nearest_release_requests"]
        joined.append(copy)
    return joined


def validate_reconstruction(rows, states):
    differences = {
        "decode_count": [],
        "generated_tokens": [],
    }
    for row in rows:
        state = states.get(row["batch_id"])
        if state is None:
            continue
        if "running_decode_request_count" in row:
            differences["decode_count"].append(
                state["active_decode_count"]
                - row["running_decode_request_count"]
            )
        if "running_decode_generated_tokens" in row:
            differences["generated_tokens"].append(
                state["generated_decode_tokens"]
                - row["running_decode_generated_tokens"]
            )
    return {
        name: {
            "n": len(values),
            "exact_fraction": (
                sum(value == 0 for value in values) / len(values)
                if values
                else None
            ),
            "mean_abs_difference": (
                float(np.mean(np.abs(values))) if values else None
            ),
            "max_abs_difference": (
                float(np.max(np.abs(values))) if values else None
            ),
        }
        for name, values in differences.items()
    }


def _candidate_residual_correlations(rows):
    if not rows:
        return {}
    residual = np.asarray(
        [row["target_ms"] - row.get("prediction_before_ms", 0.0) for row in rows]
    )
    return {
        name: _correlation(
            np.asarray([row[name] for row in rows], dtype=float), residual
        )
        for name in (*CAUSAL_CANDIDATES, *RETROSPECTIVE_FIELDS)
    }


def find_similar_pairs(
    rows,
    min_separation_s=60.0,
    target_gap_ms=2000.0,
    distance_limit=0.5,
    relative_limit=0.1,
):
    if len(rows) < 2:
        return []
    existing = np.asarray(
        [[row[name] for name in EXISTING_FEATURES] for row in rows], dtype=float
    )
    scale = np.std(existing, axis=0)
    scale[scale == 0.0] = 1.0
    ready = np.asarray(
        [row["event_time_us"] - row["target_ms"] * 1000.0 for row in rows]
    )
    finish = np.asarray([row["event_time_us"] for row in rows])
    target = np.asarray([row["target_ms"] for row in rows])
    minimum_separation_us = min_separation_s * 1_000_000.0
    pairs = []
    for left in range(len(rows) - 1):
        right = np.arange(left + 1, len(rows))
        relative = np.max(
            np.abs(existing[right] - existing[left])
            / np.maximum(
                1.0,
                np.maximum(np.abs(existing[right]), np.abs(existing[left])),
            ),
            axis=1,
        )
        distance = np.linalg.norm(
            (existing[right] - existing[left]) / scale, axis=1
        )
        selected = right[
            (np.abs(ready[right] - ready[left]) >= minimum_separation_us)
            & (np.abs(finish[right] - finish[left]) >= minimum_separation_us)
            & (np.abs(target[right] - target[left]) >= target_gap_ms)
            & (relative <= relative_limit)
            & (distance <= distance_limit)
        ]
        for other in selected:
            slow, fast = (
                (rows[left], rows[other])
                if rows[left]["target_ms"] > rows[other]["target_ms"]
                else (rows[other], rows[left])
            )
            pairs.append(
                {
                    "batches": [left["batch_id"] for left in (slow, fast)],
                    "target_ms": [left["target_ms"] for left in (slow, fast)],
                    "target_gap_ms": slow["target_ms"] - fast["target_ms"],
                    "slow_minus_fast": {
                        name: slow[name] - fast[name]
                        for name in (*CAUSAL_CANDIDATES, *RETROSPECTIVE_FIELDS)
                    },
                    "slow": slow,
                    "fast": fast,
                }
            )
    return pairs


def summarize_pairs(pairs):
    result = {"pair_count": len(pairs)}
    for name in (*CAUSAL_CANDIDATES, *RETROSPECTIVE_FIELDS):
        values = [pair["slow_minus_fast"][name] for pair in pairs]
        result[name] = {
            "slow_higher_fraction": (
                sum(value > 0 for value in values) / len(values)
                if values
                else None
            ),
            "slow_lower_fraction": (
                sum(value < 0 for value in values) / len(values)
                if values
                else None
            ),
            "median_slow_minus_fast": (
                float(np.median(values)) if values else None
            ),
            "correlation_with_target_gap": _correlation(
                np.asarray(values),
                np.asarray([pair["target_gap_ms"] for pair in pairs]),
            ),
        }
    result["examples"] = [
        {
            "batches": pair["batches"],
            "target_ms": pair["target_ms"],
            "target_gap_ms": pair["target_gap_ms"],
            "slow_minus_fast": pair["slow_minus_fast"],
            "slow_nearest_releases": sorted(
                pair["slow"]["nearest_release_requests"],
                key=lambda request: request["release_after_s"],
            )[:5],
            "fast_nearest_releases": sorted(
                pair["fast"]["nearest_release_requests"],
                key=lambda request: request["release_after_s"],
            )[:5],
            "slow_decode_template_counts": pair["slow"][
                "decode_template_counts"
            ],
            "fast_decode_template_counts": pair["fast"][
                "decode_template_counts"
            ],
        }
        for pair in sorted(pairs, key=lambda pair: pair["target_gap_ms"], reverse=True)[
            :5
        ]
    ]
    return result


def analyze_node(path, deployment):
    events, malformed = read_events([path])
    rows = read_samples(path)
    states, snapshot_diagnostics = reconstruct_sample_scheduler_state(events, rows)
    _, lifecycle_diagnostics = reconstruct_decode_scheduler_state(events)
    diagnostics = {**lifecycle_diagnostics, **snapshot_diagnostics}
    joined = join_samples(rows, states)
    max_num_seqs = max(
        (int(event.get("max_num_seqs", 0)) for event in events), default=0
    )
    saturated = [
        row
        for row in joined
        if row["running_decode_request_count"] >= 0.8 * max_num_seqs
    ]
    pairs = find_similar_pairs(saturated)
    return {
        "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
        "deployment": deployment,
        "malformed_audit_lines": malformed,
        "diagnostics": diagnostics,
        "sample_count": len(rows),
        "joined_sample_count": len(joined),
        "saturated_sample_count": len(saturated),
        "reconstruction": validate_reconstruction(rows, states),
        "saturated_residual_correlations": _candidate_residual_correlations(
            saturated
        ),
        "similar_pairs": summarize_pairs(pairs),
    }


def _analyze_job(job):
    path, deployment = job
    try:
        return analyze_node(path, deployment), None
    except ValueError as error:
        if "expected one audit session, found []" not in str(error):
            raise
        return None, {
            "node": path.name.removeprefix("lifecycle_").removesuffix(".log.gz"),
            "deployment": deployment,
            "reason": "no lifecycle audit events",
        }


def _aggregate_pair_summaries(nodes):
    # Pair examples remain node-local; aggregate only auditable directions.
    result = {"pair_count": sum(node["similar_pairs"]["pair_count"] for node in nodes)}
    for name in (*CAUSAL_CANDIDATES, *RETROSPECTIVE_FIELDS):
        higher = lower = total = 0
        medians = []
        for node in nodes:
            summary = node["similar_pairs"][name]
            count = node["similar_pairs"]["pair_count"]
            if count:
                higher += round(summary["slow_higher_fraction"] * count)
                lower += round(summary["slow_lower_fraction"] * count)
                medians.append(summary["median_slow_minus_fast"])
                total += count
        result[name] = {
            "slow_higher_fraction": higher / total if total else None,
            "slow_lower_fraction": lower / total if total else None,
            "median_node_median_slow_minus_fast": (
                float(np.median(medians)) if medians else None
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.log_dir / "manifest.json").read_text())
    deployment_by_ip = {
        node["ip"]: node.get("deployment", "unknown")
        for node in manifest.get("nodes", ())
    }
    jobs = [
        (path, deployment_by_ip.get(path.name[10:-7].replace("_", "."), "unknown"))
        for path in sorted(args.log_dir.glob("lifecycle_*.log.gz"))
    ]
    if args.workers == 1:
        analyzed = [_analyze_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            analyzed = list(executor.map(_analyze_job, jobs))
    nodes = [node for node, _ in analyzed if node is not None]
    skipped_nodes = [skipped for _, skipped in analyzed if skipped is not None]
    nodes.sort(key=lambda node: node["node"])
    deployments = sorted({node["deployment"] for node in nodes})
    output = {
        "definitions": {
            "causal_candidates": list(CAUSAL_CANDIDATES),
            "retrospective_validation_only": list(RETROSPECTIVE_FIELDS),
            "similar_pair_existing_features": list(EXISTING_FEATURES),
            "similar_pair_limits": {
                "minimum_time_separation_s": 60,
                "minimum_target_gap_ms": 2000,
                "maximum_relative_component_difference": 0.1,
                "maximum_standardized_l2_distance": 0.5,
            },
        },
        "by_deployment": {
            deployment: _aggregate_pair_summaries(
                [node for node in nodes if node["deployment"] == deployment]
            )
            for deployment in deployments
        },
        "skipped_nodes": skipped_nodes,
        "nodes": nodes,
    }
    print(json.dumps(output, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
