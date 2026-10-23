"""Independent, dependency-free validation of client and Mock Engine traffic evidence."""

import hashlib
import json
import math
from collections import Counter, defaultdict

OVERLOAD_ERROR = (
    "admission capacity is temporarily exhausted; "
    "trigger=Prefill queue overload: HIT_RATE cache eviction"
)
DEADLINE_ERROR = (
    "admission capacity is temporarily exhausted; "
    "trigger=PREFILL placement unavailable"
)
DEADLINE_ERRORS = {
    DEADLINE_ERROR,
    "admission capacity is temporarily exhausted; trigger=Prefill collection window",
    "admission capacity is temporarily exhausted; trigger=Prefill batch capacity exhausted",
    "admission capacity is temporarily exhausted; trigger=waiting for Prefill decision",
}
AUXILIARY_STAGES = {"warmup", "affinity", "probe", "calibration"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, minimum=0):
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= minimum,
        f"invalid {name}: {value!r}",
    )
    return value


def _integer(value, name, minimum=0):
    _number(value, name, minimum)
    _require(isinstance(value, int), f"noninteger {name}: {value!r}")
    return value


def _percentile(values, quantile):
    """Nearest-rank percentile; sample counts accompany every reported percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def expected_rejection(row, hard_wait_s):
    """Classification only: audit_run also requires absence of all Engine evidence."""
    return row.get("code") == 8431 and (
        row.get("error") == OVERLOAD_ERROR
        or (
            row.get("error") in DEADLINE_ERRORS
            and row["ended_s"] - row["started_s"] >= hard_wait_s - 0.05
        )
    )


def validate_trace(requests, expected_trace, stages):
    """Require the entire pre-generated arrival inventory, including rejected work."""
    planned = [
        row
        for name in ("warmup", "affinity", "calibration", "traffic")
        for row in expected_trace[name]
    ]
    expected = {row["request_id"]: row for row in planned}
    actual = {row["request_id"]: row for row in requests}
    _require(len(expected) == len(planned), "duplicate planned request IDs")
    _require(len(actual) == len(requests), "duplicate actual request IDs")
    _require(
        set(expected) == set(actual), "planned and actual request inventories differ"
    )
    _require(stages, "trace has no measurement phases")
    origin = stages[0]["start_s"]
    traffic_ids = {row["request_id"] for row in expected_trace["traffic"]}
    for rid, item in expected.items():
        for field in ("stage", "session", "input_tokens", "keys"):
            _require(
                actual[rid][field] == item[field],
                f"request {rid} changed planned {field}",
            )
        if rid in traffic_ids:
            offset = _number(item["offset_s"], f"request {rid} planned offset")
            _require(
                abs(actual[rid]["planned_s"] - origin - offset) < 1e-6,
                f"request {rid} changed planned arrival",
            )
    return {
        "planned_requests": len(planned),
        "planned_traffic_requests": len(traffic_ids),
    }


def validate_environment(stages, topology, final_snapshot):
    """Verify observed topology transitions and absence of remaining request owners."""
    confirmations = sorted(
        (row for row in topology if "expected" in row), key=lambda row: row["epoch_ms"]
    )
    _require(stages and confirmations, "missing topology evidence")
    for confirmation in confirmations:
        workers = confirmation["master"]["worker_summary"]["PREFILL"]
        _require(
            confirmation["master"].get("success") is True
            and workers["alive"] == workers["discovered"] == confirmation["expected"],
            "Master topology acknowledgement contradicts expected worker count",
        )
    convergence = []
    previous_count = None
    for stage in stages:
        before_end = [
            row for row in confirmations if row["epoch_ms"] < stage["end_epoch_ms"]
        ]
        target = stage["worker_count"]
        _require(
            before_end and before_end[-1]["expected"] == target,
            f"phase {stage['stage']} did not converge to {target} workers",
        )
        matching = [row for row in before_end if row["expected"] == target]
        if previous_count is not None and target != previous_count:
            matching = [
                row for row in matching if row["epoch_ms"] >= stage["start_epoch_ms"]
            ]
            _require(
                matching,
                f"phase {stage['stage']} reused an old topology acknowledgement",
            )
        convergence.append(
            {
                "stage": stage["stage"],
                "workers": target,
                "convergence_s": max(
                    0, (matching[0]["epoch_ms"] - stage["start_epoch_ms"]) / 1000
                ),
            }
        )
        previous_count = target
    removals = [row["remove"] for row in topology if "remove" in row]
    for removal in removals:
        _require(
            removal.get("mode") == "graceful" and removal.get("drained") is True,
            "worker removal was not gracefully drained",
        )
        remaining = removal["remaining_work"]
        _require(
            remaining and all(value == 0 for value in remaining.values()),
            "removed worker still owns work",
        )
        _require(
            removal["admission"]["rejected_rpcs_total"] == 0,
            "worker rejected RPCs during coordinated removal",
        )
    workers = final_snapshot["engines"]
    _require(workers, "empty final worker snapshot")
    _require(
        sum(worker["role"].lower() == "prefill" for worker in workers)
        == stages[-1]["worker_count"],
        "final Prefill worker count differs from final phase",
    )
    owner_fields = (
        "running",
        "scheduler_running",
        "waiting",
        "active_decode_requests",
        "response_buffers",
        "prefill_contexts",
        "decode_waiting_for_kv",
        "active_kv_tokens",
    )
    for worker in workers:
        for field in owner_fields:
            _require(
                worker.get(field) == 0,
                f"worker {worker['name']} has missing/nonzero {field}",
            )
        if worker["role"].lower() == "prefill":
            _require(
                worker.get("prefill_waiting_batches") == 0,
                f"worker {worker['name']} has remaining Prefill batches",
            )
        _require(
            not worker.get("leak_detected", False),
            f"worker {worker['name']} reported a leak",
        )
    return {
        "topology": convergence,
        "gracefully_removed_workers": len(removals),
        "drained_final_workers": len(workers),
    }


def _validate_request(row, hard_wait_s):
    rid = _integer(row["request_id"], "request_id")
    _integer(row["input_tokens"], f"request {rid} input_tokens", 1)
    _require(type(row["success"]) is bool, f"request {rid} lacks terminal outcome")
    for key in ("planned_s", "started_s", "ended_s", "planned_epoch_ms"):
        _number(row[key], f"request {rid} {key}")
    _require(
        row["planned_s"] - 0.001 <= row["started_s"] <= row["ended_s"],
        f"request {rid} invalid client time order",
    )
    lag = row["started_s"] - row["planned_s"]
    _number(row["pacing_lag_s"], f"request {rid} pacing_lag_s", -0.001)
    _require(
        abs(lag - row["pacing_lag_s"]) < 1e-6,
        f"request {rid} inconsistent pacing lag",
    )
    _require(lag < 0.25, f"request {rid} load generator fell behind")
    _validate_launch_times(row, rid)
    _require(
        isinstance(row["keys"], list)
        and row["keys"]
        and all(type(key) is int for key in row["keys"]),
        f"request {rid} missing cache-key input evidence",
    )
    _require(isinstance(row["session"], str), f"request {rid} invalid session")
    if row["success"]:
        _require(row.get("code") == 200, f"request {rid} success without admission")
        _require(
            not row.get("error") and not row.get("stream_error_code"),
            f"request {rid} success with stream error",
        )
        _number(row["ttft_s"], f"request {rid} ttft_s")
        first = _number(row["first_received_s"], f"request {rid} first_received_s")
        _require(
            row["started_s"] <= first <= row["ended_s"]
            and abs(first - row["started_s"] - row["ttft_s"]) < 1e-6,
            f"request {rid} inconsistent first-token time",
        )
    else:
        _require(
            expected_rejection(row, hard_wait_s),
            f"request {rid} unexpected scheduling/transport failure",
        )


def _validate_launch_times(row, rid):
    schedule_fields = (
        "schedule_launched_s",
        "schedule_launch_lag_s",
        "schedule_completed_s",
    )
    if any(field in row for field in schedule_fields):
        for field in schedule_fields:
            _number(row[field], f"request {rid} {field}")
        _require(
            row["started_s"]
            <= row["schedule_launched_s"]
            <= row["schedule_completed_s"]
            <= row["ended_s"],
            f"request {rid} inconsistent Schedule timestamps",
        )
        _require(
            abs(
                row["schedule_launched_s"]
                - row["planned_s"]
                - row["schedule_launch_lag_s"]
            )
            < 1e-6,
            f"request {rid} inconsistent Schedule launch lag",
        )
        _require(
            row["schedule_launch_lag_s"] < 0.25,
            f"request {rid} Schedule launch fell behind",
        )
    stream_fields = (
        "stream_started_s",
        "stream_launch_completed_s",
        "stream_launch_lag_s",
    )
    if any(field in row for field in stream_fields):
        for field in stream_fields:
            _number(row[field], f"request {rid} {field}")
        _require(
            row["schedule_completed_s"]
            <= row["stream_started_s"]
            <= row["stream_launch_completed_s"]
            and row["stream_started_s"] <= row["ended_s"],
            f"request {rid} inconsistent stream launch timestamps",
        )
        _require(
            abs(
                row["stream_launch_completed_s"]
                - row["schedule_completed_s"]
                - row["stream_launch_lag_s"]
            )
            < 1e-6,
            f"request {rid} inconsistent stream launch lag",
        )
        _require(
            row["stream_launch_lag_s"] < 0.25,
            f"request {rid} stream launch fell behind",
        )


def _validate_prefill(event, request, dispatch):
    rid = event["rid"]
    _require(event.get("cancelled") is False, f"request {rid} cancelled in Engine")
    _require(
        event["input_len"] == request["input_tokens"],
        f"request {rid} input token mismatch",
    )
    hit = _integer(event["cache_hit_tokens"], f"request {rid} cache_hit_tokens")
    _require(hit <= request["input_tokens"], f"request {rid} impossible cache hit")
    _integer(event["batch_size"], f"request {rid} batch_size", 1)
    _integer(event["batch_id"], f"request {rid} batch_id", -1)
    _require(
        (event["batch_id"] == -1) == (dispatch == "NON_BATCH"),
        f"request {rid} dispatch mode contradicts Engine batch_id",
    )
    for key in ("engine_arrival_ms", "prefill_start_ms", "prefill_done_ms", "exec_ms"):
        _number(event[key], f"request {rid} {key}", 1)
    _require(
        event["engine_arrival_ms"] - 1 <= event["prefill_start_ms"]
        and event["prefill_start_ms"] - 1 <= event["prefill_done_ms"],
        f"request {rid} invalid Engine time order",
    )
    _require(bool(event.get("engine_name")), f"request {rid} missing worker identity")
    routed_worker = request.get("worker")
    executed_worker = event.get("engine_address")
    _require(
        isinstance(routed_worker, str)
        and bool(routed_worker.strip())
        and isinstance(executed_worker, str)
        and bool(executed_worker.strip()),
        f"request {rid} missing or invalid worker address evidence",
    )
    _require(
        routed_worker == executed_worker,
        f"request {rid} executed on a different routing worker",
    )
    # Local client and Mock share the epoch clock; their monotonic observations
    # are anchored by planned_epoch_ms. Allow only millisecond stamp rounding.
    client_start_ms = (
        request["planned_epoch_ms"]
        + (request["started_s"] - request["planned_s"]) * 1000
    )
    client_first_ms = (
        request["planned_epoch_ms"]
        + (request["first_received_s"] - request["planned_s"]) * 1000
    )
    _require(
        event["engine_arrival_ms"] >= client_start_ms - 2
        and event["prefill_done_ms"] <= client_first_ms + 2,
        f"request {rid} Engine execution falls outside client request interval",
    )


def _validate_cache_eviction(event, known_engines, run_start_ms, run_end_ms):
    """Validate the pinned Mock's non-request cache lifecycle event schema."""
    _require(
        set(event)
        == {
            "event",
            "engine_name",
            "engine_incarnation",
            "engine_address",
            "timestamp_ms",
            "leaf_key",
            "chain_keys",
            "blocks_freed",
            "reason",
        },
        "evict_chain has missing or unexpected fields",
    )
    identity = tuple(
        event[key] for key in ("engine_name", "engine_incarnation", "engine_address")
    )
    _require(
        all(isinstance(value, str) and value.strip() for value in identity),
        "evict_chain has invalid Engine identity",
    )
    _require(identity in known_engines, "evict_chain has unknown Engine identity")
    stamp = _integer(event["timestamp_ms"], "evict_chain timestamp_ms", 1)
    _require(
        run_start_ms - 2 <= stamp <= run_end_ms + 2,
        "evict_chain timestamp falls outside client run interval",
    )
    _require(
        isinstance(event["reason"], str)
        and event["reason"]
        in {"admission", "growth", "completion", "insertion", "retention"},
        "evict_chain has unknown reason",
    )
    chain = event["chain_keys"]
    _require(
        isinstance(chain, list) and chain, "evict_chain has empty or invalid chain"
    )
    for key in [event["leaf_key"], *chain]:
        _integer(key, "evict_chain cache key", -(1 << 63))
        _require(key < (1 << 63), "evict_chain cache key exceeds signed long")
    _require(
        chain[0] == event["leaf_key"] and len(set(chain)) == len(chain),
        "evict_chain has inconsistent leaf or duplicate chain keys",
    )
    freed = _integer(event["blocks_freed"], "evict_chain blocks_freed", 1)
    # Prefix-tree traversal may include one final pinned ancestor, which is
    # inspected but not removed. Flat LRU and wholly unpinned chains free all.
    _require(
        len(chain) - 1 <= freed <= len(chain),
        "evict_chain has impossible freed blocks",
    )


def audit_run(requests, stages, events, *, dispatch, hard_wait_s=6.0):
    """Validate one fully drained run and compute two distinct throughput measures.

    Cohort TPM uses all successful arrivals in one phase and includes its drain.
    Completion TPM uses Prefill terminals inside the fixed epoch-ms phase window,
    including any earlier arrivals that finish there. Neither is a GPU benchmark.
    Epoch timestamps select fixed-window completions; monotonic timestamps measure
    client latency and pacing. All evidence is consumed before any report returns.
    """
    _require(dispatch in {"NON_BATCH", "BATCH"}, "invalid dispatch mode")
    _number(hard_wait_s, "hard_wait_s", 0.05)
    _require(requests and stages, "empty traffic evidence")
    by_id = {}
    for row in requests:
        _validate_request(row, hard_wait_s)
        if row["success"]:
            _require(
                row.get("enqueued_by_master") is (dispatch == "BATCH"),
                f"request {row['request_id']} response contradicts dispatch mode",
            )
        rid = row["request_id"]
        _require(rid not in by_id, f"duplicate request {rid}")
        by_id[rid] = row

    by_event_id = defaultdict(list)
    cache_events = []
    for event in events:
        _require(isinstance(event, dict), "invalid Engine event record")
        kind = event.get("event")
        if kind == "evict_chain":
            cache_events.append(event)
            continue
        rid = event.get("rid")
        _require(rid in by_id, f"Engine event has unknown request {rid!r}")
        _require(by_id[rid]["success"], f"rejected request {rid} has Engine evidence")
        _require(
            isinstance(kind, str) and kind in {"prefill_done", "decode_done"},
            f"unknown Engine event kind {kind!r}",
        )
        by_event_id[rid].append(event)
    known_engines = {
        tuple(
            event.get(key)
            for key in ("engine_name", "engine_incarnation", "engine_address")
        )
        for terminals in by_event_id.values()
        for event in terminals
        if all(
            isinstance(event.get(key), str)
            for key in ("engine_name", "engine_incarnation", "engine_address")
        )
    }
    run_start_ms = min(
        row["planned_epoch_ms"] + (row["started_s"] - row["planned_s"]) * 1000
        for row in requests
    )
    run_end_ms = max(
        row["planned_epoch_ms"] + (row["ended_s"] - row["planned_s"]) * 1000
        for row in requests
    )
    for event in cache_events:
        _validate_cache_eviction(event, known_engines, run_start_ms, run_end_ms)
    prefill = {}
    execution_groups = defaultdict(list)
    for rid, row in by_id.items():
        terminals = by_event_id[rid]
        if not row["success"]:
            _require(not terminals, f"rejected request {rid} has Engine evidence")
            continue
        for kind in ("prefill_done", "decode_done"):
            matches = [event for event in terminals if event.get("event") == kind]
            _require(len(matches) == 1, f"request {rid} needs exactly one {kind}")
            _require(
                matches[0].get("cancelled") is False,
                f"request {rid} has cancelled {kind}",
            )
        event = next(event for event in terminals if event["event"] == "prefill_done")
        _validate_prefill(event, row, dispatch)
        prefill[rid] = event
        # batch_id is a Master delivery identity. Engine regrouping can combine
        # several delivery IDs, while NON_BATCH always uses -1. Execution stamps
        # identify one actual service interval, whose exec_ms is counted once.
        # Completion has one shared timestamp; start is stamped per member.
        identity = (
            event["engine_name"],
            event.get("engine_incarnation", ""),
            event["prefill_done_ms"],
        )
        execution_groups[identity].append(event)
    for group in execution_groups.values():
        _require(
            all(event["batch_size"] == len(group) for event in group),
            "incomplete or contradictory Prefill execution batch",
        )
        _require(
            len({event["exec_ms"] for event in group}) == 1,
            "inconsistent Prefill execution duration within batch",
        )
    multi_batches = sum(len(group) > 1 for group in execution_groups.values())
    if dispatch == "BATCH":
        _require(multi_batches > 0, "BATCH run did not execute a multi-request batch")

    names = [stage["stage"] for stage in stages]
    _require(len(names) == len(set(names)), "duplicate phase name")
    _require(
        all(row["stage"] in set(names) | AUXILIARY_STAGES for row in requests),
        "request belongs to an unreported phase",
    )
    report = []
    previous_end = None
    for stage in stages:
        name = stage["stage"]
        for key in (
            "start_s",
            "drained_s",
            "start_epoch_ms",
            "end_epoch_ms",
            "seconds",
        ):
            _number(stage[key], f"phase {name} {key}")
        seconds = stage["seconds"]
        _require(seconds > 0, f"phase {name} empty measurement window")
        _require(
            abs(stage["end_epoch_ms"] - stage["start_epoch_ms"] - seconds * 1000) < 1,
            f"phase {name} inconsistent fixed window",
        )
        if previous_end is not None:
            _require(
                stage["start_epoch_ms"] >= previous_end, "overlapping phase windows"
            )
        previous_end = stage["end_epoch_ms"]
        _require(
            type(stage["allow_rejections"]) is bool, "invalid rejection expectation"
        )
        _integer(stage["worker_count"], f"phase {name} worker_count", 1)
        rows = [row for row in requests if row["stage"] == name]
        _require(rows, f"phase {name} has no arrivals")
        successes = [row for row in rows if row["success"]]
        _require(successes, f"phase {name} has zero successful requests")
        rejects = len(rows) - len(successes)
        _require(
            stage["allow_rejections"] or rejects == 0, f"phase {name} rejected traffic"
        )
        minimum_success = _number(
            stage.get("min_success_fraction", 0), "min_success_fraction"
        )
        _require(minimum_success <= 1, "min_success_fraction exceeds one")
        _require(
            len(successes) / len(rows) >= minimum_success,
            f"phase {name} misses minimum success fraction {minimum_success}",
        )
        tail = []
        if stage.get("require_tail_zero_rejections", False):
            settle = _number(stage["settle_seconds"], f"phase {name} settle_seconds")
            _require(settle < seconds, f"phase {name} has no recovery tail window")
            tail = [
                row for row in rows if row["planned_s"] - stage["start_s"] >= settle
            ]
            _require(tail, f"phase {name} has no recovery tail arrivals")
            _require(
                all(row["success"] for row in tail),
                f"phase {name} rejects recovery tail",
            )
        for row in rows:
            _require(
                stage["start_s"] - 0.001
                <= row["planned_s"]
                < stage["start_s"] + seconds
                and stage["start_epoch_ms"] - 1
                <= row["planned_epoch_ms"]
                < stage["end_epoch_ms"],
                f"request {row['request_id']} planned outside phase",
            )
            _require(
                abs(
                    row["planned_epoch_ms"]
                    - stage["start_epoch_ms"]
                    - (row["planned_s"] - stage["start_s"]) * 1000
                )
                < 2,
                f"request {row['request_id']} inconsistent planned clocks",
            )
            _require(row["ended_s"] <= stage["drained_s"], f"phase {name} not drained")
        elapsed = max(seconds, stage["drained_s"] - stage["start_s"])
        input_tokens = sum(row["input_tokens"] for row in successes)
        hit_tokens = sum(
            prefill[row["request_id"]]["cache_hit_tokens"] for row in successes
        )
        completed = [
            event
            for event in prefill.values()
            if stage["start_epoch_ms"]
            <= event["prefill_done_ms"]
            < stage["end_epoch_ms"]
        ]
        completion_input = sum(event["input_len"] for event in completed)
        completion_hit = sum(event["cache_hit_tokens"] for event in completed)
        busy_ms = Counter()
        for group in execution_groups.values():
            event = group[0]
            if (
                stage["start_epoch_ms"]
                <= event["prefill_done_ms"]
                < stage["end_epoch_ms"]
            ):
                busy_ms[event["engine_name"]] += event["exec_ms"]
        latencies = [row["ttft_s"] for row in successes]
        report.append(
            {
                **stage,
                "arrivals": len(rows),
                "completed": len(successes),
                "errors": rejects,
                "success_fraction": len(successes) / len(rows),
                "offered_qps": len(rows) / seconds,
                "offered_input_tpm": sum(row["input_tokens"] for row in rows)
                * 60
                / seconds,
                "offered_session_fraction": sum(bool(row["session"]) for row in rows)
                / len(rows),
                "input_tpm_including_drain": input_tokens * 60 / elapsed,
                "compute_tpm_including_drain": (input_tokens - hit_tokens)
                * 60
                / elapsed,
                "success_token_hit_rate": hit_tokens / input_tokens,
                "completion_requests": len(completed),
                "completion_source": "prefill_done",
                "completion_input_tpm": completion_input * 60 / seconds,
                "completion_compute_tpm": (completion_input - completion_hit)
                * 60
                / seconds,
                "completion_token_hit_rate": (
                    completion_hit / completion_input if completion_input else None
                ),
                "completed_batch_service_ms_by_worker": dict(busy_ms),
                "ttft_samples": len(latencies),
                "recovery_tail_arrivals": len(tail),
                "ttft_p50_s": _percentile(latencies, 0.5),
                "ttft_p95_s": _percentile(latencies, 0.95),
                "ttft_p99_s": _percentile(latencies, 0.99),
                "max_pacing_lag_s": max(row["pacing_lag_s"] for row in rows),
                "max_schedule_launch_lag_s": max(
                    (
                        row["schedule_launch_lag_s"]
                        for row in rows
                        if "schedule_launch_lag_s" in row
                    ),
                    default=None,
                ),
                "max_stream_launch_lag_s": max(
                    (
                        row["stream_launch_lag_s"]
                        for row in rows
                        if "stream_launch_lag_s" in row
                    ),
                    default=None,
                ),
            }
        )
    starts = {stage["stage"]: stage["start_s"] for stage in stages}
    trace = [
        [
            row["stage"],
            row["request_id"],
            row["session"],
            row["input_tokens"],
            row["keys"],
            round(row["planned_s"] - starts[row["stage"]], 6),
        ]
        for row in requests
        if row["stage"] in starts
    ]
    return {
        "stages": report,
        "requests": len(requests),
        "successful": len(prefill),
        "rejected": len(requests) - len(prefill),
        "prefill_execution_batches": len(execution_groups),
        "multi_request_batches": multi_batches,
        "cache_lifecycle": {
            "evict_chain_events": len(cache_events),
            "blocks_freed": sum(event["blocks_freed"] for event in cache_events),
            "by_reason": dict(Counter(event["reason"] for event in cache_events)),
            "by_worker": dict(Counter(event["engine_name"] for event in cache_events)),
        },
        "dispatch": dispatch,
        "input_trace_sha256": hashlib.sha256(
            json.dumps(trace, separators=(",", ":")).encode()
        ).hexdigest(),
    }
