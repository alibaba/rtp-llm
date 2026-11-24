"""Nsight SQLite projection. All timestamps remain in the export's ns domain."""

import heapq
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def classify_gpu_gap(
    submit_ns, prior_gpu_end_ns, next_gpu_start_ns, dependency_complete
):
    candidate = max(0, next_gpu_start_ns - prior_gpu_end_ns)
    if not dependency_complete:
        label = "unresolved_gpu_dependency"
    elif submit_ns > prior_gpu_end_ns:
        label = "late_host_submission"
    else:
        label = "submitted_before_prior_gpu_completion"
    return dict(
        classification=label,
        candidate_gap_ns=candidate,
        exposed_idle_ns=candidate if dependency_complete else None,
    )


_REQUIRED = {
    "StringIds": {"id", "value"},
    "NVTX_EVENTS": {"start", "end", "text", "textId", "globalTid", "uint64Value"},
    "CUPTI_ACTIVITY_KIND_RUNTIME": {
        "start",
        "end",
        "globalTid",
        "correlationId",
        "nameId",
    },
    "CUPTI_ACTIVITY_KIND_KERNEL": {
        "start",
        "end",
        "globalPid",
        "correlationId",
        "deviceId",
        "contextId",
        "streamId",
        "demangledName",
    },
}
_OPTIONAL = {
    "TARGET_INFO_SESSION_START_TIME": {"systemClockNs"},
    "ANALYSIS_DETAILS": {"startTime", "stopTime", "duration"},
    "CUPTI_ACTIVITY_KIND_DRIVER": _REQUIRED["CUPTI_ACTIVITY_KIND_RUNTIME"],
    "CUPTI_ACTIVITY_KIND_MEMCPY": {
        "start",
        "end",
        "globalPid",
        "correlationId",
        "deviceId",
        "contextId",
        "streamId",
    },
    "CUPTI_ACTIVITY_KIND_MEMSET": {
        "start",
        "end",
        "globalPid",
        "correlationId",
        "deviceId",
        "contextId",
        "streamId",
    },
    "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION": {
        "start",
        "end",
        "globalPid",
        "correlationId",
        "deviceId",
        "contextId",
        "streamId",
    },
    "TARGET_INFO_CUDA_DEVICE": {"gpuId", "cudaId", "pid", "uuid"},
    "TARGET_INFO_CUDA_CONTEXT_INFO": {
        "hwId",
        "vmId",
        "processId",
        "deviceId",
        "contextId",
    },
}


def normalize_nsys(path, steps_by_rank):
    """Validate actual Nsight 2025.6 tables, join explicit ranks and model work.

    Missing optional tables do not invalidate traces; they never establish that
    dependencies are absent. No CPU causation or exposed GPU idle is inferred
    from a runtime correlation alone.
    """
    try:
        with sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro", uri=True
        ) as db:
            db.row_factory = sqlite3.Row
            tables = {
                r[0]
                for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            missing = []
            for table, columns in {**_REQUIRED, **_OPTIONAL}.items():
                if table not in tables:
                    if table in _REQUIRED:
                        missing.append(table)
                    continue
                actual = {r[1] for r in db.execute(f'PRAGMA table_info("{table}")')}
                missing.extend(f"{table}.{c}" for c in sorted(columns - actual))
            if missing:
                return dict(
                    trace_valid=False, reasons=["unsupported_schema"], missing=missing
                )
            rows = {
                t: [dict(r) for r in db.execute(f'SELECT * FROM "{t}"')]
                for t in (*_REQUIRED, *_OPTIONAL)
                if t in tables
            }
        _normalize_nvtx_clocks(rows)
        return _normalize(rows, steps_by_rank)
    except (sqlite3.Error, OSError, TypeError, ValueError, KeyError) as exc:
        return dict(trace_valid=False, reasons=["invalid_nsys_export"], error=repr(exc))


def _complete_range(start, end):
    return isinstance(start, int) and isinstance(end, int) and end > start


def unique_cuda_launch(calls):
    """Return the outer index for one API call or a nested runtime/driver pair.

    Tuples contain (API kind, CPU thread identity, start, end). Repeated calls
    of the same kind or different threads cannot prove one launch.
    """
    if len(calls) == 1:
        return 0
    if len(calls) != 2 or {c[0] for c in calls} != {"runtime", "driver"}:
        return None
    outer = next(i for i, c in enumerate(calls) if c[0] == "runtime")
    runtime, driver = calls[outer], calls[1 - outer]
    if runtime[1] == driver[1] and runtime[2] <= driver[2] <= driver[3] <= runtime[3]:
        return outer
    return None


def _normalize_nvtx_clocks(rows):
    session = rows.get("TARGET_INFO_SESSION_START_TIME", [])
    details = rows.get("ANALYSIS_DETAILS", [])
    if not session and not details:
        return
    if len(session) != 1 or len(details) != 1:
        raise ValueError("ambiguous_capture_clock_metadata")
    origin = session[0]["systemClockNs"]
    start, stop, duration = (
        details[0][key] for key in ("startTime", "stopTime", "duration")
    )
    if (
        not all(isinstance(value, int) for value in (origin, start, stop, duration))
        or duration <= 0
        or origin <= duration
        or start != origin
        or stop - start != duration
    ):
        raise ValueError("inconsistent_capture_clock_metadata")
    for row in rows["NVTX_EVENTS"]:
        begin, end = row["start"], row["end"]
        if not _complete_range(begin, end):
            continue
        if origin <= begin < end <= stop:
            # This installed exporter leaves some boundary NVTX ranges in its
            # system-clock domain. Metadata proves the translation, not magnitude.
            row["clock_conversion"] = dict(
                raw_start_ns=begin,
                raw_end_ns=end,
                offset_ns=origin,
                duration_ns=duration,
                source="TARGET_INFO_SESSION_START_TIME.systemClockNs+ANALYSIS_DETAILS",
            )
            row["start"], row["end"] = begin - origin, end - origin
        elif begin > duration or end > duration:
            raise ValueError("nvtx_outside_capture_clock_domains")


def _normalize(rows, steps_by_rank):
    strings = {r["id"]: r["value"] for r in rows["StringIds"]}
    reasons, events, markers = set(), [], {}
    all_markers, boundary_exclusions = {}, {}
    processes = {}
    for row in rows["NVTX_EVENTS"]:
        name = row["text"] or strings.get(row["textId"], "")
        if not name.startswith("pressure.step/"):
            continue
        rank = row["uint64Value"]
        if rank is None:
            reasons.add("missing_explicit_rank_metadata")
            continue
        rank = int(rank)
        step = int(name.split("/", 1)[1])
        if rank not in steps_by_rank or step <= 0 or (rank, step) in all_markers:
            reasons.add("invalid_step_identity")
            continue
        pid = row["globalTid"] & 0xFFFFFFFFFF000000
        if rank in processes and processes[rank] != pid:
            reasons.add("ambiguous_rank_process")
        if any(r != rank and p == pid for r, p in processes.items()):
            reasons.add("ambiguous_rank_process")
        processes[rank] = pid
        all_markers[rank, step] = row
        complete = _complete_range(row["start"], row["end"])
        if complete:
            markers[rank, step] = row
        events.append(
            dict(
                kind="step",
                name=name,
                rank=rank,
                step_id=step,
                start_ns=row["start"],
                end_ns=row["end"],
                global_tid=row["globalTid"],
                complete=complete,
                **(
                    {"clock_conversion": row["clock_conversion"]}
                    if "clock_conversion" in row
                    else {}
                ),
            )
        )
    rank_edges = {}
    for rank in steps_by_rank:
        ids = [step for r, step in all_markers if r == rank]
        if ids:
            rank_edges[rank] = min(ids), max(ids)
    for identity in all_markers.keys() - markers.keys():
        rank, step = identity
        if step in rank_edges[rank]:
            boundary_exclusions[identity] = "incomplete_step_boundary"
        else:
            reasons.add("incomplete_internal_step_marker")
            boundary_exclusions[identity] = "incomplete_internal_step_marker"
    calls = {}
    by_thread = defaultdict(list)
    forward_ranges = defaultdict(list)
    for row in rows["NVTX_EVENTS"]:
        name = row["text"] or strings.get(row["textId"], "")
        if name.startswith("executor.model_forward"):
            forward_ranges[row["globalTid"]].append(row)
    for (rank, step), marker in markers.items():
        by_thread[marker["globalTid"]].append(
            (marker["start"], marker["end"], rank, step)
        )
    for ranges in by_thread.values():
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            reasons.add("overlapping_step_markers")
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        for row in rows.get(table, []):
            row["api_kind"] = "runtime" if table.endswith("RUNTIME") else "driver"
            pid = row["globalTid"] & 0xFFFFFFFFFF000000
            ranges = by_thread.get(row["globalTid"], [])
            position = bisect_right(ranges, (row["start"], float("inf"))) - 1
            if position >= 0:
                start, end, rank, step = ranges[position]
                if start <= row["start"] <= row["end"] <= end:
                    row.update(rank=rank, step_id=step)
            calls.setdefault((pid, row["correlationId"]), []).append(row)
            events.append(
                dict(
                    kind="runtime" if table.endswith("RUNTIME") else "driver",
                    name=strings.get(row["nameId"], "unknown"),
                    start_ns=row["start"],
                    end_ns=row["end"],
                    global_tid=row["globalTid"],
                    correlation=row["correlationId"],
                    rank=row.get("rank"),
                    step_id=row.get("step_id"),
                )
            )
    gpu = []
    devices = {}
    for row in rows.get("TARGET_INFO_CUDA_DEVICE", []):
        key = row["pid"], row["cudaId"]
        identity = row["uuid"] or row["gpuId"]
        if key in devices and devices[key] != identity:
            reasons.add("ambiguous_device_metadata")
        devices[key] = identity
    for table, kind in (
        ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
        ("CUPTI_ACTIVITY_KIND_MEMCPY", "copy"),
        ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
        ("CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "synchronization"),
    ):
        for row in rows.get(table, []):
            name = strings.get(row.get("demangledName"), kind)
            event = dict(
                kind=(
                    "communication"
                    if kind == "kernel"
                    and any(
                        word in name.lower()
                        for word in ("nccl", "rccl", "allreduce", "allgather")
                    )
                    else kind
                ),
                name=name,
                start_ns=row["start"],
                end_ns=row["end"],
                global_pid=row["globalPid"],
                device=row["deviceId"],
                context=row["contextId"],
                stream=row["streamId"],
                correlation=row["correlationId"],
            )
            # NVIDIA's encoded process ID retains HW/VM bits for joins. Only
            # its documented OS PID projection is used for device metadata.
            event["physical_device"] = devices.get(
                ((row["globalPid"] >> 24) & 0x00FFFFFF, row["deviceId"])
            )
            if event["end_ns"] < event["start_ns"] or (
                kind != "synchronization" and event["end_ns"] == event["start_ns"]
            ):
                reasons.add("invalid_event_duration")
                continue
            candidates = calls.get((row["globalPid"], row["correlationId"]), [])
            launch = unique_cuda_launch(
                [
                    (c["api_kind"], c["globalTid"], c["start"], c["end"])
                    for c in candidates
                ]
            )
            identities = {(c.get("rank"), c.get("step_id")) for c in candidates}
            valid_launch = launch is not None and len(identities) == 1
            if valid_launch:
                call = candidates[launch]
                # CUDA execution may begin before the API returns, but never
                # before any part of its one proven launch chain enters.
                valid_launch = all(c["start"] <= event["start_ns"] for c in candidates)
            if valid_launch:
                event.update(
                    submit_ns=call["start"],
                    submit_end_ns=call["end"],
                    submit_global_tid=call["globalTid"],
                )
                if "step_id" in call:
                    event.update(
                        rank=call["rank"], step_id=call["step_id"], association="step"
                    )
                else:
                    scopes = forward_ranges[call["globalTid"]]
                    overlaps = any(
                        (f["start"] is None or f["start"] < call["end"])
                        and (f["end"] is None or call["start"] < f["end"])
                        for f in scopes
                    ) or any(
                        start < call["end"] and call["start"] < end
                        for start, end, _, _ in by_thread[call["globalTid"]]
                    )
                    edge = any(
                        marker["globalTid"] == call["globalTid"]
                        and (
                            (
                                step == rank_edges[rank][0]
                                and marker["start"] is None
                                and isinstance(marker["end"], int)
                                and call["end"] <= marker["end"]
                            )
                            or (
                                step == rank_edges[rank][1]
                                and marker["end"] is None
                                and isinstance(marker["start"], int)
                                and call["start"] >= marker["start"]
                            )
                        )
                        for (rank, step), marker in all_markers.items()
                    )
                    if edge:
                        event["association"] = "incomplete_step_boundary"
                    elif not overlaps:
                        event["association"] = "non_forward"
                    elif kind == "kernel":
                        reasons.add("missing_model_compute_correlation")
            elif kind == "kernel":
                reasons.add(
                    "ambiguous_cuda_correlation"
                    if candidates
                    else "missing_cuda_correlation"
                )
            gpu.append(event)
    # Include stage ranges and runtime submissions as independently inspectable
    # evidence, without comparing any export time to server CLOCK_MONOTONIC.
    for row in rows["NVTX_EVENTS"]:
        name = row["text"] or strings.get(row["textId"], "")
        if not name.startswith("pressure.step/"):
            events.append(
                dict(
                    kind="stage",
                    name=name,
                    start_ns=row["start"],
                    end_ns=row["end"],
                    global_tid=row["globalTid"],
                    complete=_complete_range(row["start"], row["end"]),
                    **(
                        {"clock_conversion": row["clock_conversion"]}
                        if "clock_conversion" in row
                        else {}
                    ),
                )
            )
    rank_results = {}
    work_by_step = defaultdict(list)
    for event in gpu:
        if event["kind"] == "kernel" and "step_id" in event:
            work_by_step[event["rank"], event["step_id"]].append(event)
    forwards = defaultdict(list)
    incomplete_forwards = defaultdict(list)
    gathers = defaultdict(list)
    for event in events:
        if event["kind"] == "stage" and event["name"].startswith(
            "executor.model_forward"
        ):
            target = forwards if event["complete"] else incomplete_forwards
            target[event["global_tid"]].append(event)
        if (
            event["kind"] == "stage"
            and event["name"] == "executor.gather_model_input"
            and event["complete"]
        ):
            gathers[event["global_tid"]].append(event)
    for rank, steps in steps_by_rank.items():
        sidecar = {int(s["step_id"]): s for s in steps}
        observed, usable = [], []
        excluded = {
            str(step): reason
            for (r, step), reason in boundary_exclusions.items()
            if r == rank
        }
        first = min((step for r, step in markers if r == rank), default=None)
        for (r, step), marker in markers.items():
            if r != rank:
                continue
            work = work_by_step[rank, step]
            scopes = [
                f
                for f in forwards[marker["globalTid"]]
                if (f["start_ns"] <= marker["start"] <= marker["end"] <= f["end_ns"])
                or (marker["start"] <= f["start_ns"] <= f["end_ns"] <= marker["end"])
            ]
            if not scopes:
                # Only a first/last step can be explained by a clipped enclosing
                # forward. Require its known endpoint near that step; an unrelated
                # open range must not hide missing internal forward evidence.
                leading, trailing = rank_edges[rank]
                prior_ends = [
                    m["end"] for (r, s), m in markers.items() if r == rank and s < step
                ]
                next_starts = [
                    m["start"]
                    for (r, s), m in markers.items()
                    if r == rank and s > step
                ]
                partial = [
                    f
                    for f in incomplete_forwards[marker["globalTid"]]
                    if (
                        step == leading
                        and f["start_ns"] is None
                        and isinstance(f["end_ns"], int)
                        and marker["start"]
                        <= f["end_ns"]
                        <= min(next_starts, default=float("inf"))
                    )
                    or (
                        step == trailing
                        and f["end_ns"] is None
                        and isinstance(f["start_ns"], int)
                        and max(prior_ends, default=-float("inf"))
                        <= f["start_ns"]
                        <= marker["end"]
                    )
                ]
                # A repaired marker needs a complete forward scope to establish
                # CUDA association; clipping cannot bypass that validation.
                if len(partial) == 1 and "clock_conversion" not in marker:
                    excluded[str(step)] = "incomplete_forward_boundary"
                    boundary_exclusions[rank, step] = "incomplete_forward_boundary"
                    continue
            if len(scopes) != 1:
                reasons.add("missing_or_ambiguous_model_forward")
                work = []
            else:
                work = [
                    g
                    for g in work
                    if scopes[0]["start_ns"]
                    <= g["submit_ns"]
                    <= g["submit_end_ns"]
                    <= scopes[0]["end_ns"]
                ]
            record = sidecar.get(step)
            if not record:
                reasons.add("missing_step_sidecar")
                continue
            observed.append(step)
            clock_repaired = "clock_conversion" in marker or any(
                "clock_conversion" in scope for scope in scopes
            )
            if clock_repaired and step not in rank_edges[rank]:
                reasons.add("internal_nvtx_clock_discontinuity")
            if not work:
                reasons.add("missing_model_compute_correlation")
                excluded[str(step)] = "missing_model_compute_correlation"
            elif clock_repaired:
                reason = (
                    "clock_repaired_boundary"
                    if step in rank_edges[rank]
                    else "internal_nvtx_clock_discontinuity"
                )
                excluded[str(step)] = reason
                boundary_exclusions[rank, step] = reason
            elif (
                not record.get("complete")
                or not record.get("context_batch")
                or record.get("decode_batch")
                or record.get("profile_boundary")
            ):
                excluded[str(step)] = "incomplete_or_not_pure_prefill"
            elif step == first and not any(
                g["end_ns"] <= marker["start"] for g in gathers[marker["globalTid"]]
            ):
                excluded[str(step)] = "first_step_gather_omitted"
            else:
                usable.append(step)
        if not observed:
            reasons.add("missing_rank_step_evidence")
        pairs = sum(s + 1 in usable for s in usable)
        if not pairs:
            reasons.add("no_complete_internal_step_pair")
        rank_results[str(rank)] = dict(
            observed_steps=len(observed),
            usable_steps=len(usable),
            observed_step_ids=sorted(observed),
            usable_step_ids=sorted(usable),
            usable_pairs=pairs,
            exclusions=excluded,
        )
    # Validate every kernel, including extra work on otherwise healthy steps.
    # A valid kernel elsewhere in the step cannot hide a bad launch association.
    for event in gpu:
        if event["kind"] not in ("kernel", "communication") or "step_id" not in event:
            continue
        identity = event["rank"], event["step_id"]
        if boundary_exclusions.get(identity) in (
            "incomplete_step_boundary",
            "incomplete_forward_boundary",
        ):
            partial = [
                f
                for f in incomplete_forwards[event["submit_global_tid"]]
                if (
                    identity[1] == rank_edges[identity[0]][0]
                    and f["start_ns"] is None
                    and isinstance(f["end_ns"], int)
                    and event["submit_end_ns"] <= f["end_ns"]
                )
                or (
                    identity[1] == rank_edges[identity[0]][1]
                    and f["end_ns"] is None
                    and isinstance(f["start_ns"], int)
                    and f["start_ns"] <= event["submit_ns"]
                )
            ]
            if len(partial) != 1:
                reasons.add("missing_or_ambiguous_model_forward")
            event["association"] = boundary_exclusions[identity]
            continue
        scopes = [
            f
            for f in forwards[event["submit_global_tid"]]
            if f["start_ns"]
            <= event["submit_ns"]
            <= event["submit_end_ns"]
            <= f["end_ns"]
        ]
        if len(scopes) != 1:
            reasons.add("missing_or_ambiguous_model_forward")
    gaps = []
    gpu_ids = {id(g): len(events) + index for index, g in enumerate(gpu)}
    compute = sorted(
        (
            g
            for g in gpu
            if g["kind"] == "kernel"
            and "step_id" in g
            and (g["rank"], g["step_id"]) not in boundary_exclusions
        ),
        key=lambda g: g["start_ns"],
    )
    waiting, completed = defaultdict(list), {}
    activity = defaultdict(list)
    for g in sorted(gpu, key=lambda g: g["start_ns"]):
        device_key = (
            ("physical", g["physical_device"])
            if g["physical_device"] is not None
            else ("process_local", g["global_pid"], g["device"])
        )
        g["device_identity_kind"] = device_key[0]
        activity[device_key].append(g)
    starts = {d: [g["start_ns"] for g in values] for d, values in activity.items()}
    max_ends = {}
    for device, values in activity.items():
        end, prefix = 0, []
        for value in values:
            end = max(end, value["end_ns"])
            prefix.append(end)
        max_ends[device] = prefix
    for index, nxt in enumerate(compute):
        key = nxt["global_pid"], nxt["device"]
        pending = waiting[key]
        while pending and pending[0][0] <= nxt["start_ns"]:
            _, _, completed[key] = heapq.heappop(pending)
        heapq.heappush(pending, (nxt["end_ns"], index, nxt))
        prior = completed.get(key)
        if prior is None:
            continue
        if prior["end_ns"] == nxt["start_ns"]:
            continue
        device = (
            ("physical", nxt["physical_device"])
            if nxt["physical_device"] is not None
            else ("process_local", nxt["global_pid"], nxt["device"])
        )
        position = bisect_right(starts[device], nxt["start_ns"]) - 1
        overlaps = []
        while position >= 0 and max_ends[device][position] > prior["end_ns"]:
            g = activity[device][position]
            if (
                g is not nxt
                and g["start_ns"] < nxt["start_ns"]
                and g["end_ns"] > prior["end_ns"]
            ):
                overlaps.append(g)
            position -= 1
        gaps.append(
            dict(
                classify_gpu_gap(
                    nxt["submit_ns"], prior["end_ns"], nxt["start_ns"], False
                ),
                rank=nxt["rank"],
                step_id=nxt["step_id"],
                submit_ns=nxt["submit_ns"],
                prior_gpu_end_ns=prior["end_ns"],
                next_gpu_start_ns=nxt["start_ns"],
                prior_gpu_event_index=gpu_ids[id(prior)],
                next_gpu_event_index=gpu_ids[id(nxt)],
                overlapping_activity_indices=[gpu_ids[id(g)] for g in overlaps],
                dependency_complete=False,
                unavailable_reason="cross_stream_dependency_and_os_evidence_incomplete",
            )
        )
    return dict(
        trace_valid=not reasons,
        reasons=sorted(reasons),
        ranks=rank_results,
        time_unit="ns",
        time_domain="nsys_export",
        capture_clock={
            "session": rows.get("TARGET_INFO_SESSION_START_TIME", []),
            "analysis": rows.get("ANALYSIS_DETAILS", []),
        },
        events=events + gpu,
        gpu_gaps=gaps,
        optional_tables_present=sorted(set(rows) & set(_OPTIONAL)),
    )
