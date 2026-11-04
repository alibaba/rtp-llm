#!/usr/bin/env python3
"""Reconstruct request slots from one node's bounded rotating audit capture.

Keeps every captured request, not just the selected batches. Missing admissions,
slot transitions, sequence gaps and lost audit events invalidate completeness.
Temporal separation is never treated as proof of independent decode cohorts.
"""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path


MARKER = "[navi-request-lifecycle-audit] "


def execution_audit(event):
    result = dict(event)
    collected = event.get("workload_counts_collected", event.get("prefill_tokens", 0) > 0)
    result["workload_counts_collected"] = collected
    if not collected:
        fields = ("running_count", "waiting_count", "preempted_requests", "resumed_requests")
        result["uncollected_raw_counts"] = {key: event.get(key) for key in fields}
        result.update({key: None for key in fields})
    return result


def read_events(paths, max_bytes=2 * 1024**3, max_events=1_000_000):
    events, malformed = [], 0
    decoder = json.JSONDecoder()
    total = 0
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                total += len(line.encode())
                if total > max_bytes:
                    raise ValueError("decompressed input exceeds byte limit")
                if MARKER not in line:
                    continue
                try:
                    # AUTIL appends its own fields after the JSON message.
                    event, _ = decoder.raw_decode(line.split(MARKER, 1)[1].lstrip())
                    if not isinstance(event, dict) or "session" not in event:
                        raise ValueError("missing audit session")
                except (ValueError, TypeError):
                    malformed += 1
                    continue
                events.append(event)
                if len(events) > max_events:
                    raise ValueError("audit event limit exceeded")
    return events, malformed


def analyze(events, target_batches=(), session=None, malformed_lines=0):
    sessions = sorted({e["session"] for e in events})
    if session is None:
        if len(sessions) != 1:
            raise ValueError("select one process session; found " + repr(sessions))
        session = sessions[0]
    events = [e for e in events if e["session"] == session]
    if not events:
        raise ValueError("no events for session")
    errors = []
    error_count = 0

    def error(message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(message)

    if malformed_lines:
        error("malformed audit lines: " + str(malformed_lines))
    ordered = {}
    for event in events:
        if event.get("audit_errors", 0) or event.get("audit_failures", 0):
            error("audit writer reported dropped events")
        if "sequence" not in event:
            continue
        seq = event["sequence"]
        if seq in ordered and ordered[seq] != event:
            error("conflicting duplicate sequence " + str(seq))
        ordered[seq] = event
    sequences = sorted(ordered)
    if not sequences or sequences[0] != 1:
        error("capture missing initial scheduler events")
    for left, right in zip(sequences, sequences[1:]):
        if right != left + 1:
            error("scheduler sequence gap {}:{}".format(left, right))
        if ordered[right]["monotonic_ns"] < ordered[left]["monotonic_ns"]:
            error("scheduler monotonic clock moved backwards")
    requests = defaultdict(lambda: {"events": [], "slots": []})
    active = {}
    for seq in sequences:
        event = ordered[seq]
        rid = str(event["request_id"])
        request = requests[rid]
        request["events"].append(event)
        kind = event["event"]
        if kind == "admitted":
            if "admission" in request:
                error("duplicate admission " + rid)
            request["admission"] = event
        elif kind == "slot_acquire":
            if rid in active or "admission" not in request:
                error("acquisition without admission or duplicate acquisition " + rid)
            active[rid] = event
        elif kind == "slot_release":
            acquired = active.pop(rid, None)
            if acquired is None:
                error("release without captured acquisition " + rid)
            else:
                request["slots"].append({"acquire": acquired, "release": event})
        elif kind == "first_output":
            request.setdefault("first_output", event)
        elif kind == "terminal":
            if "terminal" in request:
                error("duplicate terminal " + rid)
            request["terminal"] = event
        elif kind == "kv_free":
            request["kv_free"] = event
    for rid, event in active.items():
        requests[rid]["slots"].append({"acquire": event, "release": None})
    for rid, request in requests.items():
        if "admission" not in request:
            error("request lifecycle without captured admission " + rid)
        request["complete"] = (
            "admission" in request and "terminal" in request and "kv_free" in request
            and rid not in active
        )
        admission = request.get("admission")
        terminal = request.get("terminal")
        first_output = request.get("first_output")
        acquired = request["slots"][0]["acquire"] if request["slots"] else None
        def elapsed(start, end):
            return (end["monotonic_ns"] - start["monotonic_ns"]) / 1e6 if start and end else None
        request["durations_ms"] = {
            "admission_to_first_slot": elapsed(admission, acquired),
            "first_slot_to_first_output": elapsed(acquired, first_output),
            "first_output_to_terminal": elapsed(first_output, terminal),
            "admission_to_terminal": elapsed(admission, terminal),
        }
        # Terminal does not necessarily release a slot or deferred KV immediately.
        request["resource_end_monotonic_ns"] = max(
            e["monotonic_ns"] for e in request["events"]
            if e["event"] in ("terminal", "kv_free", "slot_release")
        ) if request["complete"] else None
    batches = {}
    for batch in target_batches:
        own = {rid: r for rid, r in requests.items()
               if r.get("admission", {}).get("batch_id") == batch}
        if not own:
            error("target has no captured admissions: " + str(batch))
            continue
        entry = min(r["admission"]["monotonic_ns"] for r in own.values())
        resource_ends = [r["resource_end_monotonic_ns"] for r in own.values()]
        end = max(resource_ends) if all(t is not None for t in resource_ends) else None
        acquired = [s["acquire"]["monotonic_ns"] for r in own.values() for s in r["slots"]]
        first_slot = min(acquired) if acquired else None
        waiting_events = [ordered[seq] for seq in sequences if first_slot is not None
                          and entry <= ordered[seq]["monotonic_ns"] < first_slot]
        later_admitted_acquisitions = []
        for e in waiting_events:
            rid = str(e["request_id"])
            if e["event"] != "slot_acquire" or rid in own:
                continue
            arrival = requests[rid].get("admission")
            if arrival and arrival["monotonic_ns"] > entry:
                later_admitted_acquisitions.append(rid)
        predecessors = {}
        successors = {}
        for rid, request in requests.items():
            admission = request.get("admission")
            terminal = request.get("terminal")
            resource_end = request["resource_end_monotonic_ns"]
            if rid not in own and admission and admission["monotonic_ns"] >= entry:
                if end is None or admission["monotonic_ns"] <= end:
                    successors[rid] = {
                        "batch_id": admission.get("batch_id"),
                        "arrival_after_target_ms": (admission["monotonic_ns"] - entry) / 1e6,
                        "slot_acquires_during_target": [s["acquire"] for s in request["slots"]
                            if end is None or s["acquire"]["monotonic_ns"] <= end],
                        "final_output_tokens": terminal.get("output_tokens") if terminal else None,
                        "complete": request["complete"],
                    }
            if (not admission or admission["monotonic_ns"] >= entry or
                    (resource_end is not None and resource_end < entry)):
                continue
            holding = any(s["acquire"]["monotonic_ns"] <= entry and
                          (s["release"] is None or s["release"]["monotonic_ns"] > entry)
                          for s in request["slots"])
            predecessors[rid] = {
                "batch_id": admission.get("batch_id"), "holding_slot_at_arrival": holding,
                "remaining_lifetime_ms": max(0, (terminal["monotonic_ns"] - entry) / 1e6)
                if terminal else None,
                "remaining_resource_lifetime_ms": (resource_end - entry) / 1e6
                if resource_end is not None else None,
                "final_output_tokens": terminal.get("output_tokens") if terminal else None,
                "complete": request["complete"],
            }
        batches[str(batch)] = {
            "request_ids": sorted(own), "arrival_monotonic_ns": entry,
            "first_slot_monotonic_ns": first_slot,
            "resource_end_monotonic_ns": end,
            "slot_releases_while_waiting": [e for e in waiting_events if e["event"] == "slot_release"],
            "slot_acquisitions_while_waiting": [e for e in waiting_events if e["event"] == "slot_acquire"],
            "later_admitted_acquisitions_before_target": later_admitted_acquisitions,
            "predecessors": predecessors,
            "successors_during_target": successors,
            "captured_running_predecessors": sum(p["holding_slot_at_arrival"] for p in predecessors.values()),
            "logged_running_at_first_admission": min(
                own.values(), key=lambda r: r["admission"]["sequence"]
            )["admission"]["running_count"],
        }
    comparisons = []
    for index, left in enumerate(batches):
        for right in list(batches)[index + 1:]:
            a, b = batches[left], batches[right]
            cohort_a = set(a["predecessors"]) | set(a["request_ids"])
            cohort_b = set(b["predecessors"]) | set(b["request_ids"])
            comparisons.append({
                "batches": [left, right],
                "admission_separation_s": abs(a["arrival_monotonic_ns"] - b["arrival_monotonic_ns"]) / 1e9,
                "shared_unfinished_request_ids": sorted(cohort_a & cohort_b),
                "shared_lifecycle_request_ids": sorted(
                    (cohort_a | set(a["successors_during_target"])) &
                    (cohort_b | set(b["successors_during_target"]))),
                "target_resource_intervals_disjoint": (
                    a["resource_end_monotonic_ns"] < b["arrival_monotonic_ns"] or
                    b["resource_end_monotonic_ns"] < a["arrival_monotonic_ns"]
                ) if (a["resource_end_monotonic_ns"] is not None and
                      b["resource_end_monotonic_ns"] is not None) else None,
                "captured_lifecycles_complete": error_count == 0 and all(
                    requests[rid]["complete"] for rid in
                    cohort_a | cohort_b | set(a["successors_during_target"]) |
                    set(b["successors_during_target"])),
            })
    return {"session": session, "errors": errors, "error_count": error_count,
            "sequence_complete": error_count == 0,
            "fully_drained": all(r["complete"] for r in requests.values()) and bool(requests),
            "request_count": len(requests), "requests": dict(requests),
            "batches": batches, "comparisons": comparisons,
            "execution_steps": [execution_audit(e) for e in events if e.get("event") == "execution_step"],
            "limitations": ["callback spans overlap; do not sum them as GPU time",
                            "disjoint captured cohorts do not rule out all interference",
                            "incomplete sequences cannot establish cohort independence"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--target-batch", type=int, action="append", default=[])
    parser.add_argument("--max-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-events", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0 or args.max_events <= 0:
        parser.error("limits must be positive")
    events, malformed = read_events(args.logs, args.max_bytes, args.max_events)
    print(json.dumps(analyze(events, args.target_batch, args.session, malformed),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
