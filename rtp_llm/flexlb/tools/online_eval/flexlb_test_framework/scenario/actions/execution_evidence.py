"""Offline execution batches from the existing engine_events.jsonl sidecar.

Execution intervals approximate occupancy: queued work is not observable here.
Batch identity includes engine and execution start to avoid cross-engine ID
collisions and reused IDs. Regrouped batches retain non-cohort members.
"""

import json


def read_events(path):
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def execution_batches(events, request_ids=None):
    groups = {}
    for event in events:
        if event.get("event") != "prefill_done":
            continue
        engine = event["engine_name"]
        start, end, arrival = (
            event[k]
            for k in ("prefill_start_ms", "prefill_done_ms", "engine_arrival_ms")
        )
        if not arrival <= start <= end:
            raise ValueError("invalid execution interval")
        key = (engine, event["batch_id"], start)
        batch = groups.setdefault(
            key,
            dict(
                engine=engine,
                batch_id=event["batch_id"],
                start_ms=start,
                end_ms=end,
                arrival_ms=arrival,
                rids=[],
            ),
        )
        batch["end_ms"] = max(batch["end_ms"], end)
        batch["arrival_ms"] = min(batch["arrival_ms"], arrival)
        if event["rid"] not in batch["rids"]:
            batch["rids"].append(event["rid"])
    rows = list(groups.values())
    if request_ids is not None:
        ids = set(request_ids)
        rows = [r for r in rows if ids.intersection(r["rids"])]
        if set().union(*(set(r["rids"]) & ids for r in rows)) != ids:
            raise ValueError("missing Prefill execution evidence for cohort")
    return sorted(rows, key=lambda r: (r["arrival_ms"], r["batch_id"], r["engine"]))


def avoidance(events, request_ids, engines):
    """Flag placement on the only busy engine; own execution is excluded."""
    if len(set(engines)) != 2:
        raise ValueError("avoidance requires two Prefill engines")
    all_batches = execution_batches(events)
    probes = execution_batches(events, request_ids)
    evidence = []
    for probe in probes:
        t = probe["arrival_ms"]
        busy = {
            e: any(
                b != probe and b["engine"] == e and b["start_ms"] <= t <= b["end_ms"]
                for b in all_batches
            )
            for e in engines
        }
        if probe["engine"] not in busy:
            raise ValueError("probe engine outside configured pool")
        other = next(e for e in engines if e != probe["engine"])
        evidence.append(
            dict(
                batch=probe,
                busy=busy,
                violation=busy[probe["engine"]] and not busy[other],
            )
        )
    return dict(batches=evidence, violations=sum(x["violation"] for x in evidence))


def rotation(batches, max_consecutive, expected_engine_count):
    sequence = [b["engine"] for b in batches]
    longest = run = 0
    previous = None
    for engine in sequence:
        run = run + 1 if engine == previous else 1
        longest = max(longest, run)
        previous = engine
    return dict(
        sequence=sequence,
        max_consecutive=longest,
        passed=len(set(sequence)) == expected_engine_count
        and longest <= max_consecutive,
    )


def partial_outcome_bounds(
    events, request_ids, fallback_success_min, fallback_failure_max
):
    if events is None:
        return dict(
            degraded=True,
            batches=None,
            success_min=fallback_success_min,
            failure_max=fallback_failure_max,
        )
    batches = execution_batches(events, request_ids)
    return dict(
        degraded=False,
        batches=batches,
        success_min=len(set(request_ids)) - len(batches),
        failure_max=len(batches),
    )
