"""Join original request evidence to engine events without inventing identities."""

import json
from pathlib import Path


def master_incarnation(history, target, started_ms, ended_ms):
    """Resolve only an attempt wholly inside one observed process generation."""
    if target is None or started_ms is None or ended_ms is None:
        return None
    candidates = sorted(
        (h for h in history if h["target"] == target),
        key=lambda h: h["started_epoch_ms"],
    )
    before = [h for h in candidates if h["started_epoch_ms"] <= started_ms]
    if not before or any(
        started_ms < h["started_epoch_ms"] <= ended_ms for h in candidates
    ):
        return None
    return dict(before[-1])


def join_evidence(payload, environments):
    events, issues = {}, []
    for epoch, directory in environments.items():
        path = Path(directory) / "engine_events.jsonl"
        if not path.is_file():
            issues.append(dict(env_epoch=epoch, error="missing engine_events.jsonl"))
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("engine event is not an object")
                if "rid" not in event:
                    continue  # process/status events have no request identity
                key = (str(epoch), str(event["rid"]))
                events.setdefault(key, []).append(event)
            except (ValueError, TypeError) as exc:
                issues.append(dict(env_epoch=epoch, line=number, error=str(exc)))
    requests = {}
    anchor = payload["clock_anchor"]
    for resource in payload["request_resources"]:
        epoch = str(resource["resource"]["env_epoch"])
        for row in resource["records"]:
            rid = row.get("wire_request_id", row.get("rid"))
            if rid is None:
                issues.append(
                    dict(
                        resource=resource["resource"], error="missing request identity"
                    )
                )
                continue
            attempt = row.get("attempt")
            key = (epoch, str(rid), attempt)
            if key in requests:
                if requests[key]["original"] != row:
                    issues.append(
                        dict(
                            env_epoch=epoch,
                            request_id=rid,
                            attempt=attempt,
                            error="conflicting duplicate request evidence",
                        )
                    )
                requests[key]["resources"].append(resource["resource"])
                continue
            issued = row.get("issued_s")
            if issued is None and row.get("send_start_epoch_ms") is not None:
                issued = (
                    row["send_start_epoch_ms"] / 1000
                    - anchor["epoch_s"]
                    + anchor["monotonic_s"]
                )
            phases = [
                e
                for e in payload["phases"]
                if e["event"] == "start"
                and str(e["env_epoch"]) == epoch
                and issued is not None
                and e["monotonic_s"] <= issued
            ]
            engine_events = events.get((epoch, str(rid)), [])
            history = payload.get("master_incarnations", {}).get(epoch, [])
            attempts = []
            for observed in row.get("schedule_attempts", []):
                attempts.append(
                    dict(
                        observed,
                        master_incarnation=master_incarnation(
                            history,
                            observed.get("master_target"),
                            observed.get("started_epoch_ms"),
                            observed.get("ended_epoch_ms"),
                        ),
                    )
                )
            if not attempts and row.get("schedule", {}).get("started_s") is not None:
                schedule = row["schedule"]
                started = (
                    schedule["started_s"] - anchor["monotonic_s"] + anchor["epoch_s"]
                ) * 1000
                ended = schedule.get("ended_s")
                ended = (
                    None
                    if ended is None
                    else (ended - anchor["monotonic_s"] + anchor["epoch_s"]) * 1000
                )
                attempts.append(
                    dict(
                        attempt=attempt,
                        master_target=row.get("master_target"),
                        started_epoch_ms=started,
                        ended_epoch_ms=ended,
                        status=schedule.get("status"),
                        master_incarnation=master_incarnation(
                            history, row.get("master_target"), started, ended
                        ),
                    )
                )
            requests[key] = dict(
                run=payload["instance_id"],
                env_epoch=epoch,
                request_id=rid,
                attempt=attempt,
                phase=phases[-1]["stage"] if phases else None,
                master_target=row.get("master_target"),
                endpoint_generation=row.get("endpoint_generation"),
                batch_ids=sorted(
                    {
                        str(e["batch_id"])
                        for e in engine_events
                        if e.get("batch_id") is not None
                    }
                ),
                engine_events=engine_events,
                schedule_attempts=attempts,
                engine_incarnations=sorted(
                    {
                        e["engine_incarnation"]
                        for e in engine_events
                        if e.get("engine_incarnation")
                    }
                ),
                original=row,
                resources=[resource["resource"]],
            )
    return dict(
        schema_version=1,
        requests=list(requests.values()),
        issues=issues,
        identity_contract="Request identity is scoped by environment; attempts retain transport evidence; mock process incarnations are distinct from master endpoint generation, which remains null when unobserved.",
    )
