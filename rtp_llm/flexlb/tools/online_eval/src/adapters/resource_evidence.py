"""Project client resources to the execution evidence contract."""

from case_api.extension_contracts import ResourceEvidence


def flow_evidence(value, profile, deadline):
    deadline.check()
    snapshot = value.evidence_snapshot()
    manifest = snapshot if profile == "diagnostic" else {
        k: v for k, v in snapshot.items() if k not in ("records", "issued", "unfinished")}
    if profile != "diagnostic":
        manifest["request_journal"] = str(value.directory / "client_lifecycle.jsonl")
    return ResourceEvidence(requests=manifest,
        producer="python" if snapshot.get("producer_kind") == "python" else id(value),
        errors=tuple(snapshot["errors"] or ["incomplete request journal"]) if not snapshot["complete"] else ())


def request_evidence(value, profile, deadline):
    deadline.check()
    from adapters.requests import completeness
    rows = value.snapshot_records()
    integrity = completeness(rows)
    return ResourceEvidence(requests=dict(records=rows), producer="python",
        errors=() if integrity["complete"] else (dict(
            error="request consumers did not all finish", detail=integrity),))
