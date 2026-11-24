"""Certify drained native Remote mock captures without merging clock domains.

A running or crashed service has no final certificate. Legacy captures remain
archived evidence but cannot be certified by this version of the verifier.
"""

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_CAPTURE_BYTES = 8 << 20
MAX_CAPTURE_BYTES = 64 << 20
MAX_CERTIFICATE_BYTES = 64 << 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_budget(max_bytes):
    require(
        type(max_bytes) is int and 0 < max_bytes <= MAX_CAPTURE_BYTES,
        "Remote capture byte budget must be between 1 byte and 64 MiB",
    )


def read_certified_events(path, max_bytes=DEFAULT_CAPTURE_BYTES):
    _check_budget(max_bytes)
    path = Path(path)
    with path.open("rb") as capture:
        raw = capture.read(max_bytes + 1)
    require(len(raw) <= max_bytes, f"{path}: capture exceeds budget")
    return _decode_capture(path, raw)


def _decode_capture(path, raw):
    require(raw.endswith(b"\n"), f"{path}: missing complete final record")
    lines = raw.splitlines(keepends=True)
    require(bool(lines), f"{path}: missing certificate")
    try:
        records = [json.loads(line) for line in lines]
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{path}: malformed capture") from error
    footer = records[-1]
    require(isinstance(footer, dict), f"{path}: missing certificate")
    require(
        set(footer) == {"capture_complete", "event_count", "prefix_bytes"},
        f"{path}: invalid certificate",
    )
    require(
        footer["capture_complete"] is True,
        f"{path}: capture lost events or failed to write",
    )
    require(
        type(footer["event_count"]) is int
        and footer["event_count"] == len(records) - 1,
        f"{path}: event count mismatch",
    )
    require(
        type(footer["prefix_bytes"]) is int
        and footer["prefix_bytes"] == len(raw) - len(lines[-1]),
        f"{path}: prefix byte count mismatch",
    )
    events = records[:-1]
    require(
        all(
            isinstance(event, dict)
            and "event" in event
            and "capture_complete" not in event
            for event in events
        ),
        f"{path}: extra certificate or non-event record",
    )
    return events


def bind_remote_capture(launch_manifest, runtime, address):
    """Bind the external owner's exclusive per-invocation event path before ingress."""
    source = Path(launch_manifest).resolve()
    launch = json.loads(source.read_text())
    require(isinstance(launch, dict), "Remote launch manifest must be an object")
    require(
        isinstance(launch.get("launch_id"), str) and bool(launch["launch_id"]),
        "missing service launch identity",
    )
    require(launch["address"] == address, "Remote service endpoint mismatch")
    path = Path(launch["event_path"])
    require(
        path.is_absolute() and path.is_file(),
        "Remote launch requires a live absolute event path",
    )
    with path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        if stat.st_size:
            stream.seek(max(0, stat.st_size - 512))
            require(
                b'"capture_complete":' not in stream.read(),
                "Remote file already has a final footer",
            )
    return dict(
        source=str(source),
        launch=launch,
        runtime=runtime,
        device=stat.st_dev,
        inode=stat.st_ino,
        begin_bytes=stat.st_size,
    )


def _record_boundary(stream, binding):
    stat = os.fstat(stream.fileno())
    require(
        (stat.st_dev, stat.st_ino) == (binding["device"], binding["inode"]),
        "Remote event file replaced",
    )
    require(stat.st_size >= binding["begin_bytes"], "Remote event file truncated")
    if stat.st_size:
        stream.seek(stat.st_size - 1)
        require(stream.read(1) == b"\n", "Remote offset cuts a record")
    return stat.st_size


def record_remote_end(binding):
    with Path(binding["launch"]["event_path"]).open("rb") as stream:
        return _record_boundary(stream, binding)


def _service_proof(directory, max_bytes, *, archived):
    """Reconcile one completed service capture with immutable engine evidence."""
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    summary_bytes = (directory / "summary.json").read_bytes()
    manifest, summary = json.loads(manifest_bytes), json.loads(summary_bytes)
    capture = manifest["remote_service_capture"]
    require(
        "binding_error" not in capture and "binding" in capture,
        "missing Remote launch binding",
    )
    binding = capture["binding"]
    require(
        "boundary_error" not in binding and "end_bytes" in binding,
        "missing Remote measurement boundary",
    )
    launch = binding["launch"]
    runtime = binding["runtime"]
    require(
        manifest.get("complete") is True
        and capture["run_id"]
        == manifest["session_id"]
        == summary["result"]["session_id"],
        "pressure epoch incomplete or mismatched",
    )
    require(
        runtime == capture["runtime"]
        and all(
            int(runtime[k]) > 0 for k in ("server_epoch", "session_id", "generation")
        ),
        "Remote runtime triple mismatch",
    )
    if not archived:
        require(
            json.loads(Path(binding["source"]).read_text()) == launch,
            "service launch manifest changed",
        )
    pages = manifest["startup_pressure"]["response"]["ranks"]
    ranks = set(manifest["epoch"])
    require(
        bool(ranks)
        and len(pages) == len(ranks)
        and {str(p.get("world_rank", 0)) for p in pages} == ranks,
        "incomplete Remote startup rank evidence",
    )
    for page in pages:
        native = page["remote_runtime"]
        require(
            all(int(native[k]) == v for k, v in runtime.items())
            and native["options"]["mock_remote_cache_address"] == launch["address"]
            and page["service_instance"]
            == manifest["epoch"][str(page.get("world_rank", 0))],
            "Remote startup provenance mismatch",
        )
    path = Path(launch["event_path"])
    source = directory / "remote_events.jsonl" if archived else path
    with source.open("rb") as stream:
        if not archived:
            require(
                _record_boundary(stream, binding) >= binding["end_bytes"],
                "Remote capture truncated after drain",
            )
            stream.seek(0)
        raw = stream.read(max_bytes + 1)
    require(len(raw) <= max_bytes, "Remote capture exceeds byte budget")
    events = _decode_capture(path, raw)
    require(
        all(event.get("generation") == runtime["generation"] for event in events),
        "Remote generation mismatch",
    )
    start, end = binding["begin_bytes"], binding["end_bytes"]
    lines = raw.splitlines(keepends=True)
    require(
        0 <= start <= end <= len(raw) - len(lines[-1]),
        "Remote epoch outside certified events",
    )
    require(
        (not start or raw[start - 1 : start] == b"\n")
        and (not end or raw[end - 1 : end] == b"\n"),
        "Remote slice cuts a record",
    )
    window = [json.loads(line) for line in raw[start:end].splitlines()]
    require(
        not any(row["event"] == "failed" for row in window),
        "failed Remote event inside measurement",
    )
    by_operation = defaultdict(list)
    for row in window:
        op = row.get("operation")
        require(type(op) is int and op > 0, "missing Remote operation identity")
        by_operation[op].append(row)
    all_rows = Counter(row.get("operation") for row in events)
    for op, rows in by_operation.items():
        require(all_rows[op] == len(rows), "Remote operation crosses epoch boundary")
        for row in rows:
            require(
                type(row.get("payload_bytes")) is int and row["payload_bytes"] >= 0,
                "invalid Remote payload byte count",
            )
        names = Counter(row["event"] for row in rows)
        admissions = names["read_admitted"] + names["write_admitted"]
        require(
            admissions == names["complete"] == 1
            and rows[-1]["event"] == "complete"
            and rows[0]["event"] in {"read_admitted", "write_admitted"},
            "incomplete Remote operation in epoch slice",
        )
        require(
            names["read_payload"] == names["read_admitted"]
            and names["write_ack"] == names["write_admitted"],
            "missing Remote payload or acknowledgement",
        )
        if names["write_ack"]:
            received = sum(
                row["payload_bytes"] for row in rows if row["event"] == "write_received"
            )
            ack = sum(
                row["payload_bytes"] for row in rows if row["event"] == "write_ack"
            )
            require(ack > 0 and ack == received, "Remote stored bytes and ACK disagree")
    remote = summary["result"]["mock_remote"]
    require(
        remote.get(
            "remote_drain_valid",
            remote["valid"] and summary["result"]["failure"] is None,
        )
        is True,
        "native Remote drain failed",
    )
    totals = {}
    for name, key in (
        ("read_payload", "remote_read_payload_bytes"),
        ("write_ack", "remote_write_acknowledged_payload_bytes"),
    ):
        totals[key] = sum(
            row["payload_bytes"] for row in window if row["event"] == name
        )
        require(
            totals[key] == remote[key],
            "drained engine and service byte mismatch: " + name,
        )
    require(
        totals["remote_write_acknowledged_payload_bytes"] > 0,
        "no successful Remote write",
    )
    certificate = dict(
        status="certified",
        run_id=manifest["session_id"],
        launch_id=launch["launch_id"],
        runtime=runtime,
        event_path=str(path),
        capture_sha256=hashlib.sha256(raw).hexdigest(),
        begin_bytes=start,
        end_bytes=end,
        window_events=len(window),
        **totals,
    )
    certificate.update(
        schema_version=1,
        scope="remote_service",
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        footer=json.loads(lines[-1]),
    )
    return certificate, raw


def _save(path, data):
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(data)
        temporary = stream.name
    os.replace(temporary, path)


def certify_remote_epoch(directory, *, max_bytes=DEFAULT_CAPTURE_BYTES):
    """Explicit owner-side post-shutdown certification of Remote service bytes."""
    directory = Path(directory)
    _check_budget(max_bytes)
    certificate, raw = _service_proof(directory, max_bytes, archived=False)
    encoded = (json.dumps(certificate, indent=2) + "\n").encode()
    require(
        len(encoded) <= MAX_CERTIFICATE_BYTES, "Remote certificate exceeds byte budget"
    )
    _save(directory / "remote_events.jsonl", raw)
    _save(directory / "remote_certificate.json", encoded)
    return certificate


def read_remote_certificate(directory, *, max_bytes=MAX_CAPTURE_BYTES):
    """Verify a previously issued proof using only the archived epoch files."""
    directory = Path(directory)
    _check_budget(max_bytes)
    with (directory / "remote_certificate.json").open("rb") as stream:
        encoded = stream.read(MAX_CERTIFICATE_BYTES + 1)
    require(
        len(encoded) <= MAX_CERTIFICATE_BYTES, "Remote certificate exceeds byte budget"
    )
    issued = json.loads(encoded)
    expected, _ = _service_proof(directory, max_bytes, archived=True)
    require(
        issued == expected,
        "stored Remote service certificate differs from archived epoch",
    )
    return issued
