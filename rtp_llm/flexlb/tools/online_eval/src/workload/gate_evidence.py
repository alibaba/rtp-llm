"""Freeze and persist complete gate inputs independently of presentation."""

import hashlib
import json
from pathlib import Path


def new_evidence(schema_field, clock, criteria, *, instance, version=1, **payload):
    """Common acquisition envelope; case-owned payload and version stay explicit."""
    if not schema_field.endswith("_evidence_schema_version"):
        raise ValueError("gate evidence requires a named format version")
    return dict({schema_field: version}, clock=clock.to_dict(), criteria=criteria,
                errors=[], samples=[], provenance=dict(instance=instance), **payload)


def trace_workload_sha(path):
    """Ignore only run-local request identity; keep exact tokens, order and lengths."""
    h = hashlib.sha256()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            row.pop("rid", None)
            h.update(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    return h.hexdigest()


def compact_flow(snapshot):
    """Keep every request and gate field; verbose RPC metadata stays in the journal."""
    issued = {"rid", "send_start_epoch_ms", "input_len", "output_len", "pacing_lag_ms"}
    terminal = {"rid", "send_start_epoch_ms", "input_len", "output_len", "status",
                "total_ms", "ttft_ms", "observed_output_tokens"}
    result = {k: v for k, v in snapshot.items() if k not in {"issued", "records"}}
    result["issued"] = [{k: v for k, v in row.items() if k in issued}
                        for row in snapshot["issued"]]
    result["records"] = [{k: v for k, v in row.items() if k in terminal}
                         for row in snapshot["records"]]
    return result


def write_evidence(path, evidence):
    """Atomic compact JSON; raw journals remain the source of detailed RPC fields."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(evidence, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
