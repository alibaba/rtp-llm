"""Trace analysis in an exec-isolated, externally bounded process."""

import json
import sys
import time
from pathlib import Path

from .kineto import validate_kineto
from .nsys import normalize_nsys


def main():
    request = json.load(sys.stdin)
    if request["backend"] == "nsys":
        json.dump(analyze_nsys(request), sys.stdout)
        return
    started = time.monotonic()
    ranks = {}
    for rank, name in request["paths"].items():
        before = time.monotonic()
        path = Path(name)
        try:
            with path.open() as stream:
                trace = json.load(stream)
            result = validate_kineto(
                trace, request["steps"].get(rank, []), request["requested_steps"]
            )
            del trace
            result["artifact_bytes"] = path.stat().st_size
        except (OSError, ValueError) as exc:
            result = dict(
                trace_valid=False, reasons=["invalid_trace_file"], error=repr(exc)
            )
        result["analysis_elapsed_s"] = time.monotonic() - before
        ranks[rank] = result
    json.dump(
        dict(ranks=ranks, reasons=[], elapsed_s=time.monotonic() - started), sys.stdout
    )


def analyze_nsys(request):
    started = time.monotonic()
    steps = {int(rank): values for rank, values in request["steps"].items()}
    normalized = normalize_nsys(request["path"], steps)
    parsed = time.monotonic()
    path = Path(request["output"])
    partial = path.with_suffix(".partial.json")
    if path.exists():
        raise FileExistsError(path)
    with partial.open("x") as stream:
        json.dump(normalized, stream, allow_nan=False)
    partial.replace(path)
    finished = time.monotonic()
    normalized.update(
        normalized_file=str(path),
        normalized_bytes=path.stat().st_size,
        sqlite_bytes=Path(request["path"]).stat().st_size,
        normalize_s=parsed - started,
        serialize_s=finished - parsed,
        analysis_elapsed_s=finished - started,
        normalized_event_count=len(normalized.pop("events", [])),
        gpu_gap_count=len(normalized.pop("gpu_gaps", [])),
        gpu_gap_attribution="unavailable_cross_stream_dependency_and_os_evidence",
    )
    return normalized


if __name__ == "__main__":
    main()
