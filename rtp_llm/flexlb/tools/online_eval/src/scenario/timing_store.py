"""Locked, advisory duration estimates for lane scheduling."""
import fcntl
import json
import math
import os
import sys
from pathlib import Path

from artifacts.json_io import write_json
from schema_contract import matches_schema

def timing_path():
    return Path(
        os.environ.get(
            "FLEXLB_FT_TIMING_BASELINE",
            "/tmp/flexlb_ft_instance_timing_v1.json",
        )
    )


def read_timings(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not (matches_schema(data, "timings_schema_version", 1) or matches_schema(data, "run_summary_schema_version", 1)) or not isinstance(
            data.get("instances"), list
        ):
            raise ValueError("expected versioned instance timings")
        timings = {}
        for row in data["instances"]:
            duration = row.get("duration_ms")
            if (
                isinstance(row.get("id"), str)
                and type(duration) in (int, float)
                and math.isfinite(duration)
                and (duration > 0)
            ):
                timings[row["id"]] = duration / 1000
        return timings
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(
            f"warning: cannot use instance timings {path}: {exc}; using static estimates",
            file=sys.stderr,
        )
        return {}


def write_timings(rows):
    path = timing_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        merged = read_timings(path)
        for row in rows:
            if (
                row["status"] not in {"ERROR", "TIMEOUT"}
                and type(row.get("duration_ms")) in (int, float)
                and math.isfinite(row["duration_ms"])
                and (row["duration_ms"] > 0)
            ):
                merged[row["id"]] = row["duration_ms"] / 1000
        doc = {
            "timings_schema_version": 1,
            "instances": [
                {"id": key, "duration_ms": value * 1000}
                for (key, value) in sorted(merged.items())
            ],
        }
        write_json(path, doc)
