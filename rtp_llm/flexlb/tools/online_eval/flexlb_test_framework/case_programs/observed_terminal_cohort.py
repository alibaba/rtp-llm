"""Preserve terminal-window client records and independent mock source snapshots."""

from ..case_config import output

METADATA = {
    "id": "observed_terminal_cohort",
    "description": "Preserve terminal-window client records and independent mock source snapshots.",
    "category": "status",
    "tags": ["smoke", "observation"],
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def deferred(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "snapshot",
        params={"sources": ["engine_snapshot", "engine_requests"]},
    )
    case.step(
        "submit", "request", params={"count": 2, "output_len": 2, "consume": "deferred"}
    )
    case.step(
        "observing",
        "observe",
        params={
            "mode": "start",
            "sources": ["client_records", "engine_snapshot", "engine_requests"],
            "requests": output("submit", "requests"),
            "max_duration_s": 30,
            "interval_s": 0.2,
            "cohort": "terminal_in_window",
        },
    )
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    case.step(
        "frozen",
        "observe",
        params={"mode": "stop", "observation": output("observing", "observation")},
    )
    case.step(
        "completed",
        "check",
        params={
            "actual": output("terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "no_errors",
        "check",
        params={"actual": output("terminal", "error_count"), "op": "eq", "expected": 0},
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "deferred": {
        "build": deferred,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
}
