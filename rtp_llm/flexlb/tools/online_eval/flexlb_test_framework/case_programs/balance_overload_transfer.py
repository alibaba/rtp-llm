"""Explicit balance stages preserving client landing, token share, Decode deltas and latency contracts."""

from ..case_config import output

METADATA = {
    "id": "balance_overload_transfer",
    "category": "balance",
    "description": "Explicit balance stages preserving client landing, token share, Decode deltas and "
    "latency contracts.",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def decode_pressure(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step(
        "pressure",
        "balance_pressure",
        timeout_s=60,
        params={
            "fleet": output("fleet", "snapshot"),
            "target": output("fleet", "first"),
        },
    )
    case.step("status_sync", "balance_pause", timeout_s=6, params={"seconds": 1})
    case.step("before", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step("traffic", "balance_start", timeout_s=60, params={"count": 10})
    case.step(
        "terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("traffic", "requests")},
    )
    case.step("after", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step(
        "p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("traffic", "requests")],
            "fleet": output("before", "snapshot"),
            "metric": "decode_complete",
            "property": "P6",
            "after": output("after", "snapshot"),
        },
    )
    case.step(
        "p5",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("traffic", "requests")],
            "fleet": output("before", "snapshot"),
            "metric": "target_delta",
            "property": "P5",
            "after": output("after", "snapshot"),
            "target": output("pressure", "target"),
            "bands": {"strict": 0, "normal": 1, "loose": 2},
        },
    )
    case.step(
        "p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("traffic", "requests")],
            "fleet": output("before", "snapshot"),
            "metric": "takeover",
            "property": "P2",
            "after": output("after", "snapshot"),
            "target": output("pressure", "target"),
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def prefill_pressure(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", timeout_s=60, params={"role": "prefill"})
    case.step(
        "slow_both",
        "engine_control",
        timeout_s=60,
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "first"), output("fleet", "second")],
            "perf": {"prefill_fixed_ms": 5000},
        },
    )
    case.step("perf_sync", "balance_pause", timeout_s=6.5, params={"seconds": 1.5})
    case.step(
        "seed",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 147456,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 20,
        },
    )
    case.step(
        "seed_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("seed", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "restore_cool",
        "engine_control",
        timeout_s=60,
        params={
            "operation": "set_perf",
            "targets": [output("seed_pending", "cool")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cool_sync", "balance_pause", timeout_s=5.3, params={"seconds": 0.3})
    case.step("baseline", "balance_start", timeout_s=60, params={"unique_keys": False})
    case.step(
        "baseline_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("baseline", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step(
        "wave",
        "balance_start",
        timeout_s=60,
        params={
            "count": 5,
            "unique_keys": False,
            "await_completion": False,
            "interval_s": 0.12,
        },
    )
    case.step(
        "wave_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("wave", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step(
        "p5",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("wave", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "target_share",
            "property": "P5",
            "target": output("seed_pending", "hot"),
        },
    )
    case.step(
        "p7",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("wave", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "latency_ratio",
            "property": "P7",
            "baseline": output("baseline", "requests"),
        },
    )
    case.step(
        "restore_perf",
        "engine_control",
        timeout_s=60,
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "first"), output("fleet", "second")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "seed_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("seed", "requests")},
    )
    case.step("teardown", "teardown", timeout_s=120)


VARIANTS = {
    "decode_pressure": {
        "build": decode_pressure,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "prefill_pressure": {
        "build": prefill_pressure,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
