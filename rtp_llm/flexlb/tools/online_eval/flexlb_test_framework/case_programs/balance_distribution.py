"""Explicit balance stages preserving client landing, token share, Decode deltas and latency contracts."""

from ..case_config import output

METADATA = {
    "id": "balance_distribution",
    "category": "balance",
    "description": "Explicit balance stages preserving client landing, token share, Decode deltas and "
    "latency contracts.",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def uniform_serial(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", timeout_s=60, params={"role": "prefill"})
    case.step("plain", "balance_start", timeout_s=60, params={"count": 20})
    case.step(
        "plain_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("plain", "requests")},
    )
    case.step(
        "plain_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("plain", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step(
        "plain_p1",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("plain", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "max_share",
            "property": "P1",
        },
    )
    case.step(
        "plain_p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("plain", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "workers",
            "property": "P2",
        },
    )
    case.step(
        "slow_second",
        "engine_control",
        timeout_s=60,
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "second")],
            "perf": {"prefill_fixed_ms": 200},
        },
    )
    case.step("perf_sync", "balance_pause", timeout_s=6.5, params={"seconds": 1.5})
    case.step("speed_hetero", "balance_start", timeout_s=60, params={"count": 20})
    case.step(
        "speed_hetero_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("speed_hetero", "requests")},
    )
    case.step(
        "speed_hetero_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("speed_hetero", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step(
        "speed_hetero_p1",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("speed_hetero", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "max_share",
            "property": "P1",
        },
    )
    case.step(
        "speed_hetero_p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("speed_hetero", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "workers",
            "property": "P2",
        },
    )
    case.step(
        "restore_speed",
        "engine_control",
        timeout_s=60,
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "second")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def concurrent_mix(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", timeout_s=60, params={"role": "prefill"})
    case.step(
        "burst", "balance_start", timeout_s=60, params={"count": 20, "concurrency": 20}
    )
    case.step(
        "terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("burst", "requests")},
    )
    case.step(
        "p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("burst", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
            "min_success": 8,
            "allow_admission": True,
        },
    )
    case.step(
        "p1",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("burst", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "max_share",
            "property": "P1",
            "relax": 1,
        },
    )
    case.step(
        "p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("burst", "requests")],
            "fleet": output("fleet", "snapshot"),
            "metric": "workers",
            "property": "P2",
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def decode_spread(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("n10_before", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step("n10", "balance_start", timeout_s=60, params={"count": 10})
    case.step(
        "n10_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("n10", "requests")},
    )
    case.step("n10_after", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step(
        "n10_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n10", "requests")],
            "fleet": output("n10_before", "snapshot"),
            "metric": "decode_complete",
            "property": "P6",
            "after": output("n10_after", "snapshot"),
        },
    )
    case.step(
        "n10_p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n10", "requests")],
            "fleet": output("n10_before", "snapshot"),
            "metric": "decode_workers",
            "property": "P2",
            "after": output("n10_after", "snapshot"),
            "min_workers": 2,
        },
    )
    case.step(
        "n10_p1",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n10", "requests")],
            "fleet": output("n10_before", "snapshot"),
            "metric": "decode_share",
            "property": "P1",
            "after": output("n10_after", "snapshot"),
            "bands": {"strict": 0.6, "normal": 0.7, "loose": 0.8},
        },
    )
    case.step("n50_before", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step("n50", "balance_start", timeout_s=60, params={"count": 50})
    case.step(
        "n50_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("n50", "requests")},
    )
    case.step("n50_after", "balance_snapshot", timeout_s=60, params={"role": "decode"})
    case.step(
        "n50_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n50", "requests")],
            "fleet": output("n50_before", "snapshot"),
            "metric": "decode_complete",
            "property": "P6",
            "after": output("n50_after", "snapshot"),
        },
    )
    case.step(
        "n50_p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n50", "requests")],
            "fleet": output("n50_before", "snapshot"),
            "metric": "decode_workers",
            "property": "P2",
            "after": output("n50_after", "snapshot"),
            "min_workers": 3,
        },
    )
    case.step(
        "n50_p1",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [output("n50", "requests")],
            "fleet": output("n50_before", "snapshot"),
            "metric": "decode_share",
            "property": "P1",
            "after": output("n50_after", "snapshot"),
            "bands": {"strict": 0.4, "normal": 0.5, "loose": 0.6},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def length_mixed(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "balance_snapshot", timeout_s=60, params={"role": "prefill"})
    case.step(
        "wave1_long1",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 131072,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave1_long1_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave1_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave1_long2",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 135168,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave1_long2_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave1_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave1_short",
        "balance_start",
        timeout_s=60,
        params={
            "count": 6,
            "input_len": 512,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave1_long1_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave1_long1", "requests")},
    )
    case.step(
        "wave1_long2_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave1_long2", "requests")},
    )
    case.step(
        "wave1_short_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave1_short", "requests")},
    )
    case.step(
        "wave1_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave1_long1", "requests"),
                output("wave1_long2", "requests"),
                output("wave1_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step("wave1_master_clean", "balance_clean", timeout_s=30, params={})
    case.step(
        "wave2_long1",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 139264,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave2_long1_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave2_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave2_long2",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 143360,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave2_long2_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave2_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave2_short",
        "balance_start",
        timeout_s=60,
        params={
            "count": 6,
            "input_len": 512,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave2_long1_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave2_long1", "requests")},
    )
    case.step(
        "wave2_long2_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave2_long2", "requests")},
    )
    case.step(
        "wave2_short_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave2_short", "requests")},
    )
    case.step(
        "wave2_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave2_long1", "requests"),
                output("wave2_long2", "requests"),
                output("wave2_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step("wave2_master_clean", "balance_clean", timeout_s=30, params={})
    case.step(
        "wave3_long1",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 147456,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave3_long1_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave3_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave3_long2",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 131072,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave3_long2_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave3_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave3_short",
        "balance_start",
        timeout_s=60,
        params={
            "count": 6,
            "input_len": 512,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave3_long1_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave3_long1", "requests")},
    )
    case.step(
        "wave3_long2_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave3_long2", "requests")},
    )
    case.step(
        "wave3_short_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave3_short", "requests")},
    )
    case.step(
        "wave3_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave3_long1", "requests"),
                output("wave3_long2", "requests"),
                output("wave3_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step("wave3_master_clean", "balance_clean", timeout_s=30, params={})
    case.step(
        "wave4_long1",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 135168,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave4_long1_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave4_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave4_long2",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 139264,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave4_long2_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave4_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave4_short",
        "balance_start",
        timeout_s=60,
        params={
            "count": 6,
            "input_len": 512,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave4_long1_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave4_long1", "requests")},
    )
    case.step(
        "wave4_long2_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave4_long2", "requests")},
    )
    case.step(
        "wave4_short_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave4_short", "requests")},
    )
    case.step(
        "wave4_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave4_long1", "requests"),
                output("wave4_long2", "requests"),
                output("wave4_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step("wave4_master_clean", "balance_clean", timeout_s=30, params={})
    case.step(
        "wave5_long1",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 143360,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave5_long1_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave5_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave5_long2",
        "balance_start",
        timeout_s=60,
        params={
            "input_len": 147456,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave5_long2_pending",
        "balance_pending",
        timeout_s=6,
        params={
            "requests": output("wave5_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave5_short",
        "balance_start",
        timeout_s=60,
        params={
            "count": 6,
            "input_len": 512,
            "unique_keys": False,
            "defer_batch": True,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "wave5_long1_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave5_long1", "requests")},
    )
    case.step(
        "wave5_long2_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave5_long2", "requests")},
    )
    case.step(
        "wave5_short_terminal",
        "balance_wait",
        timeout_s=120,
        params={"requests": output("wave5_short", "requests")},
    )
    case.step(
        "wave5_p6",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave5_long1", "requests"),
                output("wave5_long2", "requests"),
                output("wave5_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "complete",
            "property": "P6",
        },
    )
    case.step("wave5_master_clean", "balance_clean", timeout_s=30, params={})
    case.step(
        "token_p3",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave1_long1", "requests"),
                output("wave1_long2", "requests"),
                output("wave1_short", "requests"),
                output("wave2_long1", "requests"),
                output("wave2_long2", "requests"),
                output("wave2_short", "requests"),
                output("wave3_long1", "requests"),
                output("wave3_long2", "requests"),
                output("wave3_short", "requests"),
                output("wave4_long1", "requests"),
                output("wave4_long2", "requests"),
                output("wave4_short", "requests"),
                output("wave5_long1", "requests"),
                output("wave5_long2", "requests"),
                output("wave5_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "token_share",
            "property": "P3",
        },
    )
    case.step(
        "short_p2",
        "balance_check",
        timeout_s=60,
        params={
            "requests": [
                output("wave1_long1", "requests"),
                output("wave1_long2", "requests"),
                output("wave1_short", "requests"),
                output("wave2_long1", "requests"),
                output("wave2_long2", "requests"),
                output("wave2_short", "requests"),
                output("wave3_long1", "requests"),
                output("wave3_long2", "requests"),
                output("wave3_short", "requests"),
                output("wave4_long1", "requests"),
                output("wave4_long2", "requests"),
                output("wave4_short", "requests"),
                output("wave5_long1", "requests"),
                output("wave5_long2", "requests"),
                output("wave5_short", "requests"),
            ],
            "fleet": output("fleet", "snapshot"),
            "metric": "short_workers",
            "property": "P2",
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


VARIANTS = {
    "uniform_serial": {
        "build": uniform_serial,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "concurrent_mix": {
        "build": concurrent_mix,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "decode_spread": {
        "build": decode_spread,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "length_mixed": {
        "build": length_mixed,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
