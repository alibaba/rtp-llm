"""Concurrent priority queue ordering with explicit terminal and owner checks."""

from ..case_config import output

METADATA = {
    "id": "priority_queue",
    "description": "Concurrent priority queue ordering with explicit terminal and owner checks.",
    "category": "priority",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def same_level_fifo(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "peer0", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer2", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer4", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer5", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "peer6", "priority": 50, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "terminal",
        "priority_wait",
        timeout_s=360,
        params={"requests": output("wave", "requests")},
    )
    case.step("fifo", "priority_fifo", params={"requests": output("wave", "requests")})
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def low_no_starvation(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "priority_prefill_perf",
        params={"expected_prefill": 2, "prefill_fixed_ms": 50},
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "wave0",
        "priority_start",
        timeout_s=760,
        params={
            "serial_schedule": True,
            "gap_s": 1.5,
            "requests": [
                {"tag": "w0-0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w0-1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w0-2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w0-3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w0-4", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w0-5", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w0-6", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w0-7", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "terminal0",
        "priority_wait",
        timeout_s=300,
        params={"requests": output("wave0", "requests")},
    )
    case.step("clean0", "balance_clean", timeout_s=30)
    case.step("quiet0", "balance_pause", params={"seconds": 2})
    case.step(
        "wave1",
        "priority_start",
        timeout_s=760,
        params={
            "serial_schedule": True,
            "gap_s": 1.5,
            "requests": [
                {"tag": "w1-0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w1-1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w1-2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w1-3", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "w1-4", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w1-5", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w1-6", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "w1-7", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "terminal1",
        "priority_wait",
        timeout_s=300,
        params={"requests": output("wave1", "requests")},
    )
    case.step("clean1", "balance_clean", timeout_s=30)
    case.step("quiet1", "balance_pause", params={"seconds": 2})
    case.step("final_clean", "balance_clean", timeout_s=30)
    case.step(
        "completion",
        "priority_completion",
        params={"requests": [output("wave0", "requests"), output("wave1", "requests")]},
    )
    case.step(
        "restore",
        "priority_prefill_perf",
        params={"expected_prefill": 2, "prefill_fixed_ms": 100},
    )
    case.step("teardown", "teardown", timeout_s=120)


def queue_timeout_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 10000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=95,
        params={
            "serial_schedule": True,
            "gap_s": 0,
            "requests": [
                {"tag": "h1", "priority": 70, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=6,
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "low0", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "low1", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "low2", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "h2", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=95,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=235,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "expiry",
        "priority_expiry",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def order_basic(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=95,
        params={
            "serial_schedule": True,
            "gap_s": 0,
            "requests": [
                {"tag": "ph", "priority": 50, "input_len": 2048, "output_len": 2}
            ],
        },
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=6,
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "30a", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "30b", "priority": 30, "input_len": 2048, "output_len": 2},
                {"tag": "50a", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "50b", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "70a", "priority": 70, "input_len": 2048, "output_len": 2},
                {"tag": "70b", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=95,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=215,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "order",
        "priority_order_basic",
        params={
            "first_peer_exempt": False,
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def normalize_default50(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "defer_batch": True,
            "gap_s": 0.3,
            "requests": [
                {"tag": "peer0", "input_len": 2048, "output_len": 2},
                {"tag": "peer1", "input_len": 2048, "output_len": 2, "priority": 50},
                {"tag": "peer2", "input_len": 2048, "output_len": 2},
                {"tag": "peer3", "input_len": 2048, "output_len": 2, "priority": 50},
            ],
        },
    )
    case.step(
        "terminal",
        "priority_normalize_wait",
        timeout_s=235,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params={
            "requests": [output("wave", "requests")],
            "expected_tags": ["peer0", "peer1", "peer2", "peer3"],
            "batch_admission": True,
        },
    )
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step("teardown", "teardown", timeout_s=120)


def normalize_channels(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=95,
        params={
            "serial_schedule": True,
            "gap_s": 0,
            "requests": [{"tag": "ph", "input_len": 2048, "output_len": 2}],
        },
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=6,
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "C", "input_len": 2048, "output_len": 2},
                {
                    "tag": "A",
                    "priority": 70,
                    "qos_level": 30,
                    "input_len": 2048,
                    "output_len": 2,
                },
                {"tag": "B", "qos_level": 70, "input_len": 2048, "output_len": 2},
                {"tag": "G", "priority": 70, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=95,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=215,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params={
            "requests": [output("placeholder", "requests"), output("wave", "requests")],
            "expected_tags": ["ph", "A", "B", "G", "C"],
        },
    )
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def normalize_default30(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("sync", "balance_pause", params={"seconds": 1.5})
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=95,
        params={
            "serial_schedule": True,
            "gap_s": 0,
            "requests": [
                {"tag": "ph", "input_len": 2048, "output_len": 2, "priority": 10}
            ],
        },
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=6,
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {"tag": "Y", "priority": 50, "input_len": 2048, "output_len": 2},
                {"tag": "D", "input_len": 2048, "output_len": 2},
                {"tag": "Z", "priority": 40, "input_len": 2048, "output_len": 2},
                {"tag": "X", "priority": 30, "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=95,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=35,
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=215,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params={
            "requests": [output("placeholder", "requests"), output("wave", "requests")],
            "expected_tags": ["ph", "Y", "Z", "D", "X"],
        },
    )
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step(
        "restore",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("fleet", "prefill")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("teardown", "teardown", timeout_s=120)


def normalize_metrics(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "wave",
        "priority_start",
        timeout_s=15,
        params={
            "gap_s": 0.15,
            "requests": [
                {
                    "tag": "proto70",
                    "priority": 70,
                    "qos_level": 30,
                    "input_len": 2048,
                    "output_len": 2,
                },
                {
                    "tag": "header30",
                    "qos_level": 30,
                    "input_len": 2048,
                    "output_len": 2,
                },
                {"tag": "default50", "input_len": 2048, "output_len": 2},
            ],
        },
    )
    case.step(
        "terminal",
        "priority_normalize_metrics_wait",
        timeout_s=200,
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "metrics",
        "priority_normalize_metrics",
        timeout_s=195,
        params={"requests": output("wave", "requests")},
    )
    case.step("owner_clean", "balance_clean", timeout_s=30)
    case.step("teardown", "teardown", timeout_s=120)


VARIANTS = {
    "same_level_fifo": {
        "build": same_level_fifo,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "low_no_starvation": {
        "build": low_no_starvation,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "queue_timeout_terminal": {
        "build": queue_timeout_terminal,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "order_basic": {
        "build": order_basic,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "normalize_default50": {
        "build": normalize_default50,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {"requires": ["queue"]},
    },
    "normalize_channels": {
        "build": normalize_channels,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {"requires": ["queue"]},
    },
    "normalize_default30": {
        "build": normalize_default30,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {"requires": ["queue"]},
    },
    "normalize_metrics": {
        "build": normalize_metrics,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {"requires": ["queue"]},
    },
}
