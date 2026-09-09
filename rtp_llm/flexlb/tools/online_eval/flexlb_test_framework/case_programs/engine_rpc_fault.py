"""Explicit request baselines, controlled RPC faults, consumer-complete outcomes, recovery and applicable scheduler-owner cleanup."""

from ..case_config import output

METADATA = {
    "id": "engine_rpc_fault",
    "description": "Explicit request baselines, controlled RPC faults, consumer-complete outcomes, "
    "recovery and applicable scheduler-owner cleanup.",
    "category": "engine_fault",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def enqueue_delay_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "request_total", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "baseline_ready",
        "check",
        params={
            "actual": output("baseline", "legacy_success"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "inject",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "enqueue_delay",
            "options": {"delay_ms": 1500},
        },
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "request_total", "input_len": 2048, "output_len": 10},
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "request_total", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params={
            "baseline": output("baseline", "requests"),
            "delayed": output("delayed", "requests"),
            "recovery": output("recovery", "requests"),
            "metric": "request_total",
            "min_delta_s": 1.2,
            "recovery_slack_s": 1.0,
        },
    )
    case.step("owner_clean", "rpc_owner_clean", timeout_s=10)
    case.step("cleanup", "teardown")


def generate_delay_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "baseline_ready",
        "check",
        params={
            "actual": output("baseline", "legacy_success"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "inject",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "generate_delay",
            "options": {"delay_ms": 1500},
        },
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params={
            "baseline": output("baseline", "requests"),
            "delayed": output("delayed", "requests"),
            "recovery": output("recovery", "requests"),
            "metric": "stream_ttft",
            "min_delta_s": 1.2,
            "recovery_slack_s": 1.0,
        },
    )
    case.step("owner_clean", "rpc_owner_clean", timeout_s=10)
    case.step("cleanup", "teardown")


def generate_delay_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "baseline_ready",
        "check",
        params={
            "actual": output("baseline", "legacy_success"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "inject",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "generate_delay",
            "options": {"delay_ms": 1500},
        },
    )
    case.step(
        "delayed",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "recovery",
        "rpc_latency_sample",
        timeout_s=125,
        params={"mode": "stream_ttft", "input_len": 2048, "output_len": 10},
    )
    case.step(
        "latency",
        "rpc_latency_check",
        params={
            "baseline": output("baseline", "requests"),
            "delayed": output("delayed", "requests"),
            "recovery": output("recovery", "requests"),
            "metric": "stream_ttft",
            "min_delta_s": 1.2,
            "recovery_slack_s": 1.0,
        },
    )
    case.step("cleanup", "teardown")


def enqueue_error(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "inject",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "enqueue_error",
            "options": {},
        },
    )
    case.step(
        "probe",
        "rpc_fault_probe",
        timeout_s=45,
        params={
            "input_len": 2048,
            "output_len": 10,
            "schedule_timeout_s": 30,
            "stream_timeout_s": 10,
            "expected_rpc_statuses": [
                "UNKNOWN",
                "INTERNAL",
                "UNAVAILABLE",
                "DEADLINE_EXCEEDED",
            ],
            "require_error_detail": True,
        },
    )
    case.step("clear", "engine_clear", params={"fault": output("inject", "fault")})
    case.step(
        "cancel_failed_delivery",
        "rpc_probe_cancel",
        params={"requests": output("probe", "requests")},
    )
    case.step("recovery_settle", "balance_pause", params={"seconds": 3})
    case.step(
        "recovery",
        "request",
        params={
            "count": 1,
            "input_len": 2048,
            "output_len": 2,
            "schedule_timeout_s": 30,
            "stream_timeout_s": 30,
        },
    )
    case.step(
        "recovery_terminal",
        "wait",
        timeout_s=30,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "check",
        params={
            "actual": output("recovery_terminal", "completed"),
            "op": "eq",
            "expected": True,
        },
    )
    case.step(
        "owner_clean",
        "rpc_probe_owner_clean",
        timeout_s=95,
        params={"requests": output("probe", "requests")},
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "enqueue_delay_batch": {
        "build": enqueue_delay_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "generate_delay_batch": {
        "build": generate_delay_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "generate_delay_nonbatch": {
        "build": generate_delay_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {
            "requires": [],
        },
    },
    "enqueue_error": {
        "build": enqueue_error,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
