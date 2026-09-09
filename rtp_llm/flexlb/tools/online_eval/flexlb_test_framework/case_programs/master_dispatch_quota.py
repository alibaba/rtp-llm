"""Fill one batch quota, stop its prefill, observe blocked traffic and independent scheduler TTL cleanup, then recover."""

from ..case_config import output

METADATA = {
    "id": "master_dispatch_quota",
    "description": "Fill one batch quota, stop its prefill, observe blocked traffic and independent "
    "scheduler TTL cleanup, then recover.",
    "category": "master",
    "requires": ["enqueue_batch"],
}

PROFILES = ["batch-window", "single-batch"]


def single_prefill_ttl(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 10000},
        },
    )
    case.step(
        "fill", "request", params={"count": 4, "output_len": 10, "consume": "deferred"}
    )
    case.step(
        "all_fill_admitted",
        "master_admission_check",
        params={"requests": output("fill", "requests"), "count": 4},
    )
    case.step(
        "quota_held",
        "master_wait_inflight",
        timeout_s=15,
        params={"op": "ge", "value": 1},
    )
    case.step(
        "stop", "engine_control", params={"operation": "stop", "targets": ["prefill-0"]}
    )
    case.step("eviction_begin", "master_mark", params={"wait_s": 3})
    case.step(
        "blocked",
        "master_request_batch",
        params={
            "count": 10,
            "concurrency": 10,
            "request_timeout_s": 12,
            "sample_topology": False,
        },
    )
    case.step(
        "block_verdict",
        "check",
        params={
            "actual": output("blocked", "success_rate"),
            "op": "le",
            "expected": 0.5,
        },
    )
    case.step(
        "ttl_empty",
        "master_wait_inflight",
        timeout_s=95,
        params={"op": "eq", "value": 0},
    )
    case.step(
        "start",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "normal_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("ready", "master_prefill_alive", timeout_s=30)
    case.step("settle", "master_mark", params={"wait_s": 2})
    case.step(
        "recovery",
        "master_request_batch",
        timeout_s=330,
        params={"count": 20, "concurrency": 1, "request_timeout_s": 15},
    )
    case.step(
        "recovered",
        "check",
        params={
            "actual": output("recovery", "success_rate"),
            "op": "ge",
            "expected": 0.9,
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "single_prefill_ttl": {
        "build": single_prefill_ttl,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
}
