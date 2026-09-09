"""Distinct queue-depth, KV wait/deadline, and global outstanding-capacity admission contracts."""

from ..case_config import output

METADATA = {
    "id": "admission_queue",
    "description": "Distinct queue-depth, KV wait/deadline, and global outstanding-capacity admission "
    "contracts.",
    "category": "admission",
}

PROFILES = ["batch-window", "single-batch"]


def queue_depth(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step(
        "depth_gate",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "queue_depth",
            "options": {"depth": 1},
        },
    )
    case.step(
        "occupants",
        "admission_occupy",
        timeout_s=15,
        params={"targets": ["prefill-0", "prefill-1"]},
    )
    case.step(
        "probe",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 20,
        },
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "depth_error",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "all_error_contains",
            "expected": True,
            "op": "eq",
            "text": ["queue depth"],
            "case_sensitive": True,
        },
    )
    case.step(
        "fast_reject",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "latency_max",
            "expected": 3,
            "op": "lt",
        },
    )
    case.step(
        "clear_depth", "engine_clear", params={"fault": output("depth_gate", "fault")}
    )
    case.step(
        "occupants_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 15,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step("engine_clean", "master_direct_clean", timeout_s=10)
    case.step(
        "normal_perf",
        "engine_control",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


def slo_deadline(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "kv_squeeze",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "kv_pressure",
            "options": {"tokens": 6291456},
        },
    )
    case.step("poll_squeeze", "master_mark", params={"wait_s": 1})
    case.step(
        "probe",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 12,
        },
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "deadline_error_family",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "any_error_contains",
            "expected": True,
            "op": "eq",
            "text": ["deadline", "expired", "exhaust", "8400", "8511", "8431"],
        },
    )
    case.step(
        "waited",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "latency_min",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "bounded_deadline",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "latency_max",
            "expected": 8,
            "op": "le",
        },
    )
    case.step(
        "clear_pressure",
        "engine_clear",
        params={"fault": output("kv_squeeze", "fault")},
    )
    case.step("poll_recovery", "master_mark", params={"wait_s": 1})
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 15,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=20,
        params={"target": "single", "inflight_zero": True},
    )
    case.step("cleanup", "teardown")


def master_capacity(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 4000},
        },
    )
    case.step(
        "wave",
        "admission_wave",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 15,
        },
    )
    case.step(
        "wave_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("wave", "wave")},
    )
    case.step(
        "at_least_one_reject",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "reject_count",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "at_most_two_rejects",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "reject_count",
            "expected": 2,
            "op": "le",
        },
    )
    case.step(
        "at_least_two_served",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "success_count",
            "expected": 2,
            "op": "ge",
        },
    )
    case.step(
        "no_serving_error",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "serve_error_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "typed_code",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "all_reject_code",
            "expected": 8502,
            "op": "eq",
            "scope": "rejected",
        },
    )
    case.step(
        "typed_detail",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "all_error_contains",
            "expected": True,
            "op": "eq",
            "text": ["toomanyrequests", "queue_full"],
            "scope": "rejected",
        },
    )
    case.step(
        "reject_fast",
        "admission_check",
        params={
            "rows": output("wave_done", "rows"),
            "metric": "latency_max",
            "expected": 3,
            "op": "lt",
            "scope": "rejected",
        },
    )
    case.step(
        "normal_perf",
        "engine_control",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 15,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=60,
        params={"wave": output("recovery", "wave")},
    )
    case.step(
        "recovered",
        "admission_check",
        params={
            "rows": output("recovery_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "queue_depth": {
        "build": queue_depth,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "slo_deadline": {
        "build": slo_deadline,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "master_capacity": {
        "build": master_capacity,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
}
