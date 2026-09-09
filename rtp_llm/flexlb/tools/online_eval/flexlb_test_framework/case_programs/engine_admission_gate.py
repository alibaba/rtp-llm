"""Separate prefill concurrency, decode hard gate, prefill waiting cap and KV pool capacity programs."""

from ..case_config import output

METADATA = {
    "id": "engine_admission_gate",
    "description": "Separate prefill concurrency, decode hard gate, prefill waiting cap and KV pool "
    "capacity programs.",
    "category": "admission",
    "requires": ["enqueue_batch"],
}

PROFILES = ["batch-window", "single-batch"]


def prefill_concurrency(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step(
        "fired",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 4,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "deferred",
            "request_timeout_s": 45,
            "spacing_s": 0.4,
            "spacing_every": 1,
            "keys_per_request": 0,
        },
    )
    case.step(
        "all_admitted",
        "admission_check",
        params={
            "rows": output("fired", "rows"),
            "metric": "admitted_count",
            "expected": 4,
            "op": "eq",
        },
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=20,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches", "waiting"],
            "duration_s": 10,
            "reduce": "any",
            "until_value": 1,
            "until_op": "ge",
        },
    )
    case.step(
        "park_seen",
        "admission_gauge_check",
        params={
            "snapshot": output("park", "snapshot"),
            "fields": ["prefill_waiting_batches", "waiting"],
            "stat": "max_seen",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "drained",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("fired", "wave")},
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("drained", "rows"),
            "metric": "success_count",
            "expected": 4,
            "op": "eq",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=20,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches", "waiting"],
            "duration_s": 10,
            "reduce": "all",
            "until_value": 0,
            "until_op": "eq",
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["prefill_waiting_batches", "waiting"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=120,
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
        "normal_perf",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


def decode_hard_gate(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_decode",
        "engine_control",
        params={
            "targets": ["decode-0"],
            "operation": "set_perf",
            "perf": {"decode_scale": 10},
        },
    )
    case.step(
        "fired",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 280,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 64,
            "consume": "immediate",
            "request_timeout_s": 45,
            "spacing_s": 0.05,
            "spacing_every": 25,
            "keys_per_request": 0,
        },
    )
    case.step(
        "all_admitted",
        "admission_check",
        params={
            "rows": output("fired", "rows"),
            "metric": "admitted_count",
            "expected": 280,
            "op": "eq",
        },
    )
    case.step(
        "park",
        "admission_observe",
        timeout_s=28,
        params={
            "targets": ["decode-0"],
            "fields": ["active_decode_requests", "waiting"],
            "duration_s": 18,
            "reduce": "all",
        },
    )
    case.step(
        "conditional_park",
        "admission_decode_park_check",
        params={"snapshot": output("park", "snapshot"), "gate": 128},
    )
    case.step(
        "drained",
        "admission_wait",
        timeout_s=300,
        params={"wave": output("fired", "wave")},
    )
    case.step(
        "completed_95pct",
        "admission_check",
        params={
            "rows": output("drained", "rows"),
            "metric": "success_count",
            "expected": 266,
            "op": "ge",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=25,
        params={
            "targets": ["decode-0"],
            "fields": ["waiting"],
            "duration_s": 15,
            "reduce": "all",
            "until_value": 0,
            "until_op": "eq",
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=60,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=120,
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
        "normal_perf",
        "engine_control",
        params={
            "targets": ["decode-0"],
            "operation": "set_perf",
            "perf": {"decode_scale": 1},
        },
    )
    case.step("cleanup", "teardown")


def prefill_waiting_cap(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "limited",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 3000, "max_waiting_batches": 1},
        },
    )
    case.step(
        "occupants",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 2,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "immediate",
            "request_timeout_s": 45,
            "spacing_s": 0.4,
            "spacing_every": 1,
            "keys_per_request": 0,
        },
    )
    case.step(
        "occupants_admitted",
        "admission_check",
        params={
            "rows": output("occupants", "rows"),
            "metric": "admitted_count",
            "expected": 2,
            "op": "eq",
        },
    )
    case.step(
        "saturated",
        "admission_observe",
        timeout_s=18,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches"],
            "duration_s": 8,
            "reduce": "all",
            "until_value": 1,
            "until_op": "ge",
        },
    )
    case.step(
        "cap_seen",
        "admission_gauge_check",
        params={
            "snapshot": output("saturated", "snapshot"),
            "fields": ["prefill_waiting_batches"],
            "stat": "min_latest",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "probe",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 10,
            "keys_per_request": 0,
        },
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "backpressure_error",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "all_error_contains",
            "expected": True,
            "op": "eq",
            "text": ["prefill waiting queue full", "backpressure"],
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
        "unbounded",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"max_waiting_batches": 0},
        },
    )
    case.step(
        "before_fourth",
        "admission_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches"],
            "duration_s": 0,
            "reduce": "all",
        },
    )
    case.step(
        "fourth",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "immediate",
            "request_timeout_s": 45,
            "spacing_s": 0,
            "spacing_every": 1,
            "keys_per_request": 0,
        },
    )
    case.step(
        "fourth_admitted",
        "admission_check",
        params={
            "rows": output("fourth", "rows"),
            "metric": "admitted_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "occupants_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "occupants_complete",
        "admission_check",
        params={
            "rows": output("occupants_done", "rows"),
            "metric": "success_count",
            "expected": 2,
            "op": "eq",
        },
    )
    case.step(
        "fourth_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("fourth", "wave")},
    )
    case.step(
        "pressure_recovery",
        "admission_check",
        params={
            "rows": output("fourth_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=20,
        params={
            "targets": ["prefill-0"],
            "fields": ["prefill_waiting_batches", "waiting"],
            "duration_s": 10,
            "reduce": "all",
            "until_value": 0,
            "until_op": "eq",
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["prefill_waiting_batches", "waiting"],
            "stat": "max_latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=120,
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
        "normal_perf",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 100, "max_waiting_batches": 0},
        },
    )
    case.step("cleanup", "teardown")


def kv_pool_capacity(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step(
        "occupants",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 2,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "immediate",
            "request_timeout_s": 45,
            "spacing_s": 0.4,
            "spacing_every": 1,
            "keys_per_request": 8,
        },
    )
    case.step(
        "occupants_admitted",
        "admission_check",
        params={
            "rows": output("occupants", "rows"),
            "metric": "admitted_count",
            "expected": 2,
            "op": "eq",
        },
    )
    case.step(
        "saturated",
        "admission_observe",
        timeout_s=18,
        params={
            "targets": ["prefill-0"],
            "fields": ["held_blocks"],
            "duration_s": 8,
            "reduce": "all",
            "until_value": 16,
            "until_op": "ge",
        },
    )
    case.step(
        "pool_full",
        "admission_gauge_check",
        params={
            "snapshot": output("saturated", "snapshot"),
            "fields": ["held_blocks"],
            "stat": "min_latest",
            "op": "ge",
            "expected": 16,
        },
    )
    case.step(
        "probe",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 10,
            "keys_per_request": 8,
        },
    )
    case.step(
        "probe_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("probe", "wave")},
    )
    case.step(
        "lack_mem_error",
        "admission_check",
        params={
            "rows": output("probe_done", "rows"),
            "metric": "all_error_contains",
            "expected": True,
            "op": "eq",
            "text": ["lack_mem", "insufficient kv cache", "enqueuebatch rejected"],
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
        "occupants_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("occupants", "wave")},
    )
    case.step(
        "occupants_complete",
        "admission_check",
        params={
            "rows": output("occupants_done", "rows"),
            "metric": "success_count",
            "expected": 2,
            "op": "eq",
        },
    )
    case.step(
        "available",
        "admission_observe",
        timeout_s=20,
        params={
            "targets": ["prefill-0"],
            "fields": ["available_blocks"],
            "duration_s": 10,
            "reduce": "all",
            "until_value": 8,
            "until_op": "ge",
        },
    )
    case.step(
        "pool_recovered",
        "admission_gauge_check",
        params={
            "snapshot": output("available", "snapshot"),
            "fields": ["available_blocks"],
            "stat": "min_latest",
            "op": "ge",
            "expected": 8,
        },
    )
    case.step(
        "fresh_lease",
        "admission_fire",
        timeout_s=180,
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "immediate",
            "request_timeout_s": 15,
            "spacing_s": 0,
            "spacing_every": 1,
            "keys_per_request": 8,
        },
    )
    case.step(
        "fresh_done",
        "admission_wait",
        timeout_s=120,
        params={"wave": output("fresh_lease", "wave")},
    )
    case.step(
        "fresh_succeeded",
        "admission_check",
        params={
            "rows": output("fresh_done", "rows"),
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
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=15,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "recovery",
        "admission_wave",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=120,
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
        "normal_perf",
        "engine_control",
        params={
            "targets": ["prefill-0"],
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "prefill_concurrency": {
        "build": prefill_concurrency,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "decode_hard_gate": {
        "build": decode_hard_gate,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "prefill_waiting_cap": {
        "build": prefill_waiting_cap,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "kv_pool_capacity": {
        "build": kv_pool_capacity,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
}
