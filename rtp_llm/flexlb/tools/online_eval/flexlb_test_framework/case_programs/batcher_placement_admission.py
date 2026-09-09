"""Master queue and placement backlog, sampled during Schedule; explicit old deadline, FIFO wait-return and recovery contracts."""

from ..case_config import output

METADATA = {
    "id": "batcher_placement_admission",
    "description": "Master queue and placement backlog, sampled during Schedule; explicit old "
    "deadline, FIFO wait-return and recovery contracts.",
    "category": "admission",
}

PROFILES = ["batch-window", "single-batch", "single-nonbatch", "window-nonbatch"]


def batcher_queue_capacity_park(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step(
        "sampling",
        "admission_park_start",
        timeout_s=15,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "wave",
        "admission_tracked_fire",
        timeout_s=90,
        params={
            "count": 7,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 45,
            "schedule_timeout_s": 30,
            "spacing_s": 0.4,
        },
    )
    case.step(
        "sampled",
        "admission_park_stop",
        timeout_s=15,
        params={"sampler": output("sampling", "sampler")},
    )
    case.step(
        "all_admitted",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "admitted_count",
            "expected": 7,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "park_proven",
        "admission_park_check",
        params={"samples": output("sampled", "samples")},
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=45,
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "all_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 7,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "fifo",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "await_fifo",
            "expected": True,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "empty",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["waiting", "prefill_waiting_batches"],
            "duration_s": 10,
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "park_empty",
        "admission_gauge_check",
        params={
            "snapshot": output("empty", "snapshot"),
            "fields": ["waiting", "prefill_waiting_batches"],
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
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
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
            "scope": "all",
        },
    )
    case.step(
        "relieve",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


def batcher_queue_deadline(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step(
        "sampling",
        "admission_park_start",
        timeout_s=15,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "wave",
        "admission_tracked_fire",
        timeout_s=90,
        params={
            "count": 8,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 30,
            "schedule_timeout_s": 30,
            "spacing_s": 0.15,
        },
    )
    case.step(
        "sampled",
        "admission_park_stop",
        timeout_s=15,
        params={"sampler": output("sampling", "sampler")},
    )
    case.step(
        "six_admitted",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "admitted_count",
            "expected": 6,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "two_rejected",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "reject_count",
            "expected": 2,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "deadline_family",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "deadline_reject_family",
            "expected": True,
            "op": "eq",
            "scope": "rejected",
        },
    )
    case.step(
        "deadline_min",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "schedule_latency_min",
            "expected": 1,
            "op": "ge",
            "scope": "rejected",
        },
    )
    case.step(
        "deadline_max",
        "admission_check",
        params={
            "rows": output("wave", "rows"),
            "metric": "schedule_latency_max",
            "expected": 5,
            "op": "le",
            "scope": "rejected",
        },
    )
    case.step(
        "park_proven",
        "admission_park_check",
        params={
            "samples": output("sampled", "samples"),
            "overflow_rows": output("wave", "rows"),
        },
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=30,
        params={"waves": [output("wave", "wave")]},
    )
    case.step(
        "six_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 6,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "no_serving_errors",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "serve_error_count",
            "expected": 0,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "relieve",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
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
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
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
            "scope": "all",
        },
    )
    case.step("cleanup", "teardown")


def placement_pool_wait(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 5000},
        },
    )
    case.step(
        "a",
        "admission_tracked_fire",
        timeout_s=90,
        params={
            "count": 1,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 30,
            "schedule_timeout_s": 30,
            "spacing_s": 0,
        },
    )
    case.step(
        "a_admitted",
        "admission_check",
        params={
            "rows": output("a", "rows"),
            "metric": "admitted_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "running",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["running"],
            "duration_s": 10,
            "until_op": "ge",
            "until_value": 1,
        },
    )
    case.step(
        "a_running",
        "admission_gauge_check",
        params={
            "snapshot": output("running", "snapshot"),
            "fields": ["running"],
            "stat": "max_seen",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "lease_before_b",
        "admission_lease_precondition",
        timeout_s=15,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "sampling",
        "admission_park_start",
        timeout_s=15,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "b",
        "admission_tracked_fire",
        timeout_s=90,
        params={
            "count": 1,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 2,
            "request_timeout_s": 30,
            "schedule_timeout_s": 60,
            "spacing_s": 0,
        },
    )
    case.step(
        "sampled",
        "admission_park_stop",
        timeout_s=15,
        params={"sampler": output("sampling", "sampler")},
    )
    case.step(
        "b_admitted",
        "admission_check",
        params={
            "rows": output("b", "rows"),
            "metric": "admitted_count",
            "expected": 1,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "b_rpc_parked",
        "admission_check",
        params={
            "rows": output("b", "rows"),
            "metric": "schedule_latency_min",
            "expected": 0.5,
            "op": "gt",
            "scope": "all",
        },
    )
    case.step(
        "park_proven",
        "admission_park_check",
        params={"samples": output("sampled", "samples")},
    )
    case.step(
        "done",
        "admission_drain",
        timeout_s=30,
        params={"waves": [output("a", "wave"), output("b", "wave")]},
    )
    case.step(
        "both_completed",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "success_count",
            "expected": 2,
            "op": "eq",
            "scope": "all",
        },
    )
    case.step(
        "b_after_a",
        "admission_check",
        params={
            "rows": output("done", "rows"),
            "metric": "await_strict_fifo",
            "expected": True,
            "op": "eq",
            "scope": "all",
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
            "input_len": 2048,
            "output_len": 2,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "recovery_done",
        "admission_wait",
        timeout_s=40,
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
            "scope": "all",
        },
    )
    case.step(
        "relieve",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "batcher_queue_capacity_park": {
        "build": batcher_queue_capacity_park,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "batcher_queue_deadline": {
        "build": batcher_queue_deadline,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "placement_pool_wait": {
        "build": placement_pool_wait,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {
            "requires": ["generate_stream"],
        },
    },
}
