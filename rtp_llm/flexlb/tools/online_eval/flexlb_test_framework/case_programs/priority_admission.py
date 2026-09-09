"""No preemption: after the victim reaches decode running, its acceptance permit is free and a higher-priority incomer is admitted alongside it."""

from ..case_config import output

METADATA = {
    "id": "priority_admission",
    "description": "No preemption: after the victim reaches decode running, its acceptance permit is "
    "free and a higher-priority incomer is admitted alongside it.",
    "category": "admission",
    "requires": ["enqueue_batch"],
}

PROFILES = ["batch-window"]


def permit_released_without_preemption(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "victim",
        "admission_fire",
        params={
            "count": 1,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 200,
            "priority": 30,
            "keys_per_request": 1,
            "spacing_s": 0,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "victim_admitted",
        "admission_check",
        params={
            "rows": output("victim", "rows"),
            "metric": "admitted_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "running",
        "admission_observe",
        timeout_s=15,
        params={
            "targets": ["decode-0"],
            "fields": ["running"],
            "duration_s": 10,
            "until_op": "ge",
            "until_value": 1,
        },
    )
    case.step(
        "victim_running",
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
        "incomer",
        "admission_fire",
        params={
            "count": 1,
            "consume": "immediate",
            "input_len": 512,
            "output_len": 2,
            "priority": 70,
            "keys_per_request": 1,
            "spacing_s": 0,
            "request_timeout_s": 30,
        },
    )
    case.step(
        "incomer_admitted",
        "admission_check",
        params={
            "rows": output("incomer", "rows"),
            "metric": "admitted_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "incomer_code",
        "admission_check",
        params={
            "rows": output("incomer", "rows"),
            "metric": "all_schedule_code",
            "expected": 200,
            "op": "eq",
        },
    )
    case.step(
        "accepted_fast",
        "admission_check",
        params={
            "rows": output("incomer", "rows"),
            "metric": "schedule_latency_max",
            "expected": 3,
            "op": "lt",
        },
    )
    case.step(
        "victim_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("victim", "wave")},
    )
    case.step(
        "victim_unmolested",
        "admission_check",
        params={
            "rows": output("victim_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "incomer_done",
        "admission_wait",
        timeout_s=40,
        params={"wave": output("incomer", "wave")},
    )
    case.step(
        "incomer_completed",
        "admission_check",
        params={
            "rows": output("incomer_done", "rows"),
            "metric": "success_count",
            "expected": 1,
            "op": "eq",
        },
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
        params={"targets": ["prefill-0", "decode-0"]},
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "permit_released_without_preemption": {
        "build": permit_released_without_preemption,
        "profiles": ["batch-window"],
        "metadata": {},
    },
}
