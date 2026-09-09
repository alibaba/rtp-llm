"""Preserve separate ACK, member-execution, cleanup and recovery boundaries."""

from ..case_config import output

METADATA = {
    "description": "Preserve separate ACK, member-execution, cleanup and recovery boundaries.",
    "category": "status",
    "tags": ["protocol", "migration"],
    "id": "batch_ack_and_execution",
}

PROFILES = ["batch-window"]


def ack_partial(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "ledger_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step("ledger_before", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "partial_on",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {"k": 1},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "ledger_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("ledger_cohort", "requests")},
    )
    case.step("ledger_window", "status_sample", timeout_s=30, params={"duration_s": 8})
    case.step(
        "failed_member_released_promptly",
        "status_check",
        params={
            "snapshot": output("ledger_window", "snapshot"),
            "metric": "prefill_requests",
            "op": "le",
            "expected": 3,
            "aggregate": "max",
        },
    )
    case.step(
        "partial_off",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "ledger_drained",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "ledger_drained_scheduler",
        "status_check",
        params={
            "snapshot": output("ledger_drained", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ledger_drained_prefill_batches",
        "status_check",
        params={
            "snapshot": output("ledger_drained", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ledger_drained_decode_load",
        "status_check",
        params={
            "snapshot": output("ledger_drained", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "transient_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
        },
    )
    case.step(
        "transient_before", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "transient_partial_on",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {"k": 1},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "transient_code_on",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {"code": 13},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "transient_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("transient_cohort", "requests")},
    )
    case.step(
        "transient_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("transient_cohort", "requests")},
    )
    case.step(
        "transient_partial_off",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "transient_code_off",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "transient_policy",
        "status_outcomes",
        params={
            "requests": output("transient_cohort", "requests"),
            "timeout_or_success": True,
        },
    )
    case.step(
        "transient_retry_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 3},
    )
    case.step(
        "permanent_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "permanent_before", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "permanent_partial_on",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {"k": 1},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "permanent_code_on",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {"code": 8431},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "permanent_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("permanent_cohort", "requests")},
    )
    case.step(
        "permanent_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("permanent_cohort", "requests")},
    )
    case.step(
        "permanent_partial_off",
        "status_control",
        params={
            "fault": "enqueue_ack_partial_fail",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "permanent_code_off",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "permanent_member_isolation",
        "status_outcomes",
        params={
            "requests": output("permanent_cohort", "requests"),
            "success_min": 2,
            "failure_min": 1,
            "failure_max": 2,
            "error_code": 8431,
        },
    )
    case.step(
        "permanent_member_isolation_phase",
        "status_outcomes",
        params={
            "requests": output("permanent_cohort", "requests"),
            "failure_phase": "schedule",
        },
    )
    case.step(
        "permanent_after", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "permanent_dispatch_nonempty",
        "status_check",
        params={
            "snapshot": output("permanent_after", "snapshot"),
            "metric": "prefill_enqueue_rpc",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
            "baseline": output("permanent_before", "snapshot"),
        },
    )
    case.step(
        "permanent_no_retry",
        "status_check",
        params={
            "snapshot": output("permanent_after", "snapshot"),
            "metric": "prefill_enqueue_rpc",
            "op": "le",
            "expected": 2,
            "aggregate": "last",
            "baseline": output("permanent_before", "snapshot"),
        },
    )
    case.step(
        "final_drained",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "final_drained_scheduler",
        "status_check",
        params={
            "snapshot": output("final_drained", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "final_drained_prefill_batches",
        "status_check",
        params={
            "snapshot": output("final_drained", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "final_drained_decode_load",
        "status_check",
        params={
            "snapshot": output("final_drained", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step("final_health", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "master_http_200",
        "status_check",
        params={
            "snapshot": output("final_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
            "aggregate": "last",
        },
    )
    case.step("cleanup", "teardown")


def execution_partial(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "normal_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "normal_fail_on",
        "status_control",
        params={
            "fault": "prefill_async_partial_fail",
            "config": {"k": 1, "code": 8500},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "normal_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("normal_cohort", "requests")},
    )
    case.step(
        "normal_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("normal_cohort", "requests")},
    )
    case.step(
        "normal_fail_off",
        "status_control",
        params={
            "fault": "prefill_async_partial_fail",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "normal_execution_terminal",
        "status_outcomes",
        params={
            "requests": output("normal_cohort", "requests"),
            "success_min": 2,
            "failure_min": 1,
            "failure_max": 2,
            "error_code": 8500,
        },
    )
    case.step(
        "normal_execution_terminal_phase",
        "status_outcomes",
        params={
            "requests": output("normal_cohort", "requests"),
            "failure_phase": "execution",
        },
    )
    case.step(
        "normal_drained",
        "status_sample",
        timeout_s=135,
        params={
            "duration_s": 120,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "normal_drained_scheduler",
        "status_check",
        params={
            "snapshot": output("normal_drained", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "normal_drained_prefill_batches",
        "status_check",
        params={
            "snapshot": output("normal_drained", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "normal_drained_decode_load",
        "status_check",
        params={
            "snapshot": output("normal_drained", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "late_terminal_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "no_resurrection",
        "status_check",
        params={
            "snapshot": output("late_terminal_window", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("normal_drained", "snapshot"),
        },
    )
    case.step(
        "healthy_recovery",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "healthy_recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("healthy_recovery", "requests")},
    )
    case.step(
        "healthy_recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("healthy_recovery", "requests")},
    )
    case.step(
        "healthy_recovery_success",
        "status_outcomes",
        params={
            "requests": output("healthy_recovery", "requests"),
            "success_min": 4,
            "failure_max": 0,
        },
    )
    case.step(
        "serial_perf",
        "status_perf",
        params={"prefill_fixed_ms": 3000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "serial_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "serial_fail_on",
        "status_control",
        params={
            "fault": "prefill_async_partial_fail",
            "config": {"k": 1, "code": 8500},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "serial_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("serial_cohort", "requests")},
    )
    case.step(
        "serial_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("serial_cohort", "requests")},
    )
    case.step(
        "serial_fail_off",
        "status_control",
        params={
            "fault": "prefill_async_partial_fail",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "serial_perf_restore",
        "status_perf",
        params={"prefill_fixed_ms": 100, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "serial_execution_terminal",
        "status_outcomes",
        params={
            "requests": output("serial_cohort", "requests"),
            "success_min": 2,
            "failure_min": 1,
            "failure_max": 2,
            "error_code": 8500,
        },
    )
    case.step(
        "serial_execution_terminal_phase",
        "status_outcomes",
        params={
            "requests": output("serial_cohort", "requests"),
            "failure_phase": "execution",
        },
    )
    case.step(
        "serial_drained",
        "status_sample",
        timeout_s=135,
        params={
            "duration_s": 120,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "serial_drained_scheduler",
        "status_check",
        params={
            "snapshot": output("serial_drained", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "serial_drained_prefill_batches",
        "status_check",
        params={
            "snapshot": output("serial_drained", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "serial_drained_decode_load",
        "status_check",
        params={
            "snapshot": output("serial_drained", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step("final_health", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "master_http_200",
        "status_check",
        params={
            "snapshot": output("final_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
            "aggregate": "last",
        },
    )
    case.step("cleanup", "teardown")


def ack_multi_error(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "code_8431_cohort",
        "status_prepare",
        params={
            "count": 2,
            "concurrency": 2,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "code_8431_on",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {"code": 8431},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "code_8431_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("code_8431_cohort", "requests")},
    )
    case.step(
        "code_8431_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("code_8431_cohort", "requests")},
    )
    case.step(
        "code_8431_off",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "code_8431_passthrough",
        "status_outcomes",
        params={
            "requests": output("code_8431_cohort", "requests"),
            "failure_min": 2,
            "failure_max": 2,
            "error_code": 8431,
        },
    )
    case.step(
        "code_8431_passthrough_phase",
        "status_outcomes",
        params={
            "requests": output("code_8431_cohort", "requests"),
            "failure_phase": "schedule",
        },
    )
    case.step(
        "code_8510_cohort",
        "status_prepare",
        params={
            "count": 2,
            "concurrency": 2,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "code_8510_on",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {"code": 8510},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "code_8510_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("code_8510_cohort", "requests")},
    )
    case.step(
        "code_8510_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("code_8510_cohort", "requests")},
    )
    case.step(
        "code_8510_off",
        "status_control",
        params={
            "fault": "enqueue_ack_error_code",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "code_8510_passthrough",
        "status_outcomes",
        params={
            "requests": output("code_8510_cohort", "requests"),
            "failure_min": 2,
            "failure_max": 2,
            "error_code": 8510,
        },
    )
    case.step(
        "code_8510_passthrough_phase",
        "status_outcomes",
        params={
            "requests": output("code_8510_cohort", "requests"),
            "failure_phase": "schedule",
        },
    )
    case.step(
        "failed_batches_drained",
        "status_sample",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "failed_scheduler_drained",
        "status_check",
        params={
            "snapshot": output("failed_batches_drained", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "no_retry_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "no_resurrection",
        "status_check",
        params={
            "snapshot": output("no_retry_window", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step("final_health", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "master_http_200",
        "status_check",
        params={
            "snapshot": output("final_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
            "aggregate": "last",
        },
    )
    case.step("cleanup", "teardown")


def ack_drop(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "uncertain_cohort",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
            "observe_schedule_future_terminal": True,
        },
    )
    case.step(
        "drop_on",
        "status_control",
        params={
            "fault": "enqueue_ack_drop",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "uncertain_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("uncertain_cohort", "requests")},
    )
    case.step(
        "uncertain_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("uncertain_cohort", "requests")},
    )
    case.step(
        "drop_off",
        "status_control",
        params={
            "fault": "enqueue_ack_drop",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "fence_bound",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "scheduler", "op": "le", "value": 4},
        },
    )
    case.step(
        "fence_residue_bounded",
        "status_check",
        params={
            "snapshot": output("fence_bound", "snapshot"),
            "metric": "scheduler",
            "op": "le",
            "expected": 4,
            "aggregate": "last",
        },
    )
    case.step("fence_window", "status_sample", timeout_s=30, params={"duration_s": 8})
    case.step(
        "fence_residue_non_growing",
        "status_check",
        params={
            "snapshot": output("fence_window", "snapshot"),
            "metric": "scheduler",
            "op": "le",
            "expected": 0,
            "aggregate": "last",
            "baseline": output("fence_bound", "snapshot"),
        },
    )
    case.step(
        "ttl_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "quarantine_eventually_drains",
        "status_check",
        params={
            "snapshot": output("ttl_window", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step("final_health", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "master_http_200",
        "status_check",
        params={
            "snapshot": output("final_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
            "aggregate": "last",
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "ack_partial": {
        "build": ack_partial,
        "profiles": ["batch-window"],
        "metadata": {
            "findings": ["transient_policy.contract"],
        },
    },
    "execution_partial": {
        "build": execution_partial,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "ack_multi_error": {
        "build": ack_multi_error,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "ack_drop": {
        "build": ack_drop,
        "profiles": ["batch-window"],
        "metadata": {
            "findings": ["quarantine_eventually_drains.contract"],
        },
    },
}
