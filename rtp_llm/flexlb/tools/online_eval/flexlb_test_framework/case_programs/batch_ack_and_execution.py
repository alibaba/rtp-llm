"""Preserve separate ACK, member-execution, cleanup and recovery boundaries."""

from ..case_config import output


def ack_partial(case):
    case.step("setup", "setup", timeout_s=case.value("ack_partial.setup_timeout_s"))
    case.step(
        "ledger_cohort",
        "status_prepare",
        params=case.value("ack_partial.ledger_cohort"),
    )
    case.step(
        "ledger_before",
        "status_sample",
        timeout_s=case.value("ack_partial.ledger_before_timeout_s"),
        params=case.value("ack_partial.ledger_before"),
    )
    case.step(
        "partial_on",
        "status_control",
        params=case.value("ack_partial.partial_on"),
    )
    case.step(
        "ledger_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_partial.ledger_dispatch_timeout_s"),
        params={"requests": output("ledger_cohort", "requests")},
    )
    case.step(
        "ledger_window",
        "status_sample",
        timeout_s=case.value("ack_partial.ledger_window_timeout_s"),
        params=case.value("ack_partial.ledger_window"),
    )
    case.step(
        "failed_member_released_promptly",
        "status_check",
        params=case.params(
            "ack_partial.failed_member_released_promptly",
            {"snapshot": output("ledger_window", "snapshot")},
        ),
    )
    case.step(
        "partial_off",
        "status_control",
        params=case.value("ack_partial.partial_off"),
    )
    case.step(
        "ledger_drained",
        "status_sample",
        timeout_s=case.value("ack_partial.ledger_drained_timeout_s"),
        params=case.value("ack_partial.ledger_drained"),
    )
    case.step(
        "ledger_drained_scheduler",
        "status_check",
        params=case.params(
            "ack_partial.ledger_drained_scheduler",
            {"snapshot": output("ledger_drained", "snapshot")},
        ),
    )
    case.step(
        "ledger_drained_prefill_batches",
        "status_check",
        params=case.params(
            "ack_partial.ledger_drained_prefill_batches",
            {"snapshot": output("ledger_drained", "snapshot")},
        ),
    )
    case.step(
        "ledger_drained_decode_load",
        "status_check",
        params=case.params(
            "ack_partial.ledger_drained_decode_load",
            {"snapshot": output("ledger_drained", "snapshot")},
        ),
    )
    case.step(
        "transient_cohort",
        "status_prepare",
        params=case.value("ack_partial.transient_cohort"),
    )
    case.step(
        "transient_before",
        "status_sample",
        timeout_s=case.value("ack_partial.transient_before_timeout_s"),
        params=case.value("ack_partial.transient_before"),
    )
    case.step(
        "transient_partial_on",
        "status_control",
        params=case.value("ack_partial.transient_partial_on"),
    )
    case.step(
        "transient_code_on",
        "status_control",
        params=case.value("ack_partial.transient_code_on"),
    )
    case.step(
        "transient_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_partial.transient_dispatch_timeout_s"),
        params={"requests": output("transient_cohort", "requests")},
    )
    case.step(
        "transient_wait",
        "wait",
        timeout_s=case.value("ack_partial.transient_wait_timeout_s"),
        params={"requests": output("transient_cohort", "requests")},
    )
    case.step(
        "transient_partial_off",
        "status_control",
        params=case.value("ack_partial.transient_partial_off"),
    )
    case.step(
        "transient_code_off",
        "status_control",
        params=case.value("ack_partial.transient_code_off"),
    )
    case.step(
        "transient_policy",
        "status_outcomes",
        params=case.params(
            "ack_partial.transient_policy",
            {"requests": output("transient_cohort", "requests")},
        ),
    )
    case.step(
        "transient_retry_window",
        "status_sample",
        timeout_s=case.value("ack_partial.transient_retry_window_timeout_s"),
        params=case.value("ack_partial.transient_retry_window"),
    )
    case.step(
        "permanent_cohort",
        "status_prepare",
        params=case.value("ack_partial.permanent_cohort"),
    )
    case.step(
        "permanent_before",
        "status_sample",
        timeout_s=case.value("ack_partial.permanent_before_timeout_s"),
        params=case.value("ack_partial.permanent_before"),
    )
    case.step(
        "permanent_partial_on",
        "status_control",
        params=case.value("ack_partial.permanent_partial_on"),
    )
    case.step(
        "permanent_code_on",
        "status_control",
        params=case.value("ack_partial.permanent_code_on"),
    )
    case.step(
        "permanent_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_partial.permanent_dispatch_timeout_s"),
        params={"requests": output("permanent_cohort", "requests")},
    )
    case.step(
        "permanent_wait",
        "wait",
        timeout_s=case.value("ack_partial.permanent_wait_timeout_s"),
        params={"requests": output("permanent_cohort", "requests")},
    )
    case.step(
        "permanent_partial_off",
        "status_control",
        params=case.value("ack_partial.permanent_partial_off"),
    )
    case.step(
        "permanent_code_off",
        "status_control",
        params=case.value("ack_partial.permanent_code_off"),
    )
    case.step(
        "permanent_member_isolation",
        "status_outcomes",
        params=case.params(
            "ack_partial.permanent_member_isolation",
            {"requests": output("permanent_cohort", "requests")},
        ),
    )
    case.step(
        "permanent_member_isolation_phase",
        "status_outcomes",
        params=case.params(
            "ack_partial.permanent_member_isolation_phase",
            {"requests": output("permanent_cohort", "requests")},
        ),
    )
    case.step(
        "permanent_after",
        "status_sample",
        timeout_s=case.value("ack_partial.permanent_after_timeout_s"),
        params=case.value("ack_partial.permanent_after"),
    )
    case.step(
        "permanent_dispatch_nonempty",
        "status_check",
        params=case.params(
            "ack_partial.permanent_dispatch_nonempty",
            {
                "snapshot": output("permanent_after", "snapshot"),
                "baseline": output("permanent_before", "snapshot"),
            },
        ),
    )
    case.step(
        "permanent_no_retry",
        "status_check",
        params=case.params(
            "ack_partial.permanent_no_retry",
            {
                "snapshot": output("permanent_after", "snapshot"),
                "baseline": output("permanent_before", "snapshot"),
            },
        ),
    )
    case.step(
        "final_drained",
        "status_sample",
        timeout_s=case.value("ack_partial.final_drained_timeout_s"),
        params=case.value("ack_partial.final_drained"),
    )
    case.step(
        "final_drained_scheduler",
        "status_check",
        params=case.params(
            "ack_partial.final_drained_scheduler",
            {"snapshot": output("final_drained", "snapshot")},
        ),
    )
    case.step(
        "final_drained_prefill_batches",
        "status_check",
        params=case.params(
            "ack_partial.final_drained_prefill_batches",
            {"snapshot": output("final_drained", "snapshot")},
        ),
    )
    case.step(
        "final_drained_decode_load",
        "status_check",
        params=case.params(
            "ack_partial.final_drained_decode_load",
            {"snapshot": output("final_drained", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("ack_partial.final_health_timeout_s"),
        params=case.value("ack_partial.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "ack_partial.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def execution_partial(case):
    case.step(
        "setup", "setup", timeout_s=case.value("execution_partial.setup_timeout_s")
    )
    case.step(
        "normal_cohort",
        "status_prepare",
        params=case.value("execution_partial.normal_cohort"),
    )
    case.step(
        "normal_fail_on",
        "status_control",
        params=case.value("execution_partial.normal_fail_on"),
    )
    case.step(
        "normal_dispatch",
        "status_dispatch",
        timeout_s=case.value("execution_partial.normal_dispatch_timeout_s"),
        params={"requests": output("normal_cohort", "requests")},
    )
    case.step(
        "normal_wait",
        "wait",
        timeout_s=case.value("execution_partial.normal_wait_timeout_s"),
        params={"requests": output("normal_cohort", "requests")},
    )
    case.step(
        "normal_fail_off",
        "status_control",
        params=case.value("execution_partial.normal_fail_off"),
    )
    case.step(
        "normal_execution_terminal",
        "status_outcomes",
        params=case.params(
            "execution_partial.normal_execution_terminal",
            {"requests": output("normal_cohort", "requests")},
        ),
    )
    case.step(
        "normal_execution_terminal_phase",
        "status_outcomes",
        params=case.params(
            "execution_partial.normal_execution_terminal_phase",
            {"requests": output("normal_cohort", "requests")},
        ),
    )
    case.step(
        "normal_drained",
        "status_sample",
        timeout_s=case.value("execution_partial.normal_drained_timeout_s"),
        params=case.value("execution_partial.normal_drained"),
    )
    case.step(
        "normal_drained_scheduler",
        "status_check",
        params=case.params(
            "execution_partial.normal_drained_scheduler",
            {"snapshot": output("normal_drained", "snapshot")},
        ),
    )
    case.step(
        "normal_drained_prefill_batches",
        "status_check",
        params=case.params(
            "execution_partial.normal_drained_prefill_batches",
            {"snapshot": output("normal_drained", "snapshot")},
        ),
    )
    case.step(
        "normal_drained_decode_load",
        "status_check",
        params=case.params(
            "execution_partial.normal_drained_decode_load",
            {"snapshot": output("normal_drained", "snapshot")},
        ),
    )
    case.step(
        "late_terminal_window",
        "status_sample",
        timeout_s=case.value("execution_partial.late_terminal_window_timeout_s"),
        params=case.value("execution_partial.late_terminal_window"),
    )
    case.step(
        "no_resurrection",
        "status_check",
        params=case.params(
            "execution_partial.no_resurrection",
            {
                "snapshot": output("late_terminal_window", "snapshot"),
                "baseline": output("normal_drained", "snapshot"),
            },
        ),
    )
    case.step(
        "healthy_recovery",
        "status_prepare",
        params=case.value("execution_partial.healthy_recovery"),
    )
    case.step(
        "healthy_recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("execution_partial.healthy_recovery_dispatch_timeout_s"),
        params={"requests": output("healthy_recovery", "requests")},
    )
    case.step(
        "healthy_recovery_wait",
        "wait",
        timeout_s=case.value("execution_partial.healthy_recovery_wait_timeout_s"),
        params={"requests": output("healthy_recovery", "requests")},
    )
    case.step(
        "healthy_recovery_success",
        "status_outcomes",
        params=case.params(
            "execution_partial.healthy_recovery_success",
            {"requests": output("healthy_recovery", "requests")},
        ),
    )
    case.step(
        "serial_perf",
        "status_perf",
        params=case.value("execution_partial.serial_perf"),
    )
    case.step(
        "serial_cohort",
        "status_prepare",
        params=case.value("execution_partial.serial_cohort"),
    )
    case.step(
        "serial_fail_on",
        "status_control",
        params=case.value("execution_partial.serial_fail_on"),
    )
    case.step(
        "serial_dispatch",
        "status_dispatch",
        timeout_s=case.value("execution_partial.serial_dispatch_timeout_s"),
        params={"requests": output("serial_cohort", "requests")},
    )
    case.step(
        "serial_wait",
        "wait",
        timeout_s=case.value("execution_partial.serial_wait_timeout_s"),
        params={"requests": output("serial_cohort", "requests")},
    )
    case.step(
        "serial_fail_off",
        "status_control",
        params=case.value("execution_partial.serial_fail_off"),
    )
    case.step(
        "serial_perf_restore",
        "status_perf",
        params=case.value("execution_partial.serial_perf_restore"),
    )
    case.step(
        "serial_execution_terminal",
        "status_outcomes",
        params=case.params(
            "execution_partial.serial_execution_terminal",
            {"requests": output("serial_cohort", "requests")},
        ),
    )
    case.step(
        "serial_execution_terminal_phase",
        "status_outcomes",
        params=case.params(
            "execution_partial.serial_execution_terminal_phase",
            {"requests": output("serial_cohort", "requests")},
        ),
    )
    case.step(
        "serial_drained",
        "status_sample",
        timeout_s=case.value("execution_partial.serial_drained_timeout_s"),
        params=case.value("execution_partial.serial_drained"),
    )
    case.step(
        "serial_drained_scheduler",
        "status_check",
        params=case.params(
            "execution_partial.serial_drained_scheduler",
            {"snapshot": output("serial_drained", "snapshot")},
        ),
    )
    case.step(
        "serial_drained_prefill_batches",
        "status_check",
        params=case.params(
            "execution_partial.serial_drained_prefill_batches",
            {"snapshot": output("serial_drained", "snapshot")},
        ),
    )
    case.step(
        "serial_drained_decode_load",
        "status_check",
        params=case.params(
            "execution_partial.serial_drained_decode_load",
            {"snapshot": output("serial_drained", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("execution_partial.final_health_timeout_s"),
        params=case.value("execution_partial.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "execution_partial.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def ack_multi_error(case):
    case.step("setup", "setup", timeout_s=case.value("ack_multi_error.setup_timeout_s"))
    case.step(
        "code_8431_cohort",
        "status_prepare",
        params=case.value("ack_multi_error.code_8431_cohort"),
    )
    case.step(
        "code_8431_on",
        "status_control",
        params=case.value("ack_multi_error.code_8431_on"),
    )
    case.step(
        "code_8431_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_multi_error.code_8431_dispatch_timeout_s"),
        params={"requests": output("code_8431_cohort", "requests")},
    )
    case.step(
        "code_8431_wait",
        "wait",
        timeout_s=case.value("ack_multi_error.code_8431_wait_timeout_s"),
        params={"requests": output("code_8431_cohort", "requests")},
    )
    case.step(
        "code_8431_off",
        "status_control",
        params=case.value("ack_multi_error.code_8431_off"),
    )
    case.step(
        "code_8431_passthrough",
        "status_outcomes",
        params=case.params(
            "ack_multi_error.code_8431_passthrough",
            {"requests": output("code_8431_cohort", "requests")},
        ),
    )
    case.step(
        "code_8431_passthrough_phase",
        "status_outcomes",
        params=case.params(
            "ack_multi_error.code_8431_passthrough_phase",
            {"requests": output("code_8431_cohort", "requests")},
        ),
    )
    case.step(
        "code_8510_cohort",
        "status_prepare",
        params=case.value("ack_multi_error.code_8510_cohort"),
    )
    case.step(
        "code_8510_on",
        "status_control",
        params=case.value("ack_multi_error.code_8510_on"),
    )
    case.step(
        "code_8510_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_multi_error.code_8510_dispatch_timeout_s"),
        params={"requests": output("code_8510_cohort", "requests")},
    )
    case.step(
        "code_8510_wait",
        "wait",
        timeout_s=case.value("ack_multi_error.code_8510_wait_timeout_s"),
        params={"requests": output("code_8510_cohort", "requests")},
    )
    case.step(
        "code_8510_off",
        "status_control",
        params=case.value("ack_multi_error.code_8510_off"),
    )
    case.step(
        "code_8510_passthrough",
        "status_outcomes",
        params=case.params(
            "ack_multi_error.code_8510_passthrough",
            {"requests": output("code_8510_cohort", "requests")},
        ),
    )
    case.step(
        "code_8510_passthrough_phase",
        "status_outcomes",
        params=case.params(
            "ack_multi_error.code_8510_passthrough_phase",
            {"requests": output("code_8510_cohort", "requests")},
        ),
    )
    case.step(
        "failed_batches_drained",
        "status_sample",
        timeout_s=case.value("ack_multi_error.failed_batches_drained_timeout_s"),
        params=case.value("ack_multi_error.failed_batches_drained"),
    )
    case.step(
        "failed_scheduler_drained",
        "status_check",
        params=case.params(
            "ack_multi_error.failed_scheduler_drained",
            {"snapshot": output("failed_batches_drained", "snapshot")},
        ),
    )
    case.step(
        "no_retry_window",
        "status_sample",
        timeout_s=case.value("ack_multi_error.no_retry_window_timeout_s"),
        params=case.value("ack_multi_error.no_retry_window"),
    )
    case.step(
        "no_resurrection",
        "status_check",
        params=case.params(
            "ack_multi_error.no_resurrection",
            {"snapshot": output("no_retry_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("ack_multi_error.final_health_timeout_s"),
        params=case.value("ack_multi_error.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "ack_multi_error.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def ack_drop(case):
    case.step("setup", "setup", timeout_s=case.value("ack_drop.setup_timeout_s"))
    case.step(
        "uncertain_cohort",
        "status_prepare",
        params=case.value("ack_drop.uncertain_cohort"),
    )
    case.step(
        "drop_on",
        "status_control",
        params=case.value("ack_drop.drop_on"),
    )
    case.step(
        "uncertain_dispatch",
        "status_dispatch",
        timeout_s=case.value("ack_drop.uncertain_dispatch_timeout_s"),
        params={"requests": output("uncertain_cohort", "requests")},
    )
    case.step(
        "uncertain_wait",
        "wait",
        timeout_s=case.value("ack_drop.uncertain_wait_timeout_s"),
        params={"requests": output("uncertain_cohort", "requests")},
    )
    case.step(
        "drop_off",
        "status_control",
        params=case.value("ack_drop.drop_off"),
    )
    case.step(
        "fence_bound",
        "status_sample",
        timeout_s=case.value("ack_drop.fence_bound_timeout_s"),
        params=case.value("ack_drop.fence_bound"),
    )
    case.step(
        "fence_residue_bounded",
        "status_check",
        params=case.params(
            "ack_drop.fence_residue_bounded",
            {"snapshot": output("fence_bound", "snapshot")},
        ),
    )
    case.step(
        "fence_window",
        "status_sample",
        timeout_s=case.value("ack_drop.fence_window_timeout_s"),
        params=case.value("ack_drop.fence_window"),
    )
    case.step(
        "fence_residue_non_growing",
        "status_check",
        params=case.params(
            "ack_drop.fence_residue_non_growing",
            {
                "snapshot": output("fence_window", "snapshot"),
                "baseline": output("fence_bound", "snapshot"),
            },
        ),
    )
    case.step(
        "ttl_window",
        "status_sample",
        timeout_s=case.value("ack_drop.ttl_window_timeout_s"),
        params=case.value("ack_drop.ttl_window"),
    )
    case.step(
        "quarantine_eventually_drains",
        "status_check",
        params=case.params(
            "ack_drop.quarantine_eventually_drains",
            {"snapshot": output("ttl_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("ack_drop.final_health_timeout_s"),
        params=case.value("ack_drop.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "ack_drop.master_http_200", {"snapshot": output("final_health", "snapshot")}
        ),
    )
    case.step("cleanup", "teardown")
