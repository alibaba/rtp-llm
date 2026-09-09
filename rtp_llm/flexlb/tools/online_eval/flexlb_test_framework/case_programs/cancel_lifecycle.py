"""Twelve cancellation lifecycle contracts with explicit client, Master and engine boundaries."""

from ..case_config import output


def basic_batch(case):
    case.step("setup", "setup", timeout_s=case.value("basic_batch.setup_timeout_s"))
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("basic_batch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("basic_batch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("basic_batch.first_output_window_timeout_s"),
        params=case.params(
            "basic_batch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_before_cancel",
        "cancel_check",
        params=case.params(
            "basic_batch.first_output_before_cancel",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("basic_batch.first_master_cancel_timeout_s"),
        params=case.params(
            "basic_batch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "basic_batch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("basic_batch.termination_window_timeout_s"),
        params=case.params(
            "basic_batch.termination_window", {"requests": output("target", "requests")}
        ),
    )
    case.step(
        "stream_terminated",
        "cancel_check",
        params=case.params(
            "basic_batch.stream_terminated",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "engine_receipt_window",
        "cancel_observe",
        timeout_s=case.value("basic_batch.engine_receipt_window_timeout_s"),
        params=case.params(
            "basic_batch.engine_receipt_window",
            {
                "requests": output("target", "requests"),
                "since": output("first_master_cancel", "snapshot"),
            },
        ),
    )
    case.step(
        "master_cancel_does_not_cancel_engine",
        "cancel_check",
        params=case.params(
            "basic_batch.master_cancel_does_not_cancel_engine",
            {"snapshot": output("engine_receipt_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("basic_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("basic_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("basic_batch.recovery_window_timeout_s"),
        params=case.params(
            "basic_batch.recovery_window", {"requests": output("recovery", "requests")}
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "basic_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("basic_batch.closing_drain_timeout_s"),
        params=case.value("basic_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "basic_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "basic_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "basic_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "no_forward_final",
        "cancel_observe",
        timeout_s=case.value("basic_batch.no_forward_final_timeout_s"),
        params=case.value("basic_batch.no_forward_final"),
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params=case.params(
            "basic_batch.ordinary_cancel_never_forwarded",
            {"snapshot": output("no_forward_final", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def idempotent_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("idempotent_batch.setup_timeout_s")
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("idempotent_batch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("idempotent_batch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.first_output_window_timeout_s"),
        params=case.params(
            "idempotent_batch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_before_cancel",
        "cancel_check",
        params=case.params(
            "idempotent_batch.first_output_before_cancel",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_count_baseline",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.cancel_count_baseline_timeout_s"),
        params=case.params(
            "idempotent_batch.cancel_count_baseline",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_batch.first_master_cancel_timeout_s"),
        params=case.params(
            "idempotent_batch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_batch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "first_engine_receipt",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.first_engine_receipt_timeout_s"),
        params=case.params(
            "idempotent_batch.first_engine_receipt",
            {
                "requests": output("target", "requests"),
                "since": output("first_master_cancel", "snapshot"),
            },
        ),
    )
    case.step(
        "master_cancel_does_not_cancel_engine",
        "cancel_check",
        params=case.params(
            "idempotent_batch.master_cancel_does_not_cancel_engine",
            {"snapshot": output("first_engine_receipt", "snapshot")},
        ),
    )
    case.step(
        "first_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "idempotent_batch.first_cancel_not_forwarded",
            {
                "snapshot": output("first_engine_receipt", "snapshot"),
                "baseline": output("cancel_count_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "second_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_batch.second_master_cancel_timeout_s"),
        params=case.params(
            "idempotent_batch.second_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "second_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_batch.second_master_rpc_ok",
            {"snapshot": output("second_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.termination_window_timeout_s"),
        params=case.params(
            "idempotent_batch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "stream_terminated",
        "cancel_check",
        params=case.params(
            "idempotent_batch.stream_terminated",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("idempotent_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("idempotent_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.recovery_window_timeout_s"),
        params=case.params(
            "idempotent_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "idempotent_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "after_second_cancel",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.after_second_cancel_timeout_s"),
        params=case.params(
            "idempotent_batch.after_second_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "second_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "idempotent_batch.second_cancel_not_forwarded",
            {
                "snapshot": output("after_second_cancel", "snapshot"),
                "baseline": output("first_engine_receipt", "snapshot"),
            },
        ),
    )
    case.step(
        "closing_drain_observed",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.closing_drain_observed_timeout_s"),
        params=case.value("idempotent_batch.closing_drain_observed"),
    )
    case.step(
        "no_forward_final",
        "cancel_observe",
        timeout_s=case.value("idempotent_batch.no_forward_final_timeout_s"),
        params=case.value("idempotent_batch.no_forward_final"),
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params=case.params(
            "idempotent_batch.ordinary_cancel_never_forwarded",
            {"snapshot": output("no_forward_final", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def after_terminal_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("after_terminal_batch.setup_timeout_s")
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("after_terminal_batch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("after_terminal_batch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_batch.completion_window_timeout_s"),
        params=case.params(
            "after_terminal_batch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.completed_before_cancel",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("after_terminal_batch.first_master_cancel_timeout_s"),
        params=case.params(
            "after_terminal_batch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_batch.termination_window_timeout_s"),
        params=case.params(
            "after_terminal_batch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("after_terminal_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("after_terminal_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_batch.recovery_window_timeout_s"),
        params=case.params(
            "after_terminal_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "terminal_after_cancel",
        "cancel_observe",
        timeout_s=case.value("after_terminal_batch.terminal_after_cancel_timeout_s"),
        params=case.params(
            "after_terminal_batch.terminal_after_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_not_rewritten_as_cancelled",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.terminal_not_rewritten_as_cancelled",
            {"snapshot": output("terminal_after_cancel", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("after_terminal_batch.closing_drain_timeout_s"),
        params=case.value("after_terminal_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "after_terminal_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def basic_nonbatch(case):
    case.step("setup", "setup", timeout_s=case.value("basic_nonbatch.setup_timeout_s"))
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("basic_nonbatch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("basic_nonbatch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("basic_nonbatch.first_output_window_timeout_s"),
        params=case.params(
            "basic_nonbatch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_before_cancel",
        "cancel_check",
        params=case.params(
            "basic_nonbatch.first_output_before_cancel",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("basic_nonbatch.first_master_cancel_timeout_s"),
        params=case.params(
            "basic_nonbatch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=case.value("basic_nonbatch.first_worker_cancel_timeout_s"),
        params=case.params(
            "basic_nonbatch.first_worker_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "basic_nonbatch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params=case.params(
            "basic_nonbatch.first_worker_rpc_ok",
            {"snapshot": output("first_worker_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("basic_nonbatch.termination_window_timeout_s"),
        params=case.params(
            "basic_nonbatch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "stream_terminated",
        "cancel_check",
        params=case.params(
            "basic_nonbatch.stream_terminated",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "engine_receipt_window",
        "cancel_observe",
        timeout_s=case.value("basic_nonbatch.engine_receipt_window_timeout_s"),
        params=case.params(
            "basic_nonbatch.engine_receipt_window",
            {
                "requests": output("target", "requests"),
                "since": output("first_master_cancel", "snapshot"),
            },
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("basic_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("basic_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("basic_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "basic_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "basic_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def idempotent_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("idempotent_nonbatch.setup_timeout_s")
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("idempotent_nonbatch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("idempotent_nonbatch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.first_output_window_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_before_cancel",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.first_output_before_cancel",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_count_baseline",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.cancel_count_baseline_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.cancel_count_baseline",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_nonbatch.first_master_cancel_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_nonbatch.first_worker_cancel_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.first_worker_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.first_worker_rpc_ok",
            {"snapshot": output("first_worker_cancel", "snapshot")},
        ),
    )
    case.step(
        "first_engine_receipt",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.first_engine_receipt_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.first_engine_receipt",
            {
                "requests": output("target", "requests"),
                "since": output("first_master_cancel", "snapshot"),
            },
        ),
    )
    case.step(
        "second_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_nonbatch.second_master_cancel_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.second_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "second_worker_cancel",
        "cancel_rpc",
        timeout_s=case.value("idempotent_nonbatch.second_worker_cancel_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.second_worker_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "second_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.second_master_rpc_ok",
            {"snapshot": output("second_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "second_worker_rpc_ok",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.second_worker_rpc_ok",
            {"snapshot": output("second_worker_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.termination_window_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "stream_terminated",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.stream_terminated",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("idempotent_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("idempotent_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "idempotent_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "after_second_cancel",
        "cancel_observe",
        timeout_s=case.value("idempotent_nonbatch.after_second_cancel_timeout_s"),
        params=case.params(
            "idempotent_nonbatch.after_second_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step("cleanup", "teardown")


def after_terminal_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("after_terminal_nonbatch.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("after_terminal_nonbatch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("after_terminal_nonbatch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_nonbatch.completion_window_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params=case.params(
            "after_terminal_nonbatch.completed_before_cancel",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("after_terminal_nonbatch.first_master_cancel_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.first_master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=case.value("after_terminal_nonbatch.first_worker_cancel_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.first_worker_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params=case.params(
            "after_terminal_nonbatch.first_master_rpc_ok",
            {"snapshot": output("first_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params=case.params(
            "after_terminal_nonbatch.first_worker_rpc_ok",
            {"snapshot": output("first_worker_cancel", "snapshot")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_nonbatch.termination_window_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("after_terminal_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("after_terminal_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("after_terminal_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "after_terminal_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "terminal_after_cancel",
        "cancel_observe",
        timeout_s=case.value("after_terminal_nonbatch.terminal_after_cancel_timeout_s"),
        params=case.params(
            "after_terminal_nonbatch.terminal_after_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_not_rewritten_as_cancelled",
        "cancel_check",
        params=case.params(
            "after_terminal_nonbatch.terminal_not_rewritten_as_cancelled",
            {"snapshot": output("terminal_after_cancel", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unknown_rid(case):
    case.step("setup", "setup", timeout_s=case.value("unknown_rid.setup_timeout_s"))
    case.step(
        "clean_baseline",
        "cancel_observe",
        timeout_s=case.value("unknown_rid.clean_baseline_timeout_s"),
        params=case.value("unknown_rid.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "cancel_check",
        params=case.params(
            "unknown_rid.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "cancel_check",
        params=case.params(
            "unknown_rid.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "cancel_check",
        params=case.params(
            "unknown_rid.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step("unknown", "cancel_prepare", params=case.value("unknown_rid.unknown"))
    case.step(
        "unknown_master_cancel",
        "cancel_rpc",
        timeout_s=case.value("unknown_rid.unknown_master_cancel_timeout_s"),
        params=case.params(
            "unknown_rid.unknown_master_cancel",
            {"requests": output("unknown", "requests")},
        ),
    )
    case.step(
        "unknown_is_typed_not_found",
        "cancel_check",
        params=case.params(
            "unknown_rid.unknown_is_typed_not_found",
            {"snapshot": output("unknown_master_cancel", "snapshot")},
        ),
    )
    case.step(
        "after_unknown",
        "cancel_observe",
        timeout_s=case.value("unknown_rid.after_unknown_timeout_s"),
        params=case.value("unknown_rid.after_unknown"),
    )
    case.step(
        "unknown_does_not_mutate_ledger",
        "cancel_check",
        params=case.params(
            "unknown_rid.unknown_does_not_mutate_ledger",
            {
                "snapshot": output("after_unknown", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step("cleanup", "teardown")


def anomaly_path_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("anomaly_path_batch.setup_timeout_s")
    )
    case.step(
        "target", "cancel_prepare", params=case.value("anomaly_path_batch.target")
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("anomaly_path_batch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_batch.first_output_window_timeout_s"),
        params=case.params(
            "anomaly_path_batch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=case.value("anomaly_path_batch.cancel_master_timeout_s"),
        params=case.params(
            "anomaly_path_batch.cancel_master",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_batch.termination_window_timeout_s"),
        params=case.params(
            "anomaly_path_batch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "termination_ended",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.termination_ended",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("anomaly_path_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("anomaly_path_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_batch.recovery_window_timeout_s"),
        params=case.params(
            "anomaly_path_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_batch.closing_drain_timeout_s"),
        params=case.value("anomaly_path_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "anomaly_path_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def sibling_isolation_batch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("sibling_isolation_batch.setup_timeout_s"),
    )
    case.step(
        "a",
        "cancel_prepare",
        params=case.value("sibling_isolation_batch.a"),
    )
    case.step(
        "b",
        "cancel_prepare",
        params=case.value("sibling_isolation_batch.b"),
    )
    case.step(
        "c",
        "cancel_prepare",
        params=case.value("sibling_isolation_batch.c"),
    )
    case.step(
        "dispatch_siblings",
        "cancel_dispatch_group",
        timeout_s=case.value("sibling_isolation_batch.dispatch_siblings_timeout_s"),
        params={
            "cohorts": [
                output("a", "requests"),
                output("b", "requests"),
                output("c", "requests"),
            ]
        },
    )
    case.step(
        "a_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_batch.a_open_timeout_s"),
        params={"requests": output("a", "requests")},
    )
    case.step(
        "b_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_batch.b_open_timeout_s"),
        params={"requests": output("b", "requests")},
    )
    case.step(
        "c_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_batch.c_open_timeout_s"),
        params={"requests": output("c", "requests")},
    )
    case.step(
        "a_first_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.a_first_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.a_first_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_first_received",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.a_first_received",
            {"snapshot": output("a_first_window", "snapshot")},
        ),
    )
    case.step(
        "c_first_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.c_first_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.c_first_window",
            {"requests": output("c", "requests")},
        ),
    )
    case.step(
        "c_first_received",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.c_first_received",
            {"snapshot": output("c_first_window", "snapshot")},
        ),
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("sibling_isolation_batch.b_cancel_master_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.b_cancel_master",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.b_termination_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.b_termination_window",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.b_termination_ended",
            {"snapshot": output("b_termination_window", "snapshot")},
        ),
    )
    case.step(
        "isolation_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.isolation_window_timeout_s"),
        params=case.value("sibling_isolation_batch.isolation_window"),
    )
    case.step(
        "cancelled_slot_removed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.cancelled_slot_removed",
            {"snapshot": output("isolation_window", "snapshot")},
        ),
    )
    case.step(
        "a_completion_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.a_completion_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.a_completion_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_completion_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.a_completion_ended",
            {"snapshot": output("a_completion_window", "snapshot")},
        ),
    )
    case.step(
        "a_completed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.a_completed",
            {"snapshot": output("a_completion_window", "snapshot")},
        ),
    )
    case.step(
        "c_completion_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.c_completion_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.c_completion_window",
            {"requests": output("c", "requests")},
        ),
    )
    case.step(
        "c_completion_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.c_completion_ended",
            {"snapshot": output("c_completion_window", "snapshot")},
        ),
    )
    case.step(
        "c_completed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.c_completed",
            {"snapshot": output("c_completion_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("sibling_isolation_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("sibling_isolation_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.recovery_window_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "b_state",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.b_state_timeout_s"),
        params=case.params(
            "sibling_isolation_batch.b_state", {"requests": output("b", "requests")}
        ),
    )
    case.step(
        "b_completes_independently",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.b_completes_independently",
            {"snapshot": output("b_state", "snapshot")},
        ),
    )
    case.step(
        "b_engine_not_cancelled",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.b_engine_not_cancelled",
            {"snapshot": output("b_state", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.closing_drain_timeout_s"),
        params=case.value("sibling_isolation_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "no_forward_final",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_batch.no_forward_final_timeout_s"),
        params=case.value("sibling_isolation_batch.no_forward_final"),
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params=case.params(
            "sibling_isolation_batch.ordinary_cancel_never_forwarded",
            {"snapshot": output("no_forward_final", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def phase_timing_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("phase_timing_batch.setup_timeout_s")
    )
    case.step("a", "cancel_prepare", params=case.value("phase_timing_batch.a"))
    case.step(
        "a_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_batch.a_dispatch_timeout_s"),
        params={"requests": output("a", "requests")},
    )
    case.step(
        "a_prefill_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.a_prefill_window_timeout_s"),
        params=case.params(
            "phase_timing_batch.a_prefill_window", {"requests": output("a", "requests")}
        ),
    )
    case.step(
        "a_still_prefill",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.a_still_prefill",
            {"snapshot": output("a_prefill_window", "snapshot")},
        ),
    )
    case.step(
        "a_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_batch.a_cancel_master_timeout_s"),
        params=case.params(
            "phase_timing_batch.a_cancel_master", {"requests": output("a", "requests")}
        ),
    )
    case.step(
        "a_termination_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.a_termination_window_timeout_s"),
        params=case.params(
            "phase_timing_batch.a_termination_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_termination_ended",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.a_termination_ended",
            {"snapshot": output("a_termination_window", "snapshot")},
        ),
    )
    case.step(
        "a_receipt",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.a_receipt_timeout_s"),
        params=case.params(
            "phase_timing_batch.a_receipt",
            {
                "requests": output("a", "requests"),
                "since": output("a_cancel_master", "snapshot"),
            },
        ),
    )
    case.step(
        "a_master_cancel_keeps_engine_running",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.a_master_cancel_keeps_engine_running",
            {"snapshot": output("a_receipt", "snapshot")},
        ),
    )
    case.step("b", "cancel_prepare", params=case.value("phase_timing_batch.b"))
    case.step(
        "b_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_batch.b_dispatch_timeout_s"),
        params={"requests": output("b", "requests")},
    )
    case.step(
        "b_first_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.b_first_window_timeout_s"),
        params=case.params(
            "phase_timing_batch.b_first_window", {"requests": output("b", "requests")}
        ),
    )
    case.step(
        "b_first_received",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.b_first_received",
            {"snapshot": output("b_first_window", "snapshot")},
        ),
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_batch.b_cancel_master_timeout_s"),
        params=case.params(
            "phase_timing_batch.b_cancel_master", {"requests": output("b", "requests")}
        ),
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.b_termination_window_timeout_s"),
        params=case.params(
            "phase_timing_batch.b_termination_window",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.b_termination_ended",
            {"snapshot": output("b_termination_window", "snapshot")},
        ),
    )
    case.step(
        "b_receipt",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.b_receipt_timeout_s"),
        params=case.params(
            "phase_timing_batch.b_receipt",
            {
                "requests": output("b", "requests"),
                "since": output("b_cancel_master", "snapshot"),
            },
        ),
    )
    case.step(
        "b_master_cancel_keeps_engine_running",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.b_master_cancel_keeps_engine_running",
            {"snapshot": output("b_receipt", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("phase_timing_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.recovery_window_timeout_s"),
        params=case.params(
            "phase_timing_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.closing_drain_timeout_s"),
        params=case.value("phase_timing_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "no_forward_final",
        "cancel_observe",
        timeout_s=case.value("phase_timing_batch.no_forward_final_timeout_s"),
        params=case.value("phase_timing_batch.no_forward_final"),
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params=case.params(
            "phase_timing_batch.ordinary_cancel_never_forwarded",
            {"snapshot": output("no_forward_final", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def deadline_exempt_inflight_batch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("deadline_exempt_inflight_batch.setup_timeout_s"),
    )
    case.step(
        "slow_enqueue",
        "engine_inject",
        params=case.value("deadline_exempt_inflight_batch.slow_enqueue"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("deadline_exempt_inflight_batch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "deadline_exempt_inflight_batch.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "deadline_exempt_inflight_batch.completion_window_timeout_s"
        ),
        params=case.params(
            "deadline_exempt_inflight_batch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completion_ended",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.completion_ended",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "inflight_exempt_completes",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.inflight_exempt_completes",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_state",
        "cancel_observe",
        timeout_s=case.value("deadline_exempt_inflight_batch.cancel_state_timeout_s"),
        params=case.params(
            "deadline_exempt_inflight_batch.cancel_state",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "deadline_did_not_cancel_engine",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.deadline_did_not_cancel_engine",
            {"snapshot": output("cancel_state", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("deadline_exempt_inflight_batch.closing_drain_timeout_s"),
        params=case.value("deadline_exempt_inflight_batch.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("deadline_exempt_inflight_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "deadline_exempt_inflight_batch.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "deadline_exempt_inflight_batch.recovery_window_timeout_s"
        ),
        params=case.params(
            "deadline_exempt_inflight_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step("cleanup", "teardown")


def anomaly_path_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("anomaly_path_nonbatch.setup_timeout_s")
    )
    case.step(
        "target", "cancel_prepare", params=case.value("anomaly_path_nonbatch.target")
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("anomaly_path_nonbatch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_nonbatch.first_output_window_timeout_s"),
        params=case.params(
            "anomaly_path_nonbatch.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "anomaly_path_nonbatch.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=case.value("anomaly_path_nonbatch.cancel_master_timeout_s"),
        params=case.params(
            "anomaly_path_nonbatch.cancel_master",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "cancel_worker",
        "cancel_rpc",
        timeout_s=case.value("anomaly_path_nonbatch.cancel_worker_timeout_s"),
        params=case.params(
            "anomaly_path_nonbatch.cancel_worker",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "termination_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_nonbatch.termination_window_timeout_s"),
        params=case.params(
            "anomaly_path_nonbatch.termination_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "termination_ended",
        "cancel_check",
        params=case.params(
            "anomaly_path_nonbatch.termination_ended",
            {"snapshot": output("termination_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("anomaly_path_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("anomaly_path_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("anomaly_path_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "anomaly_path_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "anomaly_path_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def sibling_isolation_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("sibling_isolation_nonbatch.setup_timeout_s"),
    )
    case.step(
        "a",
        "cancel_prepare",
        params=case.value("sibling_isolation_nonbatch.a"),
    )
    case.step(
        "b",
        "cancel_prepare",
        params=case.value("sibling_isolation_nonbatch.b"),
    )
    case.step(
        "c",
        "cancel_prepare",
        params=case.value("sibling_isolation_nonbatch.c"),
    )
    case.step(
        "dispatch_siblings",
        "cancel_dispatch_group",
        timeout_s=case.value("sibling_isolation_nonbatch.dispatch_siblings_timeout_s"),
        params={
            "cohorts": [
                output("a", "requests"),
                output("b", "requests"),
                output("c", "requests"),
            ]
        },
    )
    case.step(
        "a_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_nonbatch.a_open_timeout_s"),
        params={"requests": output("a", "requests")},
    )
    case.step(
        "b_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_nonbatch.b_open_timeout_s"),
        params={"requests": output("b", "requests")},
    )
    case.step(
        "c_open",
        "cancel_open",
        timeout_s=case.value("sibling_isolation_nonbatch.c_open_timeout_s"),
        params={"requests": output("c", "requests")},
    )
    case.step(
        "a_first_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_nonbatch.a_first_window_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.a_first_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_first_received",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.a_first_received",
            {"snapshot": output("a_first_window", "snapshot")},
        ),
    )
    case.step(
        "c_first_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_nonbatch.c_first_window_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.c_first_window",
            {"requests": output("c", "requests")},
        ),
    )
    case.step(
        "c_first_received",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.c_first_received",
            {"snapshot": output("c_first_window", "snapshot")},
        ),
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("sibling_isolation_nonbatch.b_cancel_master_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.b_cancel_master",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_cancel_worker",
        "cancel_rpc",
        timeout_s=case.value("sibling_isolation_nonbatch.b_cancel_worker_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.b_cancel_worker",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=case.value(
            "sibling_isolation_nonbatch.b_termination_window_timeout_s"
        ),
        params=case.params(
            "sibling_isolation_nonbatch.b_termination_window",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.b_termination_ended",
            {"snapshot": output("b_termination_window", "snapshot")},
        ),
    )
    case.step(
        "a_completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "sibling_isolation_nonbatch.a_completion_window_timeout_s"
        ),
        params=case.params(
            "sibling_isolation_nonbatch.a_completion_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_completion_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.a_completion_ended",
            {"snapshot": output("a_completion_window", "snapshot")},
        ),
    )
    case.step(
        "a_completed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.a_completed",
            {"snapshot": output("a_completion_window", "snapshot")},
        ),
    )
    case.step(
        "c_completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "sibling_isolation_nonbatch.c_completion_window_timeout_s"
        ),
        params=case.params(
            "sibling_isolation_nonbatch.c_completion_window",
            {"requests": output("c", "requests")},
        ),
    )
    case.step(
        "c_completion_ended",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.c_completion_ended",
            {"snapshot": output("c_completion_window", "snapshot")},
        ),
    )
    case.step(
        "c_completed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.c_completed",
            {"snapshot": output("c_completion_window", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("sibling_isolation_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("sibling_isolation_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "b_state",
        "cancel_observe",
        timeout_s=case.value("sibling_isolation_nonbatch.b_state_timeout_s"),
        params=case.params(
            "sibling_isolation_nonbatch.b_state", {"requests": output("b", "requests")}
        ),
    )
    case.step(
        "b_not_completed",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.b_not_completed",
            {"snapshot": output("b_state", "snapshot")},
        ),
    )
    case.step(
        "b_engine_cancelled",
        "cancel_check",
        params=case.params(
            "sibling_isolation_nonbatch.b_engine_cancelled",
            {"snapshot": output("b_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def phase_timing_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("phase_timing_nonbatch.setup_timeout_s")
    )
    case.step("a", "cancel_prepare", params=case.value("phase_timing_nonbatch.a"))
    case.step(
        "a_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_nonbatch.a_dispatch_timeout_s"),
        params={"requests": output("a", "requests")},
    )
    case.step(
        "a_prefill_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.a_prefill_window_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.a_prefill_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_still_prefill",
        "cancel_check",
        params=case.params(
            "phase_timing_nonbatch.a_still_prefill",
            {"snapshot": output("a_prefill_window", "snapshot")},
        ),
    )
    case.step(
        "a_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_nonbatch.a_cancel_master_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.a_cancel_master",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_cancel_worker",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_nonbatch.a_cancel_worker_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.a_cancel_worker",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_termination_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.a_termination_window_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.a_termination_window",
            {"requests": output("a", "requests")},
        ),
    )
    case.step(
        "a_termination_ended",
        "cancel_check",
        params=case.params(
            "phase_timing_nonbatch.a_termination_ended",
            {"snapshot": output("a_termination_window", "snapshot")},
        ),
    )
    case.step(
        "a_receipt",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.a_receipt_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.a_receipt",
            {
                "requests": output("a", "requests"),
                "since": output("a_cancel_master", "snapshot"),
            },
        ),
    )
    case.step("b", "cancel_prepare", params=case.value("phase_timing_nonbatch.b"))
    case.step(
        "b_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_nonbatch.b_dispatch_timeout_s"),
        params={"requests": output("b", "requests")},
    )
    case.step(
        "b_first_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.b_first_window_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.b_first_window",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_first_received",
        "cancel_check",
        params=case.params(
            "phase_timing_nonbatch.b_first_received",
            {"snapshot": output("b_first_window", "snapshot")},
        ),
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_nonbatch.b_cancel_master_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.b_cancel_master",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_cancel_worker",
        "cancel_rpc",
        timeout_s=case.value("phase_timing_nonbatch.b_cancel_worker_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.b_cancel_worker",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.b_termination_window_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.b_termination_window",
            {"requests": output("b", "requests")},
        ),
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params=case.params(
            "phase_timing_nonbatch.b_termination_ended",
            {"snapshot": output("b_termination_window", "snapshot")},
        ),
    )
    case.step(
        "b_receipt",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.b_receipt_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.b_receipt",
            {
                "requests": output("b", "requests"),
                "since": output("b_cancel_master", "snapshot"),
            },
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("phase_timing_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("phase_timing_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("phase_timing_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "phase_timing_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "phase_timing_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def deadline_exempt_inflight_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("deadline_exempt_inflight_nonbatch.setup_timeout_s"),
    )
    case.step(
        "slow_enqueue",
        "engine_inject",
        params=case.value("deadline_exempt_inflight_nonbatch.slow_enqueue"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("deadline_exempt_inflight_nonbatch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "deadline_exempt_inflight_nonbatch.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "deadline_exempt_inflight_nonbatch.completion_window_timeout_s"
        ),
        params=case.params(
            "deadline_exempt_inflight_nonbatch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completion_ended",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_nonbatch.completion_ended",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "inflight_exempt_completes",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_nonbatch.inflight_exempt_completes",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_state",
        "cancel_observe",
        timeout_s=case.value(
            "deadline_exempt_inflight_nonbatch.cancel_state_timeout_s"
        ),
        params=case.params(
            "deadline_exempt_inflight_nonbatch.cancel_state",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "deadline_did_not_cancel_engine",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_nonbatch.deadline_did_not_cancel_engine",
            {"snapshot": output("cancel_state", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("deadline_exempt_inflight_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "deadline_exempt_inflight_nonbatch.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "deadline_exempt_inflight_nonbatch.recovery_window_timeout_s"
        ),
        params=case.params(
            "deadline_exempt_inflight_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "deadline_exempt_inflight_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step("cleanup", "teardown")


def schedule_drop_delivered(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("schedule_drop_delivered.setup_timeout_s"),
    )
    case.step(
        "slow_enqueue",
        "engine_inject",
        params=case.value("schedule_drop_delivered.slow_enqueue"),
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.cancel_baseline_timeout_s"),
        params=case.value("schedule_drop_delivered.cancel_baseline"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("schedule_drop_delivered.target"),
    )
    case.step(
        "schedule_begin",
        "cancel_begin",
        timeout_s=case.value("schedule_drop_delivered.schedule_begin_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "schedule_delivery_window",
        "cancel_observe",
        timeout_s=case.value(
            "schedule_drop_delivered.schedule_delivery_window_timeout_s"
        ),
        params=case.params(
            "schedule_drop_delivered.schedule_delivery_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "schedule_drop",
        "cancel_transport",
        params=case.params(
            "schedule_drop_delivered.schedule_drop",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "schedule_exit",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.schedule_exit_timeout_s"),
        params=case.params(
            "schedule_drop_delivered.schedule_exit",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "owned_schedule_future_cancelled",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.owned_schedule_future_cancelled",
            {"snapshot": output("schedule_exit", "snapshot")},
        ),
    )
    case.step(
        "forward_window",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.forward_window_timeout_s"),
        params=case.params(
            "schedule_drop_delivered.forward_window",
            {
                "requests": output("target", "requests"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.cancel_not_forwarded",
            {
                "snapshot": output("forward_window", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.closing_drain_timeout_s"),
        params=case.value("schedule_drop_delivered.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("schedule_drop_delivered.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("schedule_drop_delivered.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.recovery_window_timeout_s"),
        params=case.params(
            "schedule_drop_delivered.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step(
        "no_forward_final",
        "cancel_observe",
        timeout_s=case.value("schedule_drop_delivered.no_forward_final_timeout_s"),
        params=case.value("schedule_drop_delivered.no_forward_final"),
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params=case.params(
            "schedule_drop_delivered.ordinary_cancel_never_forwarded",
            {"snapshot": output("no_forward_final", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def stream_break_prefill_autonomous(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("stream_break_prefill_autonomous.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("stream_break_prefill_autonomous.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "stream_break_prefill_autonomous.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "stream_break_prefill_autonomous.first_output_window_timeout_s"
        ),
        params=case.params(
            "stream_break_prefill_autonomous.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "client_stream_drop",
        "cancel_transport",
        params=case.params(
            "stream_break_prefill_autonomous.client_stream_drop",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "engine_receipt",
        "cancel_observe",
        timeout_s=case.value(
            "stream_break_prefill_autonomous.engine_receipt_timeout_s"
        ),
        params=case.params(
            "stream_break_prefill_autonomous.engine_receipt",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "engine_observed_stream_cancellation",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.engine_observed_stream_cancellation",
            {"snapshot": output("engine_receipt", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("stream_break_prefill_autonomous.engine_drain_timeout_s"),
        params=case.value("stream_break_prefill_autonomous.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("stream_break_prefill_autonomous.master_drain_timeout_s"),
        params=case.value("stream_break_prefill_autonomous.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("stream_break_prefill_autonomous.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "stream_break_prefill_autonomous.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "stream_break_prefill_autonomous.recovery_window_timeout_s"
        ),
        params=case.params(
            "stream_break_prefill_autonomous.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "stream_break_prefill_autonomous.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def stream_break_decode_autonomous(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("stream_break_decode_autonomous.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("stream_break_decode_autonomous.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "stream_break_decode_autonomous.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "stream_break_decode_autonomous.first_output_window_timeout_s"
        ),
        params=case.params(
            "stream_break_decode_autonomous.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "client_stream_drop",
        "cancel_transport",
        params=case.params(
            "stream_break_decode_autonomous.client_stream_drop",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "engine_receipt",
        "cancel_observe",
        timeout_s=case.value("stream_break_decode_autonomous.engine_receipt_timeout_s"),
        params=case.params(
            "stream_break_decode_autonomous.engine_receipt",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "engine_observed_stream_cancellation",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.engine_observed_stream_cancellation",
            {"snapshot": output("engine_receipt", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("stream_break_decode_autonomous.engine_drain_timeout_s"),
        params=case.value("stream_break_decode_autonomous.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("stream_break_decode_autonomous.master_drain_timeout_s"),
        params=case.value("stream_break_decode_autonomous.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("stream_break_decode_autonomous.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "stream_break_decode_autonomous.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "stream_break_decode_autonomous.recovery_window_timeout_s"
        ),
        params=case.params(
            "stream_break_decode_autonomous.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "stream_break_decode_autonomous.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def preemption_victim_batch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("preemption_victim_batch.setup_timeout_s"),
    )
    case.step(
        "victim",
        "cancel_prepare",
        params=case.value("preemption_victim_batch.victim"),
    )
    case.step(
        "victim_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("preemption_victim_batch.victim_dispatch_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_running",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.victim_running_timeout_s"),
        params=case.params(
            "preemption_victim_batch.victim_running",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_engine_owned",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.victim_engine_owned",
            {"snapshot": output("victim_running", "snapshot")},
        ),
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.cancel_baseline_timeout_s"),
        params=case.value("preemption_victim_batch.cancel_baseline"),
    )
    case.step(
        "high",
        "cancel_prepare",
        params=case.value("preemption_victim_batch.high"),
    )
    case.step(
        "preemptor_arrival",
        "cancel_begin",
        timeout_s=case.value("preemption_victim_batch.preemptor_arrival_timeout_s"),
        params={"requests": output("high", "requests")},
    )
    case.step(
        "victim_terminal_window",
        "cancel_observe",
        timeout_s=case.value(
            "preemption_victim_batch.victim_terminal_window_timeout_s"
        ),
        params=case.params(
            "preemption_victim_batch.victim_terminal_window",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_terminal_ended",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.victim_terminal_ended",
            {"snapshot": output("victim_terminal_window", "snapshot")},
        ),
    )
    case.step(
        "victim_typed_engine_preemption",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.victim_typed_engine_preemption",
            {"snapshot": output("victim_terminal_window", "snapshot")},
        ),
    )
    case.step(
        "high_open",
        "cancel_open",
        timeout_s=case.value("preemption_victim_batch.high_open_timeout_s"),
        params={"requests": output("high", "requests")},
    )
    case.step(
        "high_completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "preemption_victim_batch.high_completion_window_timeout_s"
        ),
        params=case.params(
            "preemption_victim_batch.high_completion_window",
            {"requests": output("high", "requests")},
        ),
    )
    case.step(
        "high_completion_ended",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.high_completion_ended",
            {"snapshot": output("high_completion_window", "snapshot")},
        ),
    )
    case.step(
        "high_completed",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.high_completed",
            {"snapshot": output("high_completion_window", "snapshot")},
        ),
    )
    case.step(
        "preemption_state",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.preemption_state_timeout_s"),
        params=case.params(
            "preemption_victim_batch.preemption_state",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_engine_cancelled",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.victim_engine_cancelled",
            {"snapshot": output("preemption_state", "snapshot")},
        ),
    )
    case.step(
        "weak_cancel_was_forwarded",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.weak_cancel_was_forwarded",
            {
                "snapshot": output("preemption_state", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.master_drain_timeout_s"),
        params=case.value("preemption_victim_batch.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.engine_drain_timeout_s"),
        params=case.value("preemption_victim_batch.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("preemption_victim_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("preemption_victim_batch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_batch.recovery_window_timeout_s"),
        params=case.params(
            "preemption_victim_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "preemption_victim_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def preemption_victim_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("preemption_victim_nonbatch.setup_timeout_s"),
    )
    case.step(
        "victim",
        "cancel_prepare",
        params=case.value("preemption_victim_nonbatch.victim"),
    )
    case.step(
        "victim_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("preemption_victim_nonbatch.victim_dispatch_timeout_s"),
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_running",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_nonbatch.victim_running_timeout_s"),
        params=case.params(
            "preemption_victim_nonbatch.victim_running",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_engine_owned",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.victim_engine_owned",
            {"snapshot": output("victim_running", "snapshot")},
        ),
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_nonbatch.cancel_baseline_timeout_s"),
        params=case.value("preemption_victim_nonbatch.cancel_baseline"),
    )
    case.step(
        "high",
        "cancel_prepare",
        params=case.value("preemption_victim_nonbatch.high"),
    )
    case.step(
        "preemptor_arrival",
        "cancel_begin",
        timeout_s=case.value("preemption_victim_nonbatch.preemptor_arrival_timeout_s"),
        params={"requests": output("high", "requests")},
    )
    case.step(
        "victim_terminal_window",
        "cancel_observe",
        timeout_s=case.value(
            "preemption_victim_nonbatch.victim_terminal_window_timeout_s"
        ),
        params=case.params(
            "preemption_victim_nonbatch.victim_terminal_window",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_terminal_ended",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.victim_terminal_ended",
            {"snapshot": output("victim_terminal_window", "snapshot")},
        ),
    )
    case.step(
        "victim_typed_engine_preemption",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.victim_typed_engine_preemption",
            {"snapshot": output("victim_terminal_window", "snapshot")},
        ),
    )
    case.step(
        "high_open",
        "cancel_open",
        timeout_s=case.value("preemption_victim_nonbatch.high_open_timeout_s"),
        params={"requests": output("high", "requests")},
    )
    case.step(
        "high_completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "preemption_victim_nonbatch.high_completion_window_timeout_s"
        ),
        params=case.params(
            "preemption_victim_nonbatch.high_completion_window",
            {"requests": output("high", "requests")},
        ),
    )
    case.step(
        "high_completion_ended",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.high_completion_ended",
            {"snapshot": output("high_completion_window", "snapshot")},
        ),
    )
    case.step(
        "high_completed",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.high_completed",
            {"snapshot": output("high_completion_window", "snapshot")},
        ),
    )
    case.step(
        "preemption_state",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_nonbatch.preemption_state_timeout_s"),
        params=case.params(
            "preemption_victim_nonbatch.preemption_state",
            {"requests": output("victim", "requests")},
        ),
    )
    case.step(
        "victim_engine_cancelled",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.victim_engine_cancelled",
            {"snapshot": output("preemption_state", "snapshot")},
        ),
    )
    case.step(
        "weak_cancel_was_forwarded",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.weak_cancel_was_forwarded",
            {
                "snapshot": output("preemption_state", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_nonbatch.engine_drain_timeout_s"),
        params=case.value("preemption_victim_nonbatch.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("preemption_victim_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("preemption_victim_nonbatch.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("preemption_victim_nonbatch.recovery_window_timeout_s"),
        params=case.params(
            "preemption_victim_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "preemption_victim_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")
