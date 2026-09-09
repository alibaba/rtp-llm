"""Cancel locality, terminal settlement and explicit Engine tombstone contracts; restart faults require EnqueueBatch."""

from ..case_config import output


def engine_notfound_settle_batch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("engine_notfound_settle_batch.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("engine_notfound_settle_batch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("engine_notfound_settle_batch.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "engine_notfound_settle_batch.completion_window_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_batch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.completed_before_cancel",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=case.value("engine_notfound_settle_batch.cancel_master_timeout_s"),
        params=case.params(
            "engine_notfound_settle_batch.cancel_master",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "direct_original_prefill_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "engine_notfound_settle_batch.direct_original_prefill_cancel_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_batch.direct_original_prefill_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ack_is_not_found",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.terminal_ack_is_not_found",
            {"snapshot": output("direct_original_prefill_cancel", "snapshot")},
        ),
    )
    case.step(
        "old_extra_wait_observed",
        "cancel_observe",
        timeout_s=case.value(
            "engine_notfound_settle_batch.old_extra_wait_observed_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_batch.old_extra_wait_observed",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "post_terminal",
        "cancel_observe",
        timeout_s=case.value("engine_notfound_settle_batch.post_terminal_timeout_s"),
        params=case.params(
            "engine_notfound_settle_batch.post_terminal",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_not_rewritten",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.terminal_not_rewritten",
            {"snapshot": output("post_terminal", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("engine_notfound_settle_batch.master_drain_timeout_s"),
        params=case.value("engine_notfound_settle_batch.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("engine_notfound_settle_batch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_notfound_settle_batch.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("engine_notfound_settle_batch.recovery_window_timeout_s"),
        params=case.params(
            "engine_notfound_settle_batch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_batch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def engine_notfound_settle_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("engine_notfound_settle_nonbatch.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("engine_notfound_settle_nonbatch.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.completion_window_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_nonbatch.completion_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_nonbatch.completed_before_cancel",
            {"snapshot": output("completion_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=case.value("engine_notfound_settle_nonbatch.cancel_master_timeout_s"),
        params=case.params(
            "engine_notfound_settle_nonbatch.cancel_master",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "cancel_worker",
        "cancel_rpc",
        timeout_s=case.value("engine_notfound_settle_nonbatch.cancel_worker_timeout_s"),
        params=case.params(
            "engine_notfound_settle_nonbatch.cancel_worker",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "direct_original_prefill_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.direct_original_prefill_cancel_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_nonbatch.direct_original_prefill_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ack_is_not_found",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_nonbatch.terminal_ack_is_not_found",
            {"snapshot": output("direct_original_prefill_cancel", "snapshot")},
        ),
    )
    case.step(
        "old_extra_wait_observed",
        "cancel_observe",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.old_extra_wait_observed_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_nonbatch.old_extra_wait_observed",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "post_terminal",
        "cancel_observe",
        timeout_s=case.value("engine_notfound_settle_nonbatch.post_terminal_timeout_s"),
        params=case.params(
            "engine_notfound_settle_nonbatch.post_terminal",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_not_rewritten",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_nonbatch.terminal_not_rewritten",
            {"snapshot": output("post_terminal", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("engine_notfound_settle_nonbatch.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "engine_notfound_settle_nonbatch.recovery_window_timeout_s"
        ),
        params=case.params(
            "engine_notfound_settle_nonbatch.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "engine_notfound_settle_nonbatch.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def prefill_dead_await_terminal(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("prefill_dead_await_terminal.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("prefill_dead_await_terminal.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("prefill_dead_await_terminal.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "prefill_dead_await_terminal.first_output_window_timeout_s"
        ),
        params=case.params(
            "prefill_dead_await_terminal.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "stop_prefill",
        "engine_control",
        params=case.value("prefill_dead_await_terminal.stop_prefill"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value("prefill_dead_await_terminal.master_cancel_timeout_s"),
        params=case.params(
            "prefill_dead_await_terminal.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=case.value("prefill_dead_await_terminal.terminal_window_timeout_s"),
        params=case.params(
            "prefill_dead_await_terminal.terminal_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.terminal_ended",
            {"snapshot": output("terminal_window", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("prefill_dead_await_terminal.master_drain_timeout_s"),
        params=case.value("prefill_dead_await_terminal.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("prefill_dead_await_terminal.engine_drain_timeout_s"),
        params=case.value("prefill_dead_await_terminal.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "restore_topology",
        "engine_control",
        params=case.value("prefill_dead_await_terminal.restore_topology"),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("prefill_dead_await_terminal.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("prefill_dead_await_terminal.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("prefill_dead_await_terminal.recovery_window_timeout_s"),
        params=case.params(
            "prefill_dead_await_terminal.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "prefill_dead_await_terminal.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def decode_retire_closes_fence(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_retire_closes_fence.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("decode_retire_closes_fence.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("decode_retire_closes_fence.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "decode_retire_closes_fence.first_output_window_timeout_s"
        ),
        params=case.params(
            "decode_retire_closes_fence.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "stop_prefill",
        "engine_control",
        params=case.value("decode_retire_closes_fence.stop_prefill"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value("decode_retire_closes_fence.master_cancel_timeout_s"),
        params=case.params(
            "decode_retire_closes_fence.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "stop_decode",
        "engine_control",
        params=case.value("decode_retire_closes_fence.stop_decode"),
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=case.value("decode_retire_closes_fence.terminal_window_timeout_s"),
        params=case.params(
            "decode_retire_closes_fence.terminal_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.terminal_ended",
            {"snapshot": output("terminal_window", "snapshot")},
        ),
    )
    case.step(
        "retirement_does_not_complete_business",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.retirement_does_not_complete_business",
            {"snapshot": output("terminal_window", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("decode_retire_closes_fence.master_drain_timeout_s"),
        params=case.value("decode_retire_closes_fence.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("decode_retire_closes_fence.engine_drain_timeout_s"),
        params=case.value("decode_retire_closes_fence.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    # Local TTL/cancel cleanup can precede the registry's offline observation.
    # Observe retirement before recreating engines so this is a real rejoin.
    for role in ("prefill", "decode"):
        observed = f"{role}_offline"
        case.step(
            observed,
            "cancel_observe",
            timeout_s=case.value("decode_retire_closes_fence.step_26_timeout_s"),
            params=case.params(
                "decode_retire_closes_fence.step_26",
                {
                    "until": {
                        "metric": f"alive_{role}",
                        "op": case.value("decode_retire_closes_fence.step_26.until.op"),
                        "value": case.value(
                            "decode_retire_closes_fence.step_26.until.value"
                        ),
                    }
                },
            ),
        )
        case.step(
            f"{role}_retired",
            "cancel_check",
            params=case.params(
                "decode_retire_closes_fence.step_27",
                {"snapshot": output(observed, "snapshot"), "metric": f"alive_{role}"},
            ),
        )
    case.step(
        "restore_topology",
        "engine_control",
        params=case.value("decode_retire_closes_fence.restore_topology"),
    )
    case.step(
        "restore_ready",
        "master_ready",
        timeout_s=case.value("decode_retire_closes_fence.restore_ready_timeout_s"),
        params=case.value("decode_retire_closes_fence.restore_ready"),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("decode_retire_closes_fence.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("decode_retire_closes_fence.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("decode_retire_closes_fence.recovery_window_timeout_s"),
        params=case.params(
            "decode_retire_closes_fence.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "decode_retire_closes_fence.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def transport_failure_one_shot(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("transport_failure_one_shot.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("transport_failure_one_shot.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("transport_failure_one_shot.target_dispatch_timeout_s"),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "transport_failure_one_shot.first_output_window_timeout_s"
        ),
        params=case.params(
            "transport_failure_one_shot.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_fault",
        "engine_inject",
        params=case.value("transport_failure_one_shot.cancel_fault"),
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value("transport_failure_one_shot.cancel_baseline_timeout_s"),
        params=case.value("transport_failure_one_shot.cancel_baseline"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value("transport_failure_one_shot.master_cancel_timeout_s"),
        params=case.params(
            "transport_failure_one_shot.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=case.value("transport_failure_one_shot.terminal_window_timeout_s"),
        params=case.params(
            "transport_failure_one_shot.terminal_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.terminal_ended",
            {"snapshot": output("terminal_window", "snapshot")},
        ),
    )
    case.step(
        "post_settle_counter",
        "cancel_observe",
        timeout_s=case.value(
            "transport_failure_one_shot.post_settle_counter_timeout_s"
        ),
        params=case.value("transport_failure_one_shot.post_settle_counter"),
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.ordinary_cancel_not_forwarded",
            {
                "snapshot": output("post_settle_counter", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("transport_failure_one_shot.master_drain_timeout_s"),
        params=case.value("transport_failure_one_shot.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("transport_failure_one_shot.engine_drain_timeout_s"),
        params=case.value("transport_failure_one_shot.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "clear_cancel_fault",
        "engine_clear",
        params={"fault": output("cancel_fault", "fault")},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("transport_failure_one_shot.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value("transport_failure_one_shot.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value("transport_failure_one_shot.recovery_window_timeout_s"),
        params=case.params(
            "transport_failure_one_shot.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "transport_failure_one_shot.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unexpected_status_await_terminal(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("unexpected_status_await_terminal.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("unexpected_status_await_terminal.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "unexpected_status_await_terminal.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "unexpected_status_await_terminal.first_output_window_timeout_s"
        ),
        params=case.params(
            "unexpected_status_await_terminal.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "cancel_fault",
        "engine_inject",
        params=case.value("unexpected_status_await_terminal.cancel_fault"),
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value(
            "unexpected_status_await_terminal.cancel_baseline_timeout_s"
        ),
        params=case.value("unexpected_status_await_terminal.cancel_baseline"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "unexpected_status_await_terminal.master_cancel_timeout_s"
        ),
        params=case.params(
            "unexpected_status_await_terminal.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=case.value(
            "unexpected_status_await_terminal.terminal_window_timeout_s"
        ),
        params=case.params(
            "unexpected_status_await_terminal.terminal_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.terminal_ended",
            {"snapshot": output("terminal_window", "snapshot")},
        ),
    )
    case.step(
        "post_settle_counter",
        "cancel_observe",
        timeout_s=case.value(
            "unexpected_status_await_terminal.post_settle_counter_timeout_s"
        ),
        params=case.value("unexpected_status_await_terminal.post_settle_counter"),
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.ordinary_cancel_not_forwarded",
            {
                "snapshot": output("post_settle_counter", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=case.value("unexpected_status_await_terminal.master_drain_timeout_s"),
        params=case.value("unexpected_status_await_terminal.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("unexpected_status_await_terminal.engine_drain_timeout_s"),
        params=case.value("unexpected_status_await_terminal.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "clear_cancel_fault",
        "engine_clear",
        params={"fault": output("cancel_fault", "fault")},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("unexpected_status_await_terminal.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "unexpected_status_await_terminal.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "unexpected_status_await_terminal.recovery_window_timeout_s"
        ),
        params=case.params(
            "unexpected_status_await_terminal.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "unexpected_status_await_terminal.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def engine_restarted_tombstoned_settle(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("engine_restarted_tombstoned_settle.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("engine_restarted_tombstoned_settle.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.first_output_window_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "first_crash_fault",
        "engine_inject",
        params=case.value("engine_restarted_tombstoned_settle.first_crash_fault"),
    )
    case.step(
        "first_crash_trigger",
        "cancel_prepare",
        params=case.value("engine_restarted_tombstoned_settle.first_crash_trigger"),
    )
    case.step(
        "first_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.first_crash_trigger_dispatch_timeout_s"
        ),
        params={"requests": output("first_crash_trigger", "requests")},
    )
    case.step(
        "first_crash_dropped",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.first_crash_dropped_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.first_crash_dropped"),
    )
    case.step(
        "first_crash_health_dropped",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.first_crash_health_dropped",
            {"snapshot": output("first_crash_dropped", "snapshot")},
        ),
    )
    case.step(
        "first_crash_restart",
        "engine_control",
        params=case.value("engine_restarted_tombstoned_settle.first_crash_restart"),
    )
    case.step(
        "first_crash_restored",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.first_crash_restored_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.first_crash_restored"),
    )
    case.step(
        "first_crash_health_restored",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.first_crash_health_restored",
            {"snapshot": output("first_crash_restored", "snapshot")},
        ),
    )
    case.step(
        "first_crash_reconnect",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.first_crash_reconnect_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.first_crash_reconnect"),
    )
    case.step(
        "first_crash_clear_fault",
        "engine_clear",
        params={"fault": output("first_crash_fault", "fault")},
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.cancel_baseline_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.cancel_baseline"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.master_cancel_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "fast_terminal",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.fast_terminal_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.fast_terminal",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "settled_in_five_second_wait",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.settled_in_five_second_wait",
            {"snapshot": output("fast_terminal", "snapshot")},
        ),
    )
    case.step(
        "cancel_did_not_complete_business",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.cancel_did_not_complete_business",
            {"snapshot": output("fast_terminal", "snapshot")},
        ),
    )
    # Ordinary Master Cancel cannot arm an Engine tombstone. Probe that boundary
    # before the test client explicitly exercises the Engine Cancel contract.
    case.step(
        "master_no_forward",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.master_no_forward_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.master_no_forward"),
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.ordinary_cancel_not_forwarded",
            {
                "snapshot": output("master_no_forward", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "explicit_engine_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.explicit_engine_cancel_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.explicit_engine_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "cancel_arrival",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.cancel_arrival_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.cancel_arrival",
            {"baseline": output("cancel_baseline", "snapshot")},
        ),
    )
    case.step(
        "explicit_cancel_reached_fresh_engine",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.explicit_cancel_reached_fresh_engine",
            {
                "snapshot": output("cancel_arrival", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "armed_fence_probe",
        "cancel_enqueue_probe",
        params=case.params(
            "engine_restarted_tombstoned_settle.armed_fence_probe",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "armed_fence_rejects_exact_rid_8429",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.armed_fence_rejects_exact_rid_8429",
            {"snapshot": output("armed_fence_probe", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.engine_drain_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "residue_bound",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.residue_bound_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.residue_bound"),
    )
    case.step(
        "residue_within_crash_trigger_bound",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.residue_within_crash_trigger_bound",
            {"snapshot": output("residue_bound", "snapshot")},
        ),
    )
    case.step(
        "residue_later",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.residue_later_timeout_s"
        ),
        params=case.value("engine_restarted_tombstoned_settle.residue_later"),
    )
    case.step(
        "residue_does_not_grow",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.residue_does_not_grow",
            {
                "snapshot": output("residue_later", "snapshot"),
                "baseline": output("residue_bound", "snapshot"),
            },
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("engine_restarted_tombstoned_settle.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "engine_restarted_tombstoned_settle.recovery_window_timeout_s"
        ),
        params=case.params(
            "engine_restarted_tombstoned_settle.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "engine_restarted_tombstoned_settle.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def fencing_lost_on_engine_restart(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("fencing_lost_on_engine_restart.setup_timeout_s"),
    )
    case.step(
        "target",
        "cancel_prepare",
        params=case.value("fencing_lost_on_engine_restart.target"),
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.target_dispatch_timeout_s"
        ),
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.first_output_window_timeout_s"
        ),
        params=case.params(
            "fencing_lost_on_engine_restart.first_output_window",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.first_output_received",
            {"snapshot": output("first_output_window", "snapshot")},
        ),
    )
    case.step(
        "first_crash_fault",
        "engine_inject",
        params=case.value("fencing_lost_on_engine_restart.first_crash_fault"),
    )
    case.step(
        "first_crash_trigger",
        "cancel_prepare",
        params=case.value("fencing_lost_on_engine_restart.first_crash_trigger"),
    )
    case.step(
        "first_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.first_crash_trigger_dispatch_timeout_s"
        ),
        params={"requests": output("first_crash_trigger", "requests")},
    )
    case.step(
        "first_crash_dropped",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.first_crash_dropped_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.first_crash_dropped"),
    )
    case.step(
        "first_crash_health_dropped",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.first_crash_health_dropped",
            {"snapshot": output("first_crash_dropped", "snapshot")},
        ),
    )
    case.step(
        "first_crash_restart",
        "engine_control",
        params=case.value("fencing_lost_on_engine_restart.first_crash_restart"),
    )
    case.step(
        "first_crash_restored",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.first_crash_restored_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.first_crash_restored"),
    )
    case.step(
        "first_crash_health_restored",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.first_crash_health_restored",
            {"snapshot": output("first_crash_restored", "snapshot")},
        ),
    )
    case.step(
        "first_crash_reconnect",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.first_crash_reconnect_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.first_crash_reconnect"),
    )
    case.step(
        "first_crash_clear_fault",
        "engine_clear",
        params={"fault": output("first_crash_fault", "fault")},
    )
    case.step(
        "cancel_baseline",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.cancel_baseline_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.cancel_baseline"),
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=case.value("fencing_lost_on_engine_restart.master_cancel_timeout_s"),
        params=case.params(
            "fencing_lost_on_engine_restart.master_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "fast_terminal",
        "cancel_observe",
        timeout_s=case.value("fencing_lost_on_engine_restart.fast_terminal_timeout_s"),
        params=case.params(
            "fencing_lost_on_engine_restart.fast_terminal",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "settled_in_five_second_wait",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.settled_in_five_second_wait",
            {"snapshot": output("fast_terminal", "snapshot")},
        ),
    )
    case.step(
        "cancel_did_not_complete_business",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.cancel_did_not_complete_business",
            {"snapshot": output("fast_terminal", "snapshot")},
        ),
    )
    # Ordinary Master Cancel cannot arm an Engine tombstone. Probe that boundary
    # before the test client explicitly exercises the Engine Cancel contract.
    case.step(
        "master_no_forward",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.master_no_forward_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.master_no_forward"),
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.ordinary_cancel_not_forwarded",
            {
                "snapshot": output("master_no_forward", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "explicit_engine_cancel",
        "cancel_rpc",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.explicit_engine_cancel_timeout_s"
        ),
        params=case.params(
            "fencing_lost_on_engine_restart.explicit_engine_cancel",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "cancel_arrival",
        "cancel_observe",
        timeout_s=case.value("fencing_lost_on_engine_restart.cancel_arrival_timeout_s"),
        params=case.params(
            "fencing_lost_on_engine_restart.cancel_arrival",
            {"baseline": output("cancel_baseline", "snapshot")},
        ),
    )
    case.step(
        "explicit_cancel_reached_fresh_engine",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.explicit_cancel_reached_fresh_engine",
            {
                "snapshot": output("cancel_arrival", "snapshot"),
                "baseline": output("cancel_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "armed_fence_probe",
        "cancel_enqueue_probe",
        params=case.params(
            "fencing_lost_on_engine_restart.armed_fence_probe",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "armed_fence_rejects_exact_rid_8429",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.armed_fence_rejects_exact_rid_8429",
            {"snapshot": output("armed_fence_probe", "snapshot")},
        ),
    )
    case.step(
        "second_crash_fault",
        "engine_inject",
        params=case.value("fencing_lost_on_engine_restart.second_crash_fault"),
    )
    case.step(
        "second_crash_trigger",
        "cancel_prepare",
        params=case.value("fencing_lost_on_engine_restart.second_crash_trigger"),
    )
    case.step(
        "second_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.second_crash_trigger_dispatch_timeout_s"
        ),
        params={"requests": output("second_crash_trigger", "requests")},
    )
    case.step(
        "second_crash_dropped",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.second_crash_dropped_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.second_crash_dropped"),
    )
    case.step(
        "second_crash_health_dropped",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.second_crash_health_dropped",
            {"snapshot": output("second_crash_dropped", "snapshot")},
        ),
    )
    case.step(
        "second_crash_restart",
        "engine_control",
        params=case.value("fencing_lost_on_engine_restart.second_crash_restart"),
    )
    case.step(
        "second_crash_restored",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.second_crash_restored_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.second_crash_restored"),
    )
    case.step(
        "second_crash_health_restored",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.second_crash_health_restored",
            {"snapshot": output("second_crash_restored", "snapshot")},
        ),
    )
    case.step(
        "second_crash_reconnect",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.second_crash_reconnect_timeout_s"
        ),
        params=case.value("fencing_lost_on_engine_restart.second_crash_reconnect"),
    )
    case.step(
        "second_crash_clear_fault",
        "engine_clear",
        params={"fault": output("second_crash_fault", "fault")},
    )
    case.step(
        "lost_fence_probe",
        "cancel_enqueue_probe",
        params=case.params(
            "fencing_lost_on_engine_restart.lost_fence_probe",
            {"requests": output("target", "requests")},
        ),
    )
    case.step(
        "memory_only_fence_is_lost",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.memory_only_fence_is_lost",
            {"snapshot": output("lost_fence_probe", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=case.value("fencing_lost_on_engine_restart.engine_drain_timeout_s"),
        params=case.value("fencing_lost_on_engine_restart.engine_drain"),
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "residue_bound",
        "cancel_observe",
        timeout_s=case.value("fencing_lost_on_engine_restart.residue_bound_timeout_s"),
        params=case.value("fencing_lost_on_engine_restart.residue_bound"),
    )
    case.step(
        "residue_within_crash_trigger_bound",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.residue_within_crash_trigger_bound",
            {"snapshot": output("residue_bound", "snapshot")},
        ),
    )
    case.step(
        "residue_later",
        "cancel_observe",
        timeout_s=case.value("fencing_lost_on_engine_restart.residue_later_timeout_s"),
        params=case.value("fencing_lost_on_engine_restart.residue_later"),
    )
    case.step(
        "residue_does_not_grow",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.residue_does_not_grow",
            {
                "snapshot": output("residue_later", "snapshot"),
                "baseline": output("residue_bound", "snapshot"),
            },
        ),
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params=case.value("fencing_lost_on_engine_restart.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.recovery_dispatch_timeout_s"
        ),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=case.value(
            "fencing_lost_on_engine_restart.recovery_window_timeout_s"
        ),
        params=case.params(
            "fencing_lost_on_engine_restart.recovery_window",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params=case.params(
            "fencing_lost_on_engine_restart.recovery_succeeds",
            {"snapshot": output("recovery_window", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")
