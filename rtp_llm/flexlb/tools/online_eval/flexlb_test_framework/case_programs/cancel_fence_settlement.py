"""Cancel locality, terminal settlement and explicit Engine tombstone contracts; restart faults require EnqueueBatch."""

from ..case_config import output

METADATA = {
    "id": "cancel_fence_settlement",
    "description": "Cancel locality, terminal settlement and explicit Engine tombstone contracts; restart faults require "
    "EnqueueBatch.",
    "category": "cancel",
    "tags": ["migration", "cancel"],
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def engine_notfound_settle_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 1},
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("target", "requests"),
            "until": {"metric": "business_finished", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "direct_original_prefill_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "terminal_ack_is_not_found",
        "cancel_check",
        params={
            "snapshot": output("direct_original_prefill_cancel", "snapshot"),
            "metric": "engine_not_found",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "old_extra_wait_observed",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 2,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "post_terminal",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "terminal_not_rewritten",
        "cancel_check",
        params={
            "snapshot": output("post_terminal", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def engine_notfound_settle_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 1},
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("target", "requests"),
            "until": {"metric": "business_finished", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "completed_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "cancel_worker",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "direct_original_prefill_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "terminal_ack_is_not_found",
        "cancel_check",
        params={
            "snapshot": output("direct_original_prefill_cancel", "snapshot"),
            "metric": "engine_not_found",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "old_extra_wait_observed",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 2,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "post_terminal",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "terminal_not_rewritten",
        "cancel_check",
        params={
            "snapshot": output("post_terminal", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def prefill_dead_await_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "output_len": 500,
            "expected_stream_statuses": ["CANCELLED", "UNAVAILABLE"],
        },
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "stop_prefill",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "restore_topology",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def decode_retire_closes_fence(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "output_len": 1000,
            "expected_stream_statuses": ["CANCELLED", "UNAVAILABLE"],
        },
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "stop_prefill",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "stop_decode",
        "engine_control",
        params={"operation": "stop", "targets": ["decode-0"]},
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=60,
        params={
            "duration_s": 45,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "retirement_does_not_complete_business",
        "cancel_check",
        params={
            "snapshot": output("terminal_window", "snapshot"),
            "metric": "business_finished",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=60,
        params={
            "duration_s": 45,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    # Local TTL/cancel cleanup can precede the registry's offline observation.
    # Observe retirement before recreating engines so this is a real rejoin.
    for role in ("prefill", "decode"):
        observed = f"{role}_offline"
        case.step(
            observed,
            "cancel_observe",
            timeout_s=45,
            params={
                "duration_s": 30,
                "include": ["info"],
                "until": {"metric": f"alive_{role}", "op": "eq", "value": 0},
            },
        )
        case.step(
            f"{role}_retired",
            "cancel_check",
            params={
                "snapshot": output(observed, "snapshot"),
                "metric": f"alive_{role}",
                "expected": 0,
                "op": "eq",
            },
        )
    case.step(
        "restore_topology",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0", "decode-0"]},
    )
    case.step(
        "restore_ready",
        "master_ready",
        timeout_s=45,
        params={"inflight_zero": True},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def transport_failure_one_shot(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 500},
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_fault",
        "engine_inject",
        params={"type": "cancel_no_respond", "targets": ["prefill-0"], "options": {}},
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "post_settle_counter", "cancel_observe", timeout_s=30, params={"duration_s": 2}
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("post_settle_counter", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "clear_cancel_fault",
        "engine_clear",
        params={"fault": output("cancel_fault", "fault")},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def unexpected_status_await_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 500},
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_fault",
        "engine_inject",
        params={
            "type": "cancel_unexpected_status",
            "targets": ["prefill-0"],
            "options": {},
        },
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "terminal_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "post_settle_counter", "cancel_observe", timeout_s=30, params={"duration_s": 2}
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("post_settle_counter", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "master_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "master_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "clear_cancel_fault",
        "engine_clear",
        params={"fault": output("cancel_fault", "fault")},
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def engine_restarted_tombstoned_settle(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "output_len": 5000,
            "expected_stream_statuses": ["CANCELLED", "UNAVAILABLE"],
        },
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_crash_fault",
        "engine_inject",
        params={"type": "crash_after", "targets": ["prefill-0"], "options": {"n": 1}},
    )
    case.step(
        "first_crash_trigger",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "consume": "manual",
            "schedule_timeout_s": 8,
            "expected_rpc_statuses": [
                "DEADLINE_EXCEEDED",
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
            ],
        },
    )
    case.step(
        "first_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("first_crash_trigger", "requests")},
    )
    case.step(
        "first_crash_dropped",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 0},
            "include": ["info"],
        },
    )
    case.step(
        "first_crash_health_dropped",
        "cancel_check",
        params={
            "snapshot": output("first_crash_dropped", "snapshot"),
            "metric": "alive_prefill",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "first_crash_restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "first_crash_restored",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 1},
            "include": ["info"],
        },
    )
    case.step(
        "first_crash_health_restored",
        "cancel_check",
        params={
            "snapshot": output("first_crash_restored", "snapshot"),
            "metric": "alive_prefill",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "first_crash_reconnect",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 3, "include": ["info"]},
    )
    case.step(
        "first_crash_clear_fault",
        "engine_clear",
        params={"fault": output("first_crash_fault", "fault")},
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "fast_terminal",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "settled_in_five_second_wait",
        "cancel_check",
        params={
            "snapshot": output("fast_terminal", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_did_not_complete_business",
        "cancel_check",
        params={
            "snapshot": output("fast_terminal", "snapshot"),
            "metric": "business_finished",
            "expected": 0,
            "op": "eq",
        },
    )
    # Ordinary Master Cancel cannot arm an Engine tombstone. Probe that boundary
    # before the test client explicitly exercises the Engine Cancel contract.
    case.step(
        "master_no_forward", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("master_no_forward", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "explicit_engine_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={
            "requests": output("target", "requests"),
            "destination": "prefill",
        },
    )
    case.step(
        "cancel_arrival",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "cancel_rpc_count", "op": "ge", "value": 1},
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "explicit_cancel_reached_fresh_engine",
        "cancel_check",
        params={
            "snapshot": output("cancel_arrival", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 1,
            "op": "ge",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "armed_fence_probe",
        "cancel_enqueue_probe",
        params={
            "requests": output("target", "requests"),
            "output_len": 2,
            "attempt": 1,
        },
    )
    case.step(
        "armed_fence_rejects_exact_rid_8429",
        "cancel_check",
        params={
            "snapshot": output("armed_fence_probe", "snapshot"),
            "metric": "fence_rejected_8429",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=60,
        params={
            "duration_s": 45,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "residue_bound",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "scheduler", "op": "le", "value": 1},
        },
    )
    case.step(
        "residue_within_crash_trigger_bound",
        "cancel_check",
        params={
            "snapshot": output("residue_bound", "snapshot"),
            "metric": "scheduler",
            "expected": 1,
            "op": "le",
        },
    )
    case.step("residue_later", "cancel_observe", timeout_s=30, params={"duration_s": 8})
    case.step(
        "residue_does_not_grow",
        "cancel_check",
        params={
            "snapshot": output("residue_later", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "le",
            "baseline": output("residue_bound", "snapshot"),
        },
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def fencing_lost_on_engine_restart(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "output_len": 5000,
            "expected_stream_statuses": ["CANCELLED", "UNAVAILABLE"],
        },
    )
    case.step(
        "target_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "first_output_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "first_output_received",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_crash_fault",
        "engine_inject",
        params={"type": "crash_after", "targets": ["prefill-0"], "options": {"n": 1}},
    )
    case.step(
        "first_crash_trigger",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "consume": "manual",
            "schedule_timeout_s": 8,
            "expected_rpc_statuses": [
                "DEADLINE_EXCEEDED",
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
            ],
        },
    )
    case.step(
        "first_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("first_crash_trigger", "requests")},
    )
    case.step(
        "first_crash_dropped",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 0},
            "include": ["info"],
        },
    )
    case.step(
        "first_crash_health_dropped",
        "cancel_check",
        params={
            "snapshot": output("first_crash_dropped", "snapshot"),
            "metric": "alive_prefill",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "first_crash_restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "first_crash_restored",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 1},
            "include": ["info"],
        },
    )
    case.step(
        "first_crash_health_restored",
        "cancel_check",
        params={
            "snapshot": output("first_crash_restored", "snapshot"),
            "metric": "alive_prefill",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "first_crash_reconnect",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 3, "include": ["info"]},
    )
    case.step(
        "first_crash_clear_fault",
        "engine_clear",
        params={"fault": output("first_crash_fault", "fault")},
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "fast_terminal",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("target", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "settled_in_five_second_wait",
        "cancel_check",
        params={
            "snapshot": output("fast_terminal", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_did_not_complete_business",
        "cancel_check",
        params={
            "snapshot": output("fast_terminal", "snapshot"),
            "metric": "business_finished",
            "expected": 0,
            "op": "eq",
        },
    )
    # Ordinary Master Cancel cannot arm an Engine tombstone. Probe that boundary
    # before the test client explicitly exercises the Engine Cancel contract.
    case.step(
        "master_no_forward", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("master_no_forward", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "explicit_engine_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={
            "requests": output("target", "requests"),
            "destination": "prefill",
        },
    )
    case.step(
        "cancel_arrival",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "cancel_rpc_count", "op": "ge", "value": 1},
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "explicit_cancel_reached_fresh_engine",
        "cancel_check",
        params={
            "snapshot": output("cancel_arrival", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 1,
            "op": "ge",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "armed_fence_probe",
        "cancel_enqueue_probe",
        params={
            "requests": output("target", "requests"),
            "output_len": 100,
            "scope": "prefill_only",
            "attempt": 1,
        },
    )
    case.step(
        "armed_fence_rejects_exact_rid_8429",
        "cancel_check",
        params={
            "snapshot": output("armed_fence_probe", "snapshot"),
            "metric": "fence_rejected_8429",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "second_crash_fault",
        "engine_inject",
        params={"type": "crash_after", "targets": ["prefill-0"], "options": {"n": 1}},
    )
    case.step(
        "second_crash_trigger",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "consume": "manual",
            "schedule_timeout_s": 8,
            "expected_rpc_statuses": [
                "DEADLINE_EXCEEDED",
                "UNAVAILABLE",
                "UNKNOWN",
                "INTERNAL",
            ],
        },
    )
    case.step(
        "second_crash_trigger_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("second_crash_trigger", "requests")},
    )
    case.step(
        "second_crash_dropped",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 0},
            "include": ["info"],
        },
    )
    case.step(
        "second_crash_health_dropped",
        "cancel_check",
        params={
            "snapshot": output("second_crash_dropped", "snapshot"),
            "metric": "alive_prefill",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "second_crash_restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "second_crash_restored",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 1},
            "include": ["info"],
        },
    )
    case.step(
        "second_crash_health_restored",
        "cancel_check",
        params={
            "snapshot": output("second_crash_restored", "snapshot"),
            "metric": "alive_prefill",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "second_crash_reconnect",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 3, "include": ["info"]},
    )
    case.step(
        "second_crash_clear_fault",
        "engine_clear",
        params={"fault": output("second_crash_fault", "fault")},
    )
    case.step(
        "lost_fence_probe",
        "cancel_enqueue_probe",
        params={
            "requests": output("target", "requests"),
            "output_len": 100,
            "scope": "prefill_only",
            "attempt": 2,
            "wait_port_s": 10,
        },
    )
    case.step(
        "memory_only_fence_is_lost",
        "cancel_check",
        params={
            "snapshot": output("lost_fence_probe", "snapshot"),
            "metric": "probe_accepted",
            "expected": 1,
            "op": "ge",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=75,
        params={
            "duration_s": 60,
            "until": {"metric": "engine_clean_total", "op": "eq", "value": 0},
        },
    )
    case.step(
        "engine_drain_inflight",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_inflight",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain_leaks",
        "cancel_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "engine_leaks",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "residue_bound",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "scheduler", "op": "le", "value": 2},
        },
    )
    case.step(
        "residue_within_crash_trigger_bound",
        "cancel_check",
        params={
            "snapshot": output("residue_bound", "snapshot"),
            "metric": "scheduler",
            "expected": 2,
            "op": "le",
        },
    )
    case.step("residue_later", "cancel_observe", timeout_s=30, params={"duration_s": 8})
    case.step(
        "residue_does_not_grow",
        "cancel_check",
        params={
            "snapshot": output("residue_later", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "le",
            "baseline": output("residue_bound", "snapshot"),
        },
    )
    case.step(
        "recovery",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 2},
    )
    case.step(
        "recovery_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("recovery", "requests"),
            "until": {"metric": "success", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "cancel_check",
        params={
            "snapshot": output("recovery_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "engine_notfound_settle_batch": {
        "build": engine_notfound_settle_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "engine_notfound_settle_nonbatch": {
        "build": engine_notfound_settle_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "prefill_dead_await_terminal": {
        "build": prefill_dead_await_terminal,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "decode_retire_closes_fence": {
        "build": decode_retire_closes_fence,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "transport_failure_one_shot": {
        "build": transport_failure_one_shot,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "unexpected_status_await_terminal": {
        "build": unexpected_status_await_terminal,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "engine_restarted_tombstoned_settle": {
        "build": engine_restarted_tombstoned_settle,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "fencing_lost_on_engine_restart": {
        "build": fencing_lost_on_engine_restart,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
}
