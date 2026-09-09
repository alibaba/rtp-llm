"""Twelve cancellation lifecycle contracts with explicit client, Master and engine boundaries."""

from ..case_config import output

METADATA = {
    "id": "cancel_lifecycle",
    "description": "Twelve cancellation lifecycle contracts with explicit client, Master and engine "
    "boundaries.",
    "category": "cancel",
    "tags": ["migration", "cancel"],
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def basic_batch(case):
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
        "first_output_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
        "stream_terminated",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "engine_receipt_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("target", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 0},
            "since": output("first_master_cancel", "snapshot"),
        },
    )
    case.step(
        "master_cancel_does_not_cancel_engine",
        "cancel_check",
        params={
            "snapshot": output("engine_receipt_window", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
            "within_s": 5,
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
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "no_forward_final", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params={
            "snapshot": output("no_forward_final", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def idempotent_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10},
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
        "first_output_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_count_baseline",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_engine_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("target", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 0},
            "since": output("first_master_cancel", "snapshot"),
        },
    )
    case.step(
        "master_cancel_does_not_cancel_engine",
        "cancel_check",
        params={
            "snapshot": output("first_engine_receipt", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
            "within_s": 5,
        },
    )
    case.step(
        "first_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("first_engine_receipt", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_count_baseline", "snapshot"),
        },
    )
    case.step(
        "second_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "second_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("second_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
        "stream_terminated",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
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
    case.step(
        "after_second_cancel",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "second_cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("after_second_cancel", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("first_engine_receipt", "snapshot"),
        },
    )
    case.step(
        "closing_drain_observed",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "no_forward_final", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params={
            "snapshot": output("no_forward_final", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def after_terminal_batch(case):
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
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
    case.step(
        "terminal_after_cancel",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "terminal_not_rewritten_as_cancelled",
        "cancel_check",
        params={
            "snapshot": output("terminal_after_cancel", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def basic_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10},
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
        "first_output_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_worker_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
        "stream_terminated",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "engine_receipt_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("target", "requests"),
            "since": output("first_master_cancel", "snapshot"),
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


def idempotent_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10},
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
        "first_output_before_cancel",
        "cancel_check",
        params={
            "snapshot": output("first_output_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_count_baseline",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_worker_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_engine_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("target", "requests"),
            "since": output("first_master_cancel", "snapshot"),
        },
    )
    case.step(
        "second_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "second_worker_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "second_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("second_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "second_worker_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("second_worker_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
        "stream_terminated",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
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
    case.step(
        "after_second_cancel",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step("cleanup", "teardown")


def after_terminal_nonbatch(case):
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
        "first_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "first_worker_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "prefill"},
    )
    case.step(
        "first_master_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_master_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "first_worker_rpc_ok",
        "cancel_check",
        params={
            "snapshot": output("first_worker_cancel", "snapshot"),
            "metric": "rpc_ok",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "termination_window",
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
    case.step(
        "terminal_after_cancel",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "terminal_not_rewritten_as_cancelled",
        "cancel_check",
        params={
            "snapshot": output("terminal_after_cancel", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def unknown_rid(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "cancel_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "cancel_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("unknown", "cancel_prepare", params={"count": 1, "concurrency": 1})
    case.step(
        "unknown_master_cancel",
        "cancel_rpc",
        timeout_s=30,
        params={
            "requests": output("unknown", "requests"),
            "destination": "master",
            "expected_rpc_statuses": ["NOT_FOUND"],
        },
    )
    case.step(
        "unknown_is_typed_not_found",
        "cancel_check",
        params={
            "snapshot": output("unknown_master_cancel", "snapshot"),
            "metric": "rpc_not_found",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("after_unknown", "cancel_observe", timeout_s=30, params={"duration_s": 0})
    case.step(
        "unknown_does_not_mutate_ledger",
        "cancel_check",
        params={
            "snapshot": output("after_unknown", "snapshot"),
            "metric": "fingerprint",
            "expected": True,
            "op": "eq",
            "baseline": output("clean_baseline", "snapshot"),
        },
    )
    case.step("cleanup", "teardown")


def anomaly_path_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("target", "cancel_prepare", params={"count": 1, "concurrency": 1})
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
        "cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("target", "requests"), "destination": "master"},
    )
    case.step(
        "termination_window",
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
        "termination_ended",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
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
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def sibling_isolation_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "a",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10, "consume": "manual"},
    )
    case.step(
        "b",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 500, "consume": "manual"},
    )
    case.step(
        "c",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10, "consume": "manual"},
    )
    case.step(
        "dispatch_siblings",
        "cancel_dispatch_group",
        timeout_s=60,
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
        timeout_s=30,
        params={"requests": output("a", "requests")},
    )
    case.step(
        "b_open",
        "cancel_open",
        timeout_s=30,
        params={"requests": output("b", "requests")},
    )
    case.step(
        "c_open",
        "cancel_open",
        timeout_s=30,
        params={"requests": output("c", "requests")},
    )
    case.step(
        "a_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("a", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_first_received",
        "cancel_check",
        params={
            "snapshot": output("a_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("c", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "c_first_received",
        "cancel_check",
        params={
            "snapshot": output("c_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "master"},
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("b", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("b_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "isolation_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 2,
            "until": {"metric": "scheduler", "op": "le", "value": 2},
        },
    )
    case.step(
        "cancelled_slot_removed",
        "cancel_check",
        params={
            "snapshot": output("isolation_window", "snapshot"),
            "metric": "scheduler",
            "expected": 2,
            "op": "le",
        },
    )
    case.step(
        "a_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("a", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("a_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "a_completed",
        "cancel_check",
        params={
            "snapshot": output("a_completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("c", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "c_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("c_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_completed",
        "cancel_check",
        params={
            "snapshot": output("c_completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
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
    case.step(
        "b_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("b", "requests")},
    )
    case.step(
        "b_completes_independently",
        "cancel_check",
        params={
            "snapshot": output("b_state", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_engine_not_cancelled",
        "cancel_check",
        params={
            "snapshot": output("b_state", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "no_forward_final", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params={
            "snapshot": output("no_forward_final", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def phase_timing_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("a", "cancel_prepare", params={"count": 1, "concurrency": 1})
    case.step(
        "a_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("a", "requests")},
    )
    case.step(
        "a_prefill_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0.1,
            "requests": output("a", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "a_still_prefill",
        "cancel_check",
        params={
            "snapshot": output("a_prefill_window", "snapshot"),
            "metric": "first_output",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "a_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("a", "requests"), "destination": "master"},
    )
    case.step(
        "a_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("a", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("a_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "a_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("a", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 0},
            "since": output("a_cancel_master", "snapshot"),
        },
    )
    case.step(
        "a_master_cancel_keeps_engine_running",
        "cancel_check",
        params={
            "snapshot": output("a_receipt", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
            "within_s": 5,
        },
    )
    case.step("b", "cancel_prepare", params={"count": 1, "concurrency": 1})
    case.step(
        "b_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("b", "requests")},
    )
    case.step(
        "b_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("b", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_first_received",
        "cancel_check",
        params={
            "snapshot": output("b_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "master"},
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("b", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("b_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("b", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 0},
            "since": output("b_cancel_master", "snapshot"),
        },
    )
    case.step(
        "b_master_cancel_keeps_engine_running",
        "cancel_check",
        params={
            "snapshot": output("b_receipt", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
            "within_s": 5,
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
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "no_forward_final", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params={
            "snapshot": output("no_forward_final", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def deadline_exempt_inflight_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_enqueue",
        "engine_inject",
        params={
            "type": "enqueue_delay",
            "targets": ["prefill-0", "prefill-1"],
            "options": {"delay_ms": 3000},
        },
    )
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
        "completion_window",
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
        "completion_ended",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "inflight_exempt_completes",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "deadline_did_not_cancel_engine",
        "cancel_check",
        params={
            "snapshot": output("cancel_state", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
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
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step("cleanup", "teardown")


def anomaly_path_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("target", "cancel_prepare", params={"count": 1, "concurrency": 1})
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
        "termination_window",
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
        "termination_ended",
        "cancel_check",
        params={
            "snapshot": output("termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
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


def sibling_isolation_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "a",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10, "consume": "manual"},
    )
    case.step(
        "b",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 500, "consume": "manual"},
    )
    case.step(
        "c",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 10, "consume": "manual"},
    )
    case.step(
        "dispatch_siblings",
        "cancel_dispatch_group",
        timeout_s=60,
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
        timeout_s=30,
        params={"requests": output("a", "requests")},
    )
    case.step(
        "b_open",
        "cancel_open",
        timeout_s=30,
        params={"requests": output("b", "requests")},
    )
    case.step(
        "c_open",
        "cancel_open",
        timeout_s=30,
        params={"requests": output("c", "requests")},
    )
    case.step(
        "a_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("a", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_first_received",
        "cancel_check",
        params={
            "snapshot": output("a_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("c", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "c_first_received",
        "cancel_check",
        params={
            "snapshot": output("c_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "master"},
    )
    case.step(
        "b_cancel_worker",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "prefill"},
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("b", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("b_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "a_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("a", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("a_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "a_completed",
        "cancel_check",
        params={
            "snapshot": output("a_completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("c", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "c_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("c_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "c_completed",
        "cancel_check",
        params={
            "snapshot": output("c_completion_window", "snapshot"),
            "metric": "business_finished",
            "expected": 1,
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
    case.step(
        "b_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("b", "requests")},
    )
    case.step(
        "b_not_completed",
        "cancel_check",
        params={
            "snapshot": output("b_state", "snapshot"),
            "metric": "business_finished",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "b_engine_cancelled",
        "cancel_check",
        params={
            "snapshot": output("b_state", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def phase_timing_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("a", "cancel_prepare", params={"count": 1, "concurrency": 1})
    case.step(
        "a_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("a", "requests")},
    )
    case.step(
        "a_prefill_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0.1,
            "requests": output("a", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "a_still_prefill",
        "cancel_check",
        params={
            "snapshot": output("a_prefill_window", "snapshot"),
            "metric": "first_output",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "a_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("a", "requests"), "destination": "master"},
    )
    case.step(
        "a_cancel_worker",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("a", "requests"), "destination": "prefill"},
    )
    case.step(
        "a_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("a", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "a_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("a_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "a_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("a", "requests"),
            "since": output("a_cancel_master", "snapshot"),
        },
    )
    case.step("b", "cancel_prepare", params={"count": 1, "concurrency": 1})
    case.step(
        "b_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("b", "requests")},
    )
    case.step(
        "b_first_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("b", "requests"),
            "until": {"metric": "first_output", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_first_received",
        "cancel_check",
        params={
            "snapshot": output("b_first_window", "snapshot"),
            "metric": "first_output",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_cancel_master",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "master"},
    )
    case.step(
        "b_cancel_worker",
        "cancel_rpc",
        timeout_s=30,
        params={"requests": output("b", "requests"), "destination": "prefill"},
    )
    case.step(
        "b_termination_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("b", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "b_termination_ended",
        "cancel_check",
        params={
            "snapshot": output("b_termination_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "b_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("b", "requests"),
            "since": output("b_cancel_master", "snapshot"),
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


def deadline_exempt_inflight_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_enqueue",
        "engine_inject",
        params={
            "type": "enqueue_delay",
            "targets": ["prefill-0", "prefill-1"],
            "options": {"delay_ms": 3000},
        },
    )
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
        "completion_window",
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
        "completion_ended",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "inflight_exempt_completes",
        "cancel_check",
        params={
            "snapshot": output("completion_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("target", "requests")},
    )
    case.step(
        "deadline_did_not_cancel_engine",
        "cancel_check",
        params={
            "snapshot": output("cancel_state", "snapshot"),
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
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step("cleanup", "teardown")


def schedule_drop_delivered(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_enqueue",
        "engine_inject",
        params={
            "type": "enqueue_delay",
            "targets": ["prefill-0", "prefill-1"],
            "options": {"delay_ms": 2000},
        },
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "target",
        "cancel_prepare",
        params={"count": 1, "concurrency": 1, "output_len": 500, "consume": "manual"},
    )
    case.step(
        "schedule_begin",
        "cancel_begin",
        timeout_s=30,
        params={"requests": output("target", "requests")},
    )
    case.step(
        "schedule_delivery_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 0.5,
            "requests": output("target", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "schedule_drop",
        "cancel_transport",
        params={"requests": output("target", "requests"), "phase": "schedule"},
    )
    case.step(
        "schedule_exit",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 5,
            "requests": output("target", "requests"),
            "until": {"metric": "schedule_cancelled", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "owned_schedule_future_cancelled",
        "cancel_check",
        params={
            "snapshot": output("schedule_exit", "snapshot"),
            "metric": "schedule_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "forward_window",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "requests": output("target", "requests"),
            "until": {"metric": "cancel_rpc_count", "op": "eq", "value": 0},
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "cancel_not_forwarded",
        "cancel_check",
        params={
            "snapshot": output("forward_window", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "closing_drain",
        "cancel_observe",
        timeout_s=55,
        params={
            "duration_s": 40,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "closing_drain_scheduler",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step(
        "closing_drain_decode_load",
        "cancel_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
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
    case.step(
        "clear_delay", "engine_clear", params={"fault": output("slow_enqueue", "fault")}
    )
    case.step(
        "no_forward_final", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "ordinary_cancel_never_forwarded",
        "cancel_check",
        params={
            "snapshot": output("no_forward_final", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 0,
            "op": "eq",
        },
    )
    case.step("cleanup", "teardown")


def stream_break_prefill_autonomous(case):
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
        "client_stream_drop",
        "cancel_transport",
        params={"requests": output("target", "requests"), "phase": "stream"},
    )
    case.step(
        "engine_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "requests": output("target", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 1},
        },
    )
    case.step(
        "engine_observed_stream_cancellation",
        "cancel_check",
        params={
            "snapshot": output("engine_receipt", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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
        "master_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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


def stream_break_decode_autonomous(case):
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
        "client_stream_drop",
        "cancel_transport",
        params={"requests": output("target", "requests"), "phase": "stream"},
    )
    case.step(
        "engine_receipt",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "requests": output("target", "requests"),
            "until": {"metric": "engine_cancelled", "op": "eq", "value": 1},
        },
    )
    case.step(
        "engine_observed_stream_cancellation",
        "cancel_check",
        params={
            "snapshot": output("engine_receipt", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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
        "master_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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


def preemption_victim_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "victim",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 200,
            "priority": 30,
            "unique_block_keys": True,
        },
    )
    case.step(
        "victim_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_running",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "requests": output("victim", "requests"),
            "until": {"metric": "decode_running", "op": "eq", "value": 1},
            "include": ["mock"],
        },
    )
    case.step(
        "victim_engine_owned",
        "cancel_check",
        params={
            "snapshot": output("victim_running", "snapshot"),
            "metric": "decode_running",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "high",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "priority": 70,
            "unique_block_keys": True,
            "consume": "manual",
            "schedule_timeout_s": 30,
        },
    )
    case.step(
        "preemptor_arrival",
        "cancel_begin",
        timeout_s=40,
        params={"requests": output("high", "requests")},
    )
    case.step(
        "victim_terminal_window",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "requests": output("victim", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "victim_terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("victim_terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "victim_typed_engine_preemption",
        "cancel_check",
        params={
            "snapshot": output("victim_terminal_window", "snapshot"),
            "metric": "typed_preempted",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "high_open",
        "cancel_open",
        timeout_s=40,
        params={"requests": output("high", "requests")},
    )
    case.step(
        "high_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("high", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "high_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("high_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "high_completed",
        "cancel_check",
        params={
            "snapshot": output("high_completion_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "preemption_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("victim", "requests")},
    )
    case.step(
        "victim_engine_cancelled",
        "cancel_check",
        params={
            "snapshot": output("preemption_state", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "weak_cancel_was_forwarded",
        "cancel_check",
        params={
            "snapshot": output("preemption_state", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 1,
            "op": "ge",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "master_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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
        timeout_s=30,
        params={
            "duration_s": 15,
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


def preemption_victim_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "victim",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 200,
            "priority": 30,
            "unique_block_keys": True,
        },
    )
    case.step(
        "victim_dispatch",
        "cancel_dispatch",
        timeout_s=60,
        params={"requests": output("victim", "requests")},
    )
    case.step(
        "victim_running",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "requests": output("victim", "requests"),
            "until": {"metric": "decode_running", "op": "eq", "value": 1},
            "include": ["mock"],
        },
    )
    case.step(
        "victim_engine_owned",
        "cancel_check",
        params={
            "snapshot": output("victim_running", "snapshot"),
            "metric": "decode_running",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "cancel_baseline", "cancel_observe", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "high",
        "cancel_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "priority": 70,
            "unique_block_keys": True,
            "consume": "manual",
            "schedule_timeout_s": 30,
        },
    )
    case.step(
        "preemptor_arrival",
        "cancel_begin",
        timeout_s=40,
        params={"requests": output("high", "requests")},
    )
    case.step(
        "victim_terminal_window",
        "cancel_observe",
        timeout_s=35,
        params={
            "duration_s": 20,
            "requests": output("victim", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "victim_terminal_ended",
        "cancel_check",
        params={
            "snapshot": output("victim_terminal_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "victim_typed_engine_preemption",
        "cancel_check",
        params={
            "snapshot": output("victim_terminal_window", "snapshot"),
            "metric": "typed_preempted",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "high_open",
        "cancel_open",
        timeout_s=40,
        params={"requests": output("high", "requests")},
    )
    case.step(
        "high_completion_window",
        "cancel_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "requests": output("high", "requests"),
            "until": {"metric": "stream_ended", "op": "eq", "value": 1},
            "include": ["client_records"],
        },
    )
    case.step(
        "high_completion_ended",
        "cancel_check",
        params={
            "snapshot": output("high_completion_window", "snapshot"),
            "metric": "stream_ended",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "high_completed",
        "cancel_check",
        params={
            "snapshot": output("high_completion_window", "snapshot"),
            "metric": "success",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "preemption_state",
        "cancel_observe",
        timeout_s=30,
        params={"duration_s": 0, "requests": output("victim", "requests")},
    )
    case.step(
        "victim_engine_cancelled",
        "cancel_check",
        params={
            "snapshot": output("preemption_state", "snapshot"),
            "metric": "engine_cancelled",
            "expected": 1,
            "op": "eq",
        },
    )
    case.step(
        "weak_cancel_was_forwarded",
        "cancel_check",
        params={
            "snapshot": output("preemption_state", "snapshot"),
            "metric": "cancel_rpc_count",
            "expected": 1,
            "op": "ge",
            "baseline": output("cancel_baseline", "snapshot"),
        },
    )
    case.step(
        "engine_drain",
        "cancel_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
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
    "basic_batch": {
        "build": basic_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "idempotent_batch": {
        "build": idempotent_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "after_terminal_batch": {
        "build": after_terminal_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "basic_nonbatch": {
        "build": basic_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "idempotent_nonbatch": {
        "build": idempotent_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "after_terminal_nonbatch": {
        "build": after_terminal_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "unknown_rid": {
        "build": unknown_rid,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "anomaly_path_batch": {
        "build": anomaly_path_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "sibling_isolation_batch": {
        "build": sibling_isolation_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "phase_timing_batch": {
        "build": phase_timing_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "deadline_exempt_inflight_batch": {
        "build": deadline_exempt_inflight_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "anomaly_path_nonbatch": {
        "build": anomaly_path_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "sibling_isolation_nonbatch": {
        "build": sibling_isolation_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "phase_timing_nonbatch": {
        "build": phase_timing_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "deadline_exempt_inflight_nonbatch": {
        "build": deadline_exempt_inflight_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
    "schedule_drop_delivered": {
        "build": schedule_drop_delivered,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "stream_break_prefill_autonomous": {
        "build": stream_break_prefill_autonomous,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "stream_break_decode_autonomous": {
        "build": stream_break_decode_autonomous,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {
            "requires": ["generate_stream"],
        },
    },
    "preemption_victim_batch": {
        "build": preemption_victim_batch,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {},
    },
    "preemption_victim_nonbatch": {
        "build": preemption_victim_nonbatch,
        "profiles": ["single-nonbatch", "window-nonbatch"],
        "metadata": {},
    },
}
