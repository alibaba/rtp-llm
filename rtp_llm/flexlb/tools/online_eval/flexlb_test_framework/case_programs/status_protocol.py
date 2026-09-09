"""Explicit status protocol boundary programs; old whole-case expected_fail is narrowed to reviewed checks only."""

from ..case_config import output

METADATA = {
    "description": "Explicit status protocol boundary programs; old whole-case expected_fail is "
    "narrowed to reviewed checks only.",
    "category": "status",
    "tags": ["protocol", "migration"],
    "id": "status_protocol",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def inflight_ttl_cleanup(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_prefill",
        "status_perf",
        params={"prefill_fixed_ms": 10000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "silent",
        "status_prepare",
        params={
            "count": 6,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 10,
            "consume": "deferred",
        },
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=195,
        params={"duration_s": 180, "interval_s": 2},
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "p_suppress",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "prefill",
            "requests": output("silent", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_suppress",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "decode",
            "requests": output("silent", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "silent_dispatch",
        "status_dispatch",
        timeout_s=190,
        params={"requests": output("silent", "requests")},
    )
    case.step(
        "accepted_window",
        "status_sample",
        timeout_s=30,
        params={
            "duration_s": 15,
            "interval_s": 0.5,
            "until": {"metric": "prefill_accepted", "op": "ge", "value": 6},
        },
    )
    case.step(
        "six_prefill_accepted",
        "status_check",
        params={
            "snapshot": output("accepted_window", "snapshot"),
            "metric": "prefill_accepted",
            "op": "ge",
            "expected": 6,
            "aggregate": "last",
            "baseline": output("ttl_before", "snapshot"),
        },
    )
    case.step(
        "active_at_twelve_seconds",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 12},
    )
    case.step(
        "ledger_nonempty_after_twelve",
        "status_check",
        params={
            "snapshot": output("active_at_twelve_seconds", "snapshot"),
            "metric": "scheduler",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "ttl_drain",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "interval_s": 2,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "scheduler_ttl_drain",
        "status_check",
        params={
            "snapshot": output("ttl_drain", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ttl_event_window",
        "status_sample",
        timeout_s=120,
        params={
            "duration_s": 105,
            "until": {"metric": "ttl_scheduler", "op": "ge", "value": 6},
            "include": ["inflight", "mock", "info", "ttl"],
        },
    )
    case.step(
        "six_ttl_eviction_events",
        "status_check",
        params={
            "snapshot": output("ttl_event_window", "snapshot"),
            "metric": "ttl_scheduler",
            "op": "ge",
            "expected": 6,
            "aggregate": "last",
            "baseline": output("ttl_before", "snapshot"),
        },
    )
    case.step(
        "p_clear",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "prefill",
            "requests": output("silent", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_clear",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "decode",
            "requests": output("silent", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def prefill_suppress_all(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=195,
        params={"duration_s": 180, "interval_s": 2},
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "traffic",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "stream_timeout_s": 45,
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
        },
    )
    case.step(
        "on_0",
        "status_control",
        params={
            "fault": "status_suppress_running",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "on_1",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params={"requests": output("traffic", "requests"), "timeout_or_success": True},
    )
    case.step(
        "scheduler_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "scheduler_retires",
        "status_check",
        params={
            "snapshot": output("scheduler_window", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "prefill_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "prefill_batches", "op": "eq", "value": 0},
        },
    )
    case.step(
        "prefill_retires",
        "status_check",
        params={
            "snapshot": output("prefill_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params={
            "snapshot": output("ttl_channel_after", "snapshot"),
            "metric": "ttl_scheduler",
            "op": "ge",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "prefill_ttl_channel_reachable",
        "status_check",
        params={
            "snapshot": output("ttl_channel_after", "snapshot"),
            "metric": "ttl_prefill",
            "op": "ge",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "off_0",
        "status_control",
        params={
            "fault": "status_suppress_running",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "off_1",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step("final_owners", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step(
        "final_scheduler_zero",
        "status_check",
        params={
            "snapshot": output("final_owners", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "final_prefill_batches_zero",
        "status_check",
        params={
            "snapshot": output("final_owners", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def prefill_suppress_finished(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "traffic",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "stream_timeout_s": 20,
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
        },
    )
    case.step(
        "on_0",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params={"requests": output("traffic", "requests"), "timeout_or_success": True},
    )
    case.step(
        "off_0",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def decode_suppress_finished(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "traffic",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "stream_timeout_s": 45,
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
        },
    )
    case.step(
        "on_0",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": True,
            "role": "decode",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params={"requests": output("traffic", "requests"), "timeout_or_success": True},
    )
    case.step(
        "p_finished_window",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "prefill_batches", "op": "eq", "value": 0},
        },
    )
    case.step(
        "prefill_finishes_independently",
        "status_check",
        params={
            "snapshot": output("p_finished_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "off_0",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": False,
            "role": "decode",
        },
    )
    case.step(
        "decode_recovery_window",
        "status_sample",
        timeout_s=135,
        params={
            "duration_s": 120,
            "until": {"metric": "decode_total_load", "op": "eq", "value": 0},
        },
    )
    case.step(
        "decode_retires_after_clear",
        "status_check",
        params={
            "snapshot": output("decode_recovery_window", "snapshot"),
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


def no_respond(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_prefill",
        "status_perf",
        params={"prefill_fixed_ms": 3000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "live",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "live_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("live", "requests")},
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=195,
        params={"duration_s": 180, "interval_s": 2},
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "retire_on",
        "status_control",
        params={
            "fault": "status_no_respond",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "some_retired_window",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 1},
            "include": ["inflight", "mock", "info", "ttl"],
        },
    )
    case.step(
        "at_least_one_prefill_retired",
        "status_check",
        params={
            "snapshot": output("some_retired_window", "snapshot"),
            "metric": "alive_prefill",
            "op": "le",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "all_retired_window",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "eq", "value": 0},
            "include": ["inflight", "mock", "info", "ttl"],
        },
    )
    case.step(
        "all_prefill_retired",
        "status_check",
        params={
            "snapshot": output("all_retired_window", "snapshot"),
            "metric": "alive_prefill",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "retired_ledger_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "retired_ledger_drained",
        "status_check",
        params={
            "snapshot": output("retired_ledger_window", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params={
            "snapshot": output("ttl_channel_after", "snapshot"),
            "metric": "ttl_scheduler",
            "op": "ge",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "retire_off",
        "status_control",
        params={
            "fault": "status_no_respond",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "final_scheduler", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "final_scheduler_zero",
        "status_check",
        params={
            "snapshot": output("final_scheduler", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "topology_recovery",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["inflight", "mock", "info", "ttl"],
        },
    )
    case.step(
        "topology_recovers",
        "status_check",
        params={
            "snapshot": output("topology_recovery", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
            "aggregate": "last",
        },
    )
    case.step(
        "reconnect_window", "status_sample", timeout_s=30, params={"duration_s": 2}
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def version_regress(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_prefill",
        "status_perf",
        params={"prefill_fixed_ms": 3000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "live",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "live_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("live", "requests")},
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=195,
        params={"duration_s": 180, "interval_s": 2},
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "retire_on",
        "status_control",
        params={
            "fault": "status_version_regress",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "some_retired_window",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 1},
            "include": ["inflight", "mock", "info", "ttl"],
        },
    )
    case.step(
        "at_least_one_prefill_retired",
        "status_check",
        params={
            "snapshot": output("some_retired_window", "snapshot"),
            "metric": "alive_prefill",
            "op": "le",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "retired_ledger_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "retired_ledger_drained",
        "status_check",
        params={
            "snapshot": output("retired_ledger_window", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight", "mock", "info", "ttl"]},
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params={
            "snapshot": output("ttl_channel_after", "snapshot"),
            "metric": "ttl_scheduler",
            "op": "ge",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "retire_off",
        "status_control",
        params={
            "fault": "status_version_regress",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "final_scheduler", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "final_scheduler_zero",
        "status_check",
        params={
            "snapshot": output("final_scheduler", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "topology_recovery",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["inflight", "mock", "info", "ttl"],
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


def unknown_rid_finished(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ghost",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "ghost_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("ghost", "requests"),
            "rids": "all",
        },
    )
    case.step("ghost_window", "status_sample", timeout_s=30, params={"duration_s": 3})
    case.step(
        "ghost_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "unknown_terminal_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "unknown_terminal_ignored",
        "status_check",
        params={
            "snapshot": output("unknown_terminal_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("clean_baseline", "snapshot"),
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


def unknown_rid_running(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ghost",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "ghost_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RUNNING"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("ghost", "requests"),
            "rids": "all",
        },
    )
    case.step("ghost_window", "status_sample", timeout_s=30, params={"duration_s": 3})
    case.step(
        "ghost_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "unknown_active_clear_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "unknown_active_retires_after_clear",
        "status_check",
        params={
            "snapshot": output("unknown_active_clear_window", "snapshot"),
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


def unknown_batchid(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_prefill",
        "status_perf",
        params={"prefill_fixed_ms": 3000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "baseline_request",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "baseline_request_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("baseline_request", "requests")},
    )
    case.step(
        "baseline_request_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("baseline_request", "requests")},
    )
    case.step(
        "baseline_request_success",
        "status_outcomes",
        params={
            "requests": output("baseline_request", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "real",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "real_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("real", "requests")},
    )
    case.step(
        "legacy_effective_batch_zero",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "batch_id": 0},
            "enabled": True,
            "role": "prefill",
            "selection": "landing",
            "requests": output("real", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "real_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("real", "requests")},
    )
    case.step(
        "real_request_unaffected",
        "status_outcomes",
        params={
            "requests": output("real", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "fake_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def special_ids(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "slow_prefill",
        "status_perf",
        params={"prefill_fixed_ms": 3000, "restore_prefill_fixed_ms": 100},
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "negative_rid_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"rid": -1, "phase": "finished", "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
        },
    )
    case.step(
        "negative_rid_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "negative_rid_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "negative_rid_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "negative_rid_ignored",
        "status_check",
        params={
            "snapshot": output("negative_rid_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("clean_baseline", "snapshot"),
        },
    )
    case.step(
        "zero_rid_channel_probe",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"rid": 0, "phase": "finished"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "expected_http": [200, 400],
        },
    )
    case.step(
        "zero_rid_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "zero_real",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "zero_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("zero_real", "requests")},
    )
    case.step(
        "zero_batch_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "batch_id": 0, "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "landing",
            "requests": output("zero_real", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "zero_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("zero_real", "requests")},
    )
    case.step(
        "zero_batch_unaffected",
        "status_outcomes",
        params={
            "requests": output("zero_real", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "zero_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "negative_real",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "negative_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("negative_real", "requests")},
    )
    case.step(
        "negative_batch_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "batch_id": -1, "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "landing",
            "requests": output("negative_real", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "negative_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("negative_real", "requests")},
    )
    case.step(
        "negative_batch_unaffected",
        "status_outcomes",
        params={
            "requests": output("negative_real", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "negative_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def unbatched_single_request(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "omitted_running",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "omitted_running_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "omitted_running_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RUNNING"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("omitted_running", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "omitted_running_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 3},
    )
    case.step(
        "omitted_running_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "omitted_running_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "omitted_running_ignored",
        "status_check",
        params={
            "snapshot": output("omitted_running_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("omitted_running_before", "snapshot"),
        },
    )
    case.step(
        "omitted_finished",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "omitted_finished_before",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "omitted_finished_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("omitted_finished", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "omitted_finished_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 3},
    )
    case.step(
        "omitted_finished_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "omitted_finished_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "omitted_finished_ignored",
        "status_check",
        params={
            "snapshot": output("omitted_finished_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("omitted_finished_before", "snapshot"),
        },
    )
    case.step(
        "zero_running",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "zero_running_before", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "zero_running_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RUNNING", "batch_id": 0},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("zero_running", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "zero_running_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "zero_running_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "zero_running_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "zero_running_ignored",
        "status_check",
        params={
            "snapshot": output("zero_running_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("zero_running_before", "snapshot"),
        },
    )
    case.step(
        "zero_finished",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "zero_finished_before", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "zero_finished_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "batch_id": 0, "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("zero_finished", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "zero_finished_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "zero_finished_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "zero_finished_ignored_after_clear",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 0},
    )
    case.step(
        "zero_finished_ignored",
        "status_check",
        params={
            "snapshot": output("zero_finished_ignored_after_clear", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("zero_finished_before", "snapshot"),
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def foreign_batchid(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ghost",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "foreign_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished", "batch_id": 10000000, "error_code": 8500},
            "enabled": True,
            "role": "prefill",
            "selection": "all",
            "requests": output("ghost", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "foreign_ghost_window", "status_sample", timeout_s=30, params={"duration_s": 3}
    )
    case.step(
        "foreign_terminal_ignored",
        "status_check",
        params={
            "snapshot": output("foreign_ghost_window", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("clean_baseline", "snapshot"),
        },
    )
    case.step(
        "real_traffic",
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
        "real_traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("real_traffic", "requests")},
    )
    case.step(
        "real_traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("real_traffic", "requests")},
    )
    case.step(
        "real_traffic_success",
        "status_outcomes",
        params={
            "requests": output("real_traffic", "requests"),
            "success_min": 4,
            "failure_max": 0,
        },
    )
    case.step(
        "foreign_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "after_foreign",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_foreign_scheduler",
        "status_check",
        params={
            "snapshot": output("after_foreign", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_foreign_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_foreign", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_foreign_decode_load",
        "status_check",
        params={
            "snapshot": output("after_foreign", "snapshot"),
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


def duplicate_finished(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "duplicate_on",
        "status_control",
        params={
            "fault": "status_duplicate_finished",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "traffic",
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
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_success",
        "status_outcomes",
        params={
            "requests": output("traffic", "requests"),
            "success_min": 4,
            "failure_max": 0,
        },
    )
    case.step(
        "initial_drain",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "duplicate_replay_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 5},
    )
    case.step(
        "duplicate_off",
        "status_control",
        params={
            "fault": "status_duplicate_finished",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "terminal_replay_is_idempotent",
        "status_check",
        params={
            "snapshot": output("duplicate_replay_window", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("initial_drain", "snapshot"),
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def cursor_regress(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "history",
        "status_prepare",
        params={
            "count": 3,
            "concurrency": 3,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "history_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("history", "requests")},
    )
    case.step(
        "history_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("history", "requests")},
    )
    case.step(
        "history_success",
        "status_outcomes",
        params={
            "requests": output("history", "requests"),
            "success_min": 3,
            "failure_max": 0,
        },
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "cursor_on",
        "status_control",
        params={
            "fault": "status_cursor_regress",
            "config": {"n": 3},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step("replay_before", "status_sample", timeout_s=30, params={"duration_s": 0})
    case.step("replay_window", "status_sample", timeout_s=30, params={"duration_s": 5})
    case.step(
        "cursor_off",
        "status_control",
        params={
            "fault": "status_cursor_regress",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "cursor_replay_is_idempotent",
        "status_check",
        params={
            "snapshot": output("replay_window", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("replay_before", "snapshot"),
        },
    )
    case.step(
        "after_replay",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_replay_scheduler",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_replay_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_replay_decode_load",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def finished_then_running(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "settled",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "settled_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("settled", "requests")},
    )
    case.step(
        "settled_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("settled", "requests")},
    )
    case.step(
        "settled_success",
        "status_outcomes",
        params={
            "requests": output("settled", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "terminal_replay",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "finished"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("settled", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "terminal_window", "status_sample", timeout_s=30, params={"duration_s": 2}
    )
    case.step(
        "terminal_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "active_replay",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RUNNING"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("settled", "requests"),
            "rids": "all",
        },
    )
    case.step("active_window", "status_sample", timeout_s=30, params={"duration_s": 5})
    case.step(
        "active_off",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "terminal_cannot_resurrect",
        "status_check",
        params={
            "snapshot": output("active_window", "snapshot"),
            "metric": "fingerprint",
            "op": "eq",
            "expected": True,
            "aggregate": "last",
            "baseline": output("clean_baseline", "snapshot"),
        },
    )
    case.step(
        "after_replay",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_replay_scheduler",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_replay_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_replay_decode_load",
        "status_check",
        params={
            "snapshot": output("after_replay", "snapshot"),
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


def zombie_completed_running(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "decode_zombie_on",
        "status_control",
        params={
            "fault": "status_zombie_running",
            "config": {},
            "enabled": True,
            "role": "decode",
        },
    )
    case.step(
        "traffic",
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
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_success",
        "status_outcomes",
        params={
            "requests": output("traffic", "requests"),
            "success_min": 4,
            "failure_max": 0,
        },
    )
    case.step("zombie_window", "status_sample", timeout_s=30, params={"duration_s": 10})
    case.step(
        "decode_zombie_off",
        "status_control",
        params={
            "fault": "status_zombie_running",
            "config": {},
            "enabled": False,
            "role": "decode",
        },
    )
    case.step(
        "after_zombie",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_zombie_scheduler",
        "status_check",
        params={
            "snapshot": output("after_zombie", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_zombie_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_zombie", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_zombie_decode_load",
        "status_check",
        params={
            "snapshot": output("after_zombie", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "decode_load_zero",
        "status_check",
        params={
            "snapshot": output("after_zombie", "snapshot"),
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


def zombie_fake_running(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=35,
        params={
            "duration_s": 20,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params={
            "snapshot": output("clean_baseline", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "ghosts",
        "status_prepare",
        params={
            "count": 3,
            "concurrency": 3,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "ghosts_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RUNNING"},
            "enabled": True,
            "role": "prefill",
            "selection": "first",
            "requests": output("ghosts", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "active_ghost_window",
        "status_sample",
        timeout_s=75,
        params={"duration_s": 60, "interval_s": 5},
    )
    case.step(
        "resident_growth_bounded",
        "status_check",
        params={
            "snapshot": output("active_ghost_window", "snapshot"),
            "metric": "scheduler",
            "op": "le",
            "expected": 3,
            "aggregate": "last",
            "baseline": output("clean_baseline", "snapshot"),
        },
    )
    case.step(
        "master_healthy_during_active",
        "status_check",
        params={
            "snapshot": output("active_ghost_window", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
            "aggregate": "last",
        },
    )
    case.step(
        "ghosts_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "clear_retirement_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "scheduler", "op": "eq", "value": 0},
        },
    )
    case.step(
        "ghosts_retire_after_clear",
        "status_check",
        params={
            "snapshot": output("clear_retirement_window", "snapshot"),
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


def decode_before_prefill(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "traffic",
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
        "p_terminals_off",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "decode_completes_requests",
        "status_outcomes",
        params={
            "requests": output("traffic", "requests"),
            "success_min": 4,
            "failure_max": 0,
        },
    )
    case.step(
        "event_driven_window",
        "status_sample",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "prefill_batches", "op": "eq", "value": 0},
        },
    )
    case.step(
        "decode_terminal_retires_prefill_promptly",
        "status_check",
        params={
            "snapshot": output("event_driven_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "fallback_retirement_window",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "prefill_batches", "op": "eq", "value": 0},
        },
    )
    case.step(
        "p_terminal_restore",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "prefill_after_clear", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "prefill_zero_after_clear",
        "status_check",
        params={
            "snapshot": output("prefill_after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def decode_running_before_prefill(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "traffic",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "p_suppress",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_finished_suppress",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": True,
            "role": "decode",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "nonempty_window",
        "status_sample",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "prefill_batches", "op": "ge", "value": 1},
        },
    )
    case.step(
        "batch_really_dispatched",
        "status_check",
        params={
            "snapshot": output("nonempty_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "intermediate_hold_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 10},
    )
    case.step(
        "intermediate_cannot_retire_prefill",
        "status_check",
        params={
            "snapshot": output("intermediate_hold_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "intermediate_cannot_retire_scheduler",
        "status_check",
        params={
            "snapshot": output("intermediate_hold_window", "snapshot"),
            "metric": "scheduler",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "p_clear",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_finished_clear",
        "status_control",
        params={
            "fault": "status_suppress_finished",
            "config": {},
            "enabled": False,
            "role": "decode",
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def decode_waiting_before_prefill(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "traffic",
        "status_prepare",
        params={
            "count": 4,
            "concurrency": 4,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "p_suppress",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_suppress",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": True,
            "role": "decode",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_waiting_on",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {"phase": "RECEIVED"},
            "enabled": True,
            "role": "decode",
            "selection": "all",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "nonempty_window",
        "status_sample",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "prefill_batches", "op": "ge", "value": 1},
        },
    )
    case.step(
        "batch_really_dispatched",
        "status_check",
        params={
            "snapshot": output("nonempty_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "intermediate_hold_window",
        "status_sample",
        timeout_s=30,
        params={"duration_s": 10},
    )
    case.step(
        "intermediate_cannot_retire_prefill",
        "status_check",
        params={
            "snapshot": output("intermediate_hold_window", "snapshot"),
            "metric": "prefill_batches",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "intermediate_cannot_retire_scheduler",
        "status_check",
        params={
            "snapshot": output("intermediate_hold_window", "snapshot"),
            "metric": "scheduler",
            "op": "ge",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "p_clear",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "prefill",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_clear",
        "status_control",
        params={
            "fault": "status_suppress_rids",
            "config": {},
            "enabled": False,
            "role": "decode",
            "requests": output("traffic", "requests"),
            "rids": "all",
        },
    )
    case.step(
        "d_waiting_clear",
        "status_control",
        params={
            "fault": "status_fake_task",
            "config": {},
            "enabled": False,
            "role": "decode",
        },
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params={
            "snapshot": output("after_clear", "snapshot"),
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


def fetch_error(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "faulted",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
            "expected_rpc_statuses": ["UNKNOWN", "INTERNAL"],
        },
    )
    case.step(
        "fetch_fault_on",
        "status_control",
        params={
            "fault": "fetch_error",
            "config": {},
            "enabled": True,
            "role": "prefill",
        },
    )
    case.step(
        "faulted_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("faulted", "requests")},
    )
    case.step(
        "faulted_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("faulted", "requests")},
    )
    case.step(
        "fetch_fault_surfaces",
        "status_outcomes",
        params={
            "requests": output("faulted", "requests"),
            "failure_min": 1,
            "failure_max": 1,
        },
    )
    case.step(
        "fetch_fault_off",
        "status_control",
        params={
            "fault": "fetch_error",
            "config": {},
            "enabled": False,
            "role": "prefill",
        },
    )
    case.step(
        "fresh_request",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "fresh_request_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("fresh_request", "requests")},
    )
    case.step(
        "fresh_request_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("fresh_request", "requests")},
    )
    case.step(
        "fresh_request_success",
        "status_outcomes",
        params={
            "requests": output("fresh_request", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "after_fetch_error",
        "status_sample",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
        },
    )
    case.step(
        "after_fetch_error_scheduler",
        "status_check",
        params={
            "snapshot": output("after_fetch_error", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_fetch_error_prefill_batches",
        "status_check",
        params={
            "snapshot": output("after_fetch_error", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "after_fetch_error_decode_load",
        "status_check",
        params={
            "snapshot": output("after_fetch_error", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "engine_after_fetch", "status_sample", timeout_s=30, params={"duration_s": 0}
    )
    case.step(
        "prefill_engine_drained",
        "status_check",
        params={
            "snapshot": output("engine_after_fetch", "snapshot"),
            "metric": "prefill_engine_inflight",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params={
            "requests": output("recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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


def debug_snapshot(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "debug_before",
        "status_sample",
        timeout_s=30,
        params={"include": ["inflight", "mock", "debug"], "duration_s": 0},
    )
    case.step(
        "completed",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "completed_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("completed", "requests")},
    )
    case.step(
        "completed_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("completed", "requests")},
    )
    case.step(
        "completed_success",
        "status_outcomes",
        params={
            "requests": output("completed", "requests"),
            "success_min": 1,
            "failure_max": 0,
        },
    )
    case.step(
        "tombstone_window",
        "status_sample",
        timeout_s=30,
        params={
            "include": ["inflight", "mock", "debug"],
            "duration_s": 15,
            "requests": output("completed", "requests"),
            "until": {"metric": "scheduler_tombstones", "op": "eq", "value": 1},
        },
    )
    case.step(
        "resource_free_queryable_tombstone",
        "status_check",
        params={
            "snapshot": output("tombstone_window", "snapshot"),
            "metric": "scheduler_tombstones",
            "op": "eq",
            "expected": 1,
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


def normal_no_fetch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "fresh_baseline",
        "status_sample",
        timeout_s=30,
        params={"include": ["inflight", "mock", "debug"], "duration_s": 0},
    )
    case.step(
        "fresh_accepted_zero",
        "status_check",
        params={
            "snapshot": output("fresh_baseline", "snapshot"),
            "metric": "accepted",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "fresh_fetch_zero",
        "status_check",
        params={
            "snapshot": output("fresh_baseline", "snapshot"),
            "metric": "fetch_rpc",
            "op": "eq",
            "expected": 0,
            "aggregate": "last",
        },
    )
    case.step(
        "unfetched",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "deferred",
        },
    )
    case.step(
        "schedule_only",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("unfetched", "requests")},
    )
    case.step(
        "prefill_completion_window",
        "status_sample",
        timeout_s=35,
        params={
            "include": ["inflight", "mock", "debug"],
            "duration_s": 20,
            "requests": output("unfetched", "requests"),
            "until": {"metric": "cohort_prefill_completed", "op": "eq", "value": 1},
        },
    )
    case.step(
        "prefill_completed",
        "status_check",
        params={
            "snapshot": output("prefill_completion_window", "snapshot"),
            "metric": "cohort_prefill_completed",
            "op": "eq",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "master_enqueued",
        "status_check",
        params={
            "snapshot": output("prefill_completion_window", "snapshot"),
            "metric": "cohort_enqueued",
            "op": "eq",
            "expected": 1,
            "aggregate": "all",
        },
    )
    case.step(
        "completion_window_no_fetch",
        "status_check",
        params={
            "snapshot": output("prefill_completion_window", "snapshot"),
            "metric": "fetch_rpc",
            "op": "eq",
            "expected": 0,
            "aggregate": "all",
        },
    )
    case.step(
        "post_completion_window",
        "status_sample",
        timeout_s=30,
        params={
            "include": ["inflight", "mock", "debug"],
            "duration_s": 2,
            "requests": output("unfetched", "requests"),
        },
    )
    case.step(
        "prefill_stays_completed",
        "status_check",
        params={
            "snapshot": output("post_completion_window", "snapshot"),
            "metric": "cohort_prefill_completed",
            "op": "eq",
            "expected": 1,
            "aggregate": "all",
        },
    )
    case.step(
        "post_completion_no_fetch",
        "status_check",
        params={
            "snapshot": output("post_completion_window", "snapshot"),
            "metric": "fetch_rpc",
            "op": "eq",
            "expected": 0,
            "aggregate": "all",
        },
    )
    case.step(
        "client_never_fetches",
        "status_check",
        params={
            "snapshot": output("post_completion_window", "snapshot"),
            "metric": "cohort_fetch_invocations",
            "op": "eq",
            "expected": 0,
            "aggregate": "all",
        },
    )
    case.step(
        "one_prefill_acceptance",
        "status_check",
        params={
            "snapshot": output("post_completion_window", "snapshot"),
            "metric": "prefill_accepted",
            "op": "eq",
            "expected": 1,
            "aggregate": "last",
        },
    )
    case.step(
        "separate_recovery",
        "status_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 2048,
            "output_len": 2,
            "consume": "immediate",
        },
    )
    case.step(
        "separate_recovery_dispatch",
        "status_dispatch",
        timeout_s=75,
        params={"requests": output("separate_recovery", "requests")},
    )
    case.step(
        "separate_recovery_wait",
        "wait",
        timeout_s=75,
        params={"requests": output("separate_recovery", "requests")},
    )
    case.step(
        "separate_recovery_success",
        "status_outcomes",
        params={
            "requests": output("separate_recovery", "requests"),
            "success_min": 1,
            "failure_max": 0,
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
    "inflight_ttl_cleanup": {
        "build": inflight_ttl_cleanup,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "prefill_suppress_all": {
        "build": prefill_suppress_all,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "prefill_suppress_finished": {
        "build": prefill_suppress_finished,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "decode_suppress_finished": {
        "build": decode_suppress_finished,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "no_respond": {
        "build": no_respond,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "version_regress": {
        "build": version_regress,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "unknown_rid_finished": {
        "build": unknown_rid_finished,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "unknown_rid_running": {
        "build": unknown_rid_running,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "unknown_batchid": {
        "build": unknown_batchid,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "special_ids": {
        "build": special_ids,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "unbatched_single_request": {
        "build": unbatched_single_request,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "foreign_batchid": {
        "build": foreign_batchid,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "duplicate_finished": {
        "build": duplicate_finished,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "cursor_regress": {
        "build": cursor_regress,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "finished_then_running": {
        "build": finished_then_running,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "zombie_completed_running": {
        "build": zombie_completed_running,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "zombie_fake_running": {
        "build": zombie_fake_running,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "decode_before_prefill": {
        "build": decode_before_prefill,
        "profiles": ["batch-window"],
        "metadata": {
            "findings": ["decode_terminal_retires_prefill_promptly.contract"],
        },
    },
    "decode_running_before_prefill": {
        "build": decode_running_before_prefill,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "decode_waiting_before_prefill": {
        "build": decode_waiting_before_prefill,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "fetch_error": {
        "build": fetch_error,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "debug_snapshot": {
        "build": debug_snapshot,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "normal_no_fetch": {
        "build": normal_no_fetch,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
}
