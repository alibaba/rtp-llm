"""Explicit client routing programs: all-master outage, wraparound recovery, no fallback for business/deadline errors, and direct GenerateStreamCall fault recovery."""

from ..case_config import output

METADATA = {
    "id": "client_fallback_failback",
    "description": "Explicit client routing programs: all-master outage, wraparound recovery, no "
    "fallback for business/deadline errors, and direct GenerateStreamCall fault "
    "recovery.",
    "category": "master",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def all_masters_down(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow",
        "master_client_start",
        params={"targets": ["A", "B"], "duration_s": 60, "fallback": True},
    )
    case.step("kill_a_time", "master_mark", params={"wait_s": 12})
    case.step("kill_a", "master_fault", params={"mode": "kill", "target": "A"})
    case.step("kill_b_time", "master_mark", params={"wait_s": 0.5})
    case.step("kill_b", "master_fault", params={"mode": "kill", "target": "B"})
    case.step("outage_end", "master_mark", params={"wait_s": 10})
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=90,
        params={"client": output("flow", "client")},
    )
    case.step(
        "steady",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "until": output("kill_a_time", "epoch_s"),
        },
    )
    case.step(
        "outage",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("kill_b_time", "epoch_s"),
            "until": output("outage_end", "epoch_s"),
        },
    )
    case.step(
        "fallback_rows",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("kill_b_time", "epoch_s"),
            "until": output("outage_end", "epoch_s"),
            "route": "fallback",
        },
    )
    case.step(
        "steady_master",
        "master_client_check",
        params={
            "rows": output("steady", "rows"),
            "metric": "route_share",
            "op": "eq",
            "expected": 1,
            "route": "master",
            "min_samples": 10,
        },
    )
    case.step(
        "fallback_success",
        "master_client_check",
        params={
            "rows": output("fallback_rows", "rows"),
            "metric": "success_rate",
            "op": "ge",
            "expected": 0.9,
            "min_samples": 10,
        },
    )
    case.step(
        "fallback_share",
        "master_client_check",
        params={
            "rows": output("outage", "rows"),
            "metric": "route_share",
            "op": "ge",
            "expected": 0.8,
            "route": "fallback",
        },
    )
    case.step(
        "no_master_during_outage",
        "master_client_check",
        params={
            "rows": output("outage", "rows"),
            "metric": "route_count",
            "op": "eq",
            "expected": 0,
            "route": "master",
        },
    )
    case.step(
        "outage_errors",
        "master_client_check",
        params={
            "rows": output("outage", "rows"),
            "metric": "failed_rate_above_one",
            "op": "le",
            "expected": 0.05,
        },
    )
    case.step(
        "unique_requests",
        "master_client_check",
        params={
            "rows": output("finish", "rows"),
            "metric": "duplicate_ids",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def wraparound(case):
    window = case.number("checkpoint_window_s", 5, minimum=3, maximum=10)
    owner_limit = case.number("max_owner_load", 128, maximum=5000)
    flow = output("flow", "client")

    def checkpoint(name, target, *, fault=None, baseline=True):
        params = {
            "client": flow,
            "target": target,
            "duration_s": window,
            "min_samples": 30,
            "max_owner_load": owner_limit,
        }
        if fault:
            params.update(
                fault=output(fault, "fault"),
                min_target_share=0.5,
                min_success=0.9,
                max_prefill_share=1,
            )
        if baseline:
            params["baseline"] = output("baseline_a", "snapshot")
        case.step(
            name, "master_client_checkpoint", timeout_s=window + 35, params=params
        )

    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow",
        "master_client_start",
        params={"targets": ["A", "B"], "duration_s": 150, "live_events": True},
    )
    checkpoint("baseline_a", "A", baseline=False)
    case.step("kill_a", "master_fault", params={"mode": "kill", "target": "A"})
    checkpoint("switch_to_b", "B", fault="kill_a")
    checkpoint("steady_b", "B")
    case.step(
        "restart_a",
        "master_restore",
        timeout_s=180,
        params={"fault": output("kill_a", "fault")},
    )
    case.step("ready_a", "master_topology_ready", timeout_s=60, params={"target": "A"})
    case.step(
        "recovery_a",
        "master_request_batch",
        timeout_s=330,
        params={"target": "A", "count": 20, "concurrency": 10},
    )
    case.step(
        "recovery_distribution",
        "master_probe_distribution",
        params={"requests": output("recovery_a", "requests")},
    )
    # A owns no active requests here; B still drives their shared engines.
    case.step("idle_a", "master_owner_idle", timeout_s=30, params={"target": "A"})
    checkpoint("coexist_sticky_b", "B")
    case.step("kill_b", "master_fault", params={"mode": "kill", "target": "B"})
    checkpoint("switch_to_a", "A", fault="kill_b")
    checkpoint("steady_a", "A")
    case.step("finish", "master_client_finish", timeout_s=180, params={"client": flow})
    case.step(
        "all_requests_accounted",
        "master_client_reconcile",
        params={"client": flow, "rows": output("finish", "rows")},
    )
    case.step(
        "overall_success",
        "master_client_check",
        params={
            "rows": output("finish", "rows"),
            "metric": "success_rate",
            "op": "ge",
            "expected": 0.95,
        },
    )
    case.step("clean_a", "master_inflight_clean", timeout_s=35, params={"target": "A"})
    case.step("cleanup", "teardown")


def negative_errorcode(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "flow",
        "master_client_start",
        params={
            "targets": ["A", "B"],
            "duration_s": 75,
            "timeout_ms": 800,
            "fallback": True,
        },
    )
    case.step("freeze_begin", "master_mark", params={"wait_s": 10})
    case.step("freeze_a", "master_fault", params={"target": "A", "mode": "freeze"})
    case.step("freeze_end", "master_mark", params={"wait_s": 8})
    case.step("thaw_a", "master_restore", params={"fault": output("freeze_a", "fault")})
    case.step("settled", "master_mark", params={"wait_s": 8})
    case.step(
        "business_fault",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "enqueue_ack_error_code",
            "options": {"code": 8431},
        },
    )
    case.step("business_begin", "master_mark", params={"wait_s": 0})
    case.step("business_end", "master_mark", params={"wait_s": 10})
    case.step(
        "clear_business",
        "engine_clear",
        params={"fault": output("business_fault", "fault")},
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=120,
        params={"client": output("flow", "client")},
    )
    case.step(
        "business",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("business_begin", "epoch_s"),
            "until": output("business_end", "epoch_s"),
        },
    )
    case.step(
        "schedule_errors",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("business_begin", "epoch_s"),
            "until": output("business_end", "epoch_s"),
            "status": "schedule_error",
        },
    )
    case.step(
        "deadline_window",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("freeze_begin", "epoch_s"),
            "until": output("freeze_end", "epoch_s"),
        },
    )
    case.step(
        "deadlines",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("freeze_begin", "epoch_s"),
            "until": output("freeze_end", "epoch_s"),
            "error_kind": "deadline",
        },
    )
    case.step(
        "business_errors_seen",
        "master_client_check",
        params={
            "rows": output("schedule_errors", "rows"),
            "metric": "sample_count",
            "op": "ge",
            "expected": 5,
            "min_samples": 5,
        },
    )
    case.step(
        "business_code",
        "master_client_check",
        params={
            "rows": output("schedule_errors", "rows"),
            "metric": "wrong_error_code",
            "op": "eq",
            "expected": 0,
            "code": 8431,
            "min_samples": 5,
        },
    )
    case.step(
        "business_route",
        "master_client_check",
        params={
            "rows": output("business", "rows"),
            "metric": "route_share",
            "op": "eq",
            "expected": 1,
            "route": "master",
        },
    )
    case.step(
        "business_no_fallback",
        "master_client_check",
        params={
            "rows": output("business", "rows"),
            "metric": "route_count",
            "op": "eq",
            "expected": 0,
            "route": "fallback",
        },
    )
    case.step(
        "business_no_failed",
        "master_client_check",
        params={
            "rows": output("business", "rows"),
            "metric": "route_count",
            "op": "eq",
            "expected": 0,
            "route": "failed",
        },
    )
    case.step(
        "business_no_retry",
        "master_client_check",
        params={
            "rows": output("business", "rows"),
            "metric": "failover_count",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "deadlines_seen",
        "master_client_check",
        params={
            "rows": output("deadlines", "rows"),
            "metric": "sample_count",
            "op": "ge",
            "expected": 3,
            "min_samples": 3,
        },
    )
    case.step(
        "deadline_route",
        "master_client_check",
        params={
            "rows": output("deadlines", "rows"),
            "metric": "route_share",
            "op": "eq",
            "expected": 1,
            "route": "failed",
            "min_samples": 3,
        },
    )
    case.step(
        "deadline_no_retry",
        "master_client_check",
        params={
            "rows": output("deadlines", "rows"),
            "metric": "failover_count",
            "op": "eq",
            "expected": 0,
            "min_samples": 3,
        },
    )
    case.step(
        "deadline_no_fallback",
        "master_client_check",
        params={
            "rows": output("deadline_window", "rows"),
            "metric": "route_count",
            "op": "eq",
            "expected": 0,
            "route": "fallback",
        },
    )
    case.step(
        "tail_settle",
        "master_wait_inflight",
        timeout_s=150,
        params={"target": "A", "op": "le", "value": 8},
    )
    case.step("cleanup", "teardown")


def direct_generate_error(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "baseline",
        "master_direct_request",
        timeout_s=30,
        params={"engine": "prefill-0"},
    )
    case.step(
        "baseline_finished",
        "check",
        params={"actual": output("baseline", "finished"), "op": "eq", "expected": True},
    )
    case.step(
        "baseline_no_error",
        "check",
        params={"actual": output("baseline", "error"), "op": "eq", "expected": False},
    )
    case.step(
        "generate_fault",
        "engine_inject",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "type": "generate_error",
            "options": {},
        },
    )
    case.step(
        "injected",
        "master_direct_request",
        timeout_s=30,
        params={"engine": "prefill-0"},
    )
    case.step(
        "injected_error",
        "check",
        params={"actual": output("injected", "error"), "op": "eq", "expected": True},
    )
    case.step("injected_clean", "master_direct_clean")
    case.step(
        "clear_generate",
        "engine_clear",
        params={"fault": output("generate_fault", "fault")},
    )
    case.step(
        "recovery",
        "master_direct_request",
        timeout_s=30,
        params={"engine": "prefill-0"},
    )
    case.step(
        "recovered_finished",
        "check",
        params={"actual": output("recovery", "finished"), "op": "eq", "expected": True},
    )
    case.step(
        "recovered_no_error",
        "check",
        params={"actual": output("recovery", "error"), "op": "eq", "expected": False},
    )
    case.step("recovered_clean", "master_direct_clean")
    case.step("cleanup", "teardown")


VARIANTS = {
    "all_masters_down": {
        "build": all_masters_down,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "wraparound": {
        "build": wraparound,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "negative_errorcode": {
        "build": negative_errorcode,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "direct_generate_error": {
        "build": direct_generate_error,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
