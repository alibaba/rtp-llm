"""Nine engine recovery contracts with explicit generation, cache, transport and resource-owner evidence."""

from ..case_config import output

METADATA = {
    "id": "engine_fault_recovery",
    "description": "Nine engine recovery contracts with explicit generation, cache, transport and "
    "resource-owner evidence.",
    "category": "engine_fault",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def generation_bump(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step("target", "recovery_select", params={"targets": ["prefill-0"]})
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params={"count": 6, "concurrency": 6, "unique_key_count": 3},
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("baseline", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params={
            "snapshot": output("baseline_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 6,
        },
    )
    case.step(
        "outage",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "retired", "op": "ge", "value": 1},
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
            "interval_s": 0.2,
        },
    )
    case.step(
        "transport_retired",
        "recovery_check",
        params={
            "snapshot": output("retirement", "snapshot"),
            "metric": "retired",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "alive_back",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "recovered_generation",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log", "inflight"],
        },
    )
    case.step(
        "generation_is_new",
        "recovery_check",
        params={
            "snapshot": output("recovered_generation", "snapshot"),
            "metric": "created",
            "op": "ge",
            "expected": 1,
            "baseline": output("generation_before", "snapshot"),
        },
    )
    case.step(
        "recovered_prefill_batch_ledger_zero",
        "recovery_check",
        params={
            "snapshot": output("recovered_generation", "snapshot"),
            "metric": "target_prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "recovered_prefill_member_ledger_zero",
        "recovery_check",
        params={
            "snapshot": output("recovered_generation", "snapshot"),
            "metric": "target_prefill_requests",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 30},
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params={
            "snapshot": output("recovery_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step("cleanup", "teardown")


def status_gap_no_bump(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step("target", "recovery_select", params={"targets": ["prefill-0"]})
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params={"count": 4, "concurrency": 4, "unique_key_count": 3},
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("baseline", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params={
            "snapshot": output("baseline_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 4,
        },
    )
    case.step(
        "short_gap",
        "status_control",
        params={"fault": "status_no_respond", "selection": "first"},
    )
    case.step(
        "two_poll_ticks", "recovery_pause", timeout_s=30, params={"duration_s": 0.045}
    )
    case.step(
        "resume_status",
        "status_control",
        params={"fault": "status_no_respond", "selection": "first", "enabled": False},
    )
    case.step(
        "hung_rpc_deadline_and_resume",
        "recovery_pause",
        timeout_s=30,
        params={"duration_s": 2},
    )
    case.step(
        "post_gap_generation",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log", "info"],
        },
    )
    case.step(
        "no_new_generation",
        "recovery_check",
        params={
            "snapshot": output("post_gap_generation", "snapshot"),
            "metric": "created",
            "op": "le",
            "expected": 0,
            "baseline": output("generation_before", "snapshot"),
        },
    )
    case.step(
        "discovered_intact",
        "recovery_check",
        params={
            "snapshot": output("post_gap_generation", "snapshot"),
            "metric": "discovered_prefill",
            "op": "eq",
            "expected": 2,
        },
    )
    case.step(
        "alive_intact",
        "recovery_check",
        params={
            "snapshot": output("post_gap_generation", "snapshot"),
            "metric": "alive_prefill",
            "op": "eq",
            "expected": 2,
        },
    )
    case.step(
        "post_gap",
        "recovery_prepare",
        params={"count": 4, "concurrency": 4, "unique_key_count": 3},
    )
    case.step(
        "post_gap_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("post_gap", "requests")},
    )
    case.step(
        "post_gap_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("post_gap", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "post_gap_succeeds",
        "recovery_check",
        params={
            "snapshot": output("post_gap_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 4,
        },
    )
    case.step("cleanup", "teardown")


def down_phases(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params={
            "count": 20,
            "concurrency": 10,
            "unique_key_count": 3,
            "generate_payload": "legacy_default",
            "stream_timeout_s": 15,
            "measure_ttft": True,
        },
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=150,
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("baseline", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params={
            "snapshot": output("baseline_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 20,
        },
    )
    case.step(
        "baseline_health",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "baseline_health_http_200",
        "recovery_check",
        params={
            "snapshot": output("baseline_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "outage",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "eviction",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 1},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "survivor_only_alive",
        "recovery_check",
        params={
            "snapshot": output("eviction", "snapshot"),
            "metric": "alive_prefill",
            "op": "le",
            "expected": 1,
        },
    )
    case.step(
        "takeover",
        "recovery_prepare",
        params={"count": 20, "concurrency": 10, "unique_key_count": 3},
    )
    case.step(
        "takeover_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("takeover", "requests")},
    )
    case.step(
        "takeover_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("takeover", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "downtime_health",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "downtime_health_http_200",
        "recovery_check",
        params={
            "snapshot": output("downtime_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "takeover_at_least_ninety_percent",
        "recovery_check",
        params={
            "snapshot": output("takeover_state", "snapshot"),
            "metric": "success",
            "op": "ge",
            "expected": 18,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "alive_back",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "recovery_batch",
        "recovery_prepare",
        params={
            "count": 20,
            "concurrency": 10,
            "unique_key_count": 3,
            "generate_payload": "legacy_default",
            "stream_timeout_s": 15,
            "measure_ttft": True,
        },
    )
    case.step(
        "recovery_batch_dispatch",
        "recovery_dispatch",
        timeout_s=150,
        params={"requests": output("recovery_batch", "requests")},
    )
    case.step(
        "recovery_batch_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery_batch", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_at_least_ninety_five_percent",
        "recovery_check",
        params={
            "snapshot": output("recovery_batch_state", "snapshot"),
            "metric": "success",
            "op": "ge",
            "expected": 19,
        },
    )
    case.step(
        "ttft_recovers",
        "recovery_ttft_check",
        params={
            "baseline": output("baseline_state", "snapshot"),
            "recovered": output("recovery_batch_state", "snapshot"),
        },
    )
    case.step(
        "recovered_health",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "recovered_health_http_200",
        "recovery_check",
        params={
            "snapshot": output("recovered_health", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "closing_drain",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "closing_drain_scheduler",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "closing_drain_decode_load",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def flap(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("background_flow", "elastic_cold_flow")
    case.step("flow_ramp", "recovery_pause", timeout_s=30, params={"duration_s": 1})
    case.step(
        "stop_1",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_1", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_1",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_1",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_1_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_1", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_1",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_1", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_2",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_2", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_2",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_2",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_2_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_2", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_2",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_2", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_3",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_3", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_3",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_3",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_3_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_3", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_3",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_3", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_4",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_4", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_4",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_4",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_4_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_4", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_4",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_4", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_5",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_5", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_5",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_5",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_5_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_5", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_5",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_5", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_6",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "down_window_6", "recovery_pause", timeout_s=30, params={"duration_s": 0.8}
    )
    case.step(
        "alive_during_6",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["info"]},
    )
    case.step(
        "master_cycle_6",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["inflight"]},
    )
    case.step(
        "master_cycle_6_http_200",
        "recovery_check",
        params={
            "snapshot": output("master_cycle_6", "snapshot"),
            "metric": "master_http",
            "op": "eq",
            "expected": 200,
        },
    )
    case.step(
        "start_6",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step("up_gap_6", "recovery_pause", timeout_s=30, params={"duration_s": 0.4})
    case.step(
        "stop_flow",
        "recovery_flow_stop",
        timeout_s=50,
        params={"flow": output("background_flow", "flow")},
    )
    case.step(
        "flow_availability",
        "recovery_flow_assert",
        params={"result": output("stop_flow", "result"), "min_success_rate": 0.5},
    )
    case.step(
        "topology",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "topology_mismatch", "op": "eq", "value": 0},
            "include": ["info"],
            "interval_s": 0.2,
        },
    )
    case.step(
        "topology_discovered",
        "recovery_check",
        params={
            "snapshot": output("topology", "snapshot"),
            "metric": "discovered_prefill",
            "op": "eq",
            "expected": 2,
        },
    )
    case.step(
        "topology_alive",
        "recovery_check",
        params={
            "snapshot": output("topology", "snapshot"),
            "metric": "alive_prefill",
            "op": "eq",
            "expected": 2,
        },
    )
    case.step(
        "post_flap",
        "recovery_prepare",
        params={"count": 20, "concurrency": 10, "unique_key_count": 3},
    )
    case.step(
        "post_flap_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("post_flap", "requests")},
    )
    case.step(
        "post_flap_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("post_flap", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "post_flap_recovers",
        "recovery_check",
        params={
            "snapshot": output("post_flap_state", "snapshot"),
            "metric": "success",
            "op": "ge",
            "expected": 19,
        },
    )
    case.step(
        "closing_drain",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "closing_drain_scheduler",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "closing_drain_prefill_batches",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "closing_drain_decode_load",
        "recovery_check",
        params={
            "snapshot": output("closing_drain", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def kv_resync(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "seed",
        "recovery_prepare",
        params={
            "count": 1,
            "concurrency": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
        },
    )
    case.step(
        "seed_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("seed", "requests")},
    )
    case.step(
        "seed_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("seed", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "seed_succeeds",
        "recovery_check",
        params={
            "snapshot": output("seed_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step(
        "seed_landing", "kv_landing", params={"requests": output("seed", "requests")}
    )
    case.step(
        "holder",
        "recovery_select",
        params={"targets": [output("seed_landing", "engine")]},
    )
    case.step(
        "seed_ownership",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 8,
            "keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
            "until": {"metric": "target_family_overlap", "op": "eq", "value": 10},
            "selection": output("holder", "selection"),
            "include": ["mock"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "holder_has_whole_family",
        "recovery_check",
        params={
            "snapshot": output("seed_ownership", "snapshot"),
            "metric": "target_family_overlap",
            "op": "eq",
            "expected": 10,
        },
    )
    case.step(
        "cache_sync_before_outage",
        "recovery_pause",
        timeout_s=30,
        params={"duration_s": 4.5},
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("holder", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "stop_holder",
        "engine_control",
        params={"operation": "stop", "targets": [output("holder", "engine")]},
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "retired", "op": "ge", "value": 1},
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
            "interval_s": 0.2,
        },
    )
    case.step(
        "holder_retired",
        "recovery_check",
        params={
            "snapshot": output("retirement", "snapshot"),
            "metric": "retired",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "restart_holder",
        "engine_control",
        params={"operation": "start", "targets": [output("holder", "engine")]},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "holder_alive_back",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "new_generation",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "holder_generation_bumped",
        "recovery_check",
        params={
            "snapshot": output("new_generation", "snapshot"),
            "metric": "created",
            "op": "ge",
            "expected": 1,
            "baseline": output("generation_before", "snapshot"),
        },
    )
    case.step(
        "intact_wave",
        "recovery_prepare",
        params={
            "count": 5,
            "concurrency": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
        },
    )
    case.step(
        "intact_wave_dispatch",
        "recovery_dispatch",
        timeout_s=255,
        params={"requests": output("intact_wave", "requests")},
    )
    case.step(
        "intact_wave_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("intact_wave", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "intact_routes",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("holder", "selection"),
            "requests": output("intact_wave", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "memory_intact_holder_survives",
        "recovery_check",
        params={
            "snapshot": output("intact_routes", "snapshot"),
            "metric": "target_success_landings",
            "op": "ge",
            "expected": 4,
        },
    )
    case.step(
        "wipe_family",
        "recovery_cache_control",
        params={
            "selection": output("holder", "selection"),
            "operation": "evict",
            "keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
        },
    )
    case.step(
        "family_gone",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 8,
            "keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
            "until": {"metric": "target_family_overlap", "op": "eq", "value": 0},
            "selection": output("holder", "selection"),
            "include": ["mock"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "wiped_family_absent",
        "recovery_check",
        params={
            "snapshot": output("family_gone", "snapshot"),
            "metric": "target_family_overlap",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "cache_sync_after_wipe",
        "recovery_pause",
        timeout_s=30,
        params={"duration_s": 4.5},
    )
    case.step(
        "wiped_wave",
        "recovery_prepare",
        params={
            "count": 5,
            "concurrency": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                900001,
                900002,
                900003,
                900004,
                900005,
                900006,
                900007,
                900008,
                900009,
                900010,
            ],
        },
    )
    case.step(
        "wiped_wave_dispatch",
        "recovery_dispatch",
        timeout_s=255,
        params={"requests": output("wiped_wave", "requests")},
    )
    case.step(
        "wiped_wave_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("wiped_wave", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "wiped_routes",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("holder", "selection"),
            "requests": output("wiped_wave", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "memory_lost_old_holder_spreads",
        "recovery_check",
        params={
            "snapshot": output("wiped_routes", "snapshot"),
            "metric": "target_success_landings",
            "op": "le",
            "expected": 3,
        },
    )
    case.step("cleanup", "teardown")


def kv_usage_reset(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step("target", "recovery_select", params={"targets": ["prefill-0"]})
    case.step(
        "evict_current_cache",
        "recovery_cache_control",
        params={"selection": output("target", "selection"), "operation": "evict"},
    )
    case.step(
        "empty_cache_sync", "recovery_pause", timeout_s=30, params={"duration_s": 4.5}
    )
    case.step(
        "empty_usage",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "include": ["mock"],
        },
    )
    case.step(
        "baseline_usage_zero",
        "recovery_check",
        params={
            "snapshot": output("empty_usage", "snapshot"),
            "metric": "target_kv_tokens",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "set_pressure",
        "recovery_cache_control",
        params={
            "selection": output("target", "selection"),
            "operation": "absolute_pressure",
            "tokens": 4000000,
        },
    )
    case.step(
        "pressure_reported",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 8,
            "until": {"metric": "target_kv_tokens", "op": "ge", "value": 4000000},
            "selection": output("target", "selection"),
            "include": ["mock"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "pressure_construction_observed",
        "recovery_check",
        params={
            "snapshot": output("pressure_reported", "snapshot"),
            "metric": "target_kv_tokens",
            "op": "ge",
            "expected": 4000000,
        },
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "outage",
        "engine_control",
        params={"operation": "stop", "targets": ["prefill-0"]},
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "retired", "op": "ge", "value": 1},
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
            "interval_s": 0.2,
        },
    )
    case.step(
        "pressure_generation_retired",
        "recovery_check",
        params={
            "snapshot": output("retirement", "snapshot"),
            "metric": "retired",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "restart",
        "engine_control",
        params={"operation": "start", "targets": ["prefill-0"]},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "alive_back",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "recovered_generation",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log", "mock"],
        },
    )
    case.step(
        "new_capacity_generation",
        "recovery_check",
        params={
            "snapshot": output("recovered_generation", "snapshot"),
            "metric": "created",
            "op": "ge",
            "expected": 1,
            "baseline": output("generation_before", "snapshot"),
        },
    )
    case.step(
        "lack_mem_baseline",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["mock"]},
    )
    case.step(
        "first_wave",
        "recovery_prepare",
        params={
            "count": 3,
            "concurrency": 1,
            "unique_key_count": 1,
            "unique_key_start": 1,
            "stream_timeout_s": 10,
        },
    )
    case.step(
        "first_wave_dispatch",
        "recovery_dispatch",
        timeout_s=150,
        params={"requests": output("first_wave", "requests")},
    )
    case.step(
        "first_wave_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("first_wave", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "first_wave_succeeds",
        "recovery_check",
        params={
            "snapshot": output("first_wave_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 3,
        },
    )
    case.step(
        "lack_mem_after",
        "recovery_observe",
        timeout_s=30,
        params={"duration_s": 0, "include": ["mock"]},
    )
    case.step(
        "first_wave_no_lack_mem",
        "recovery_check",
        params={
            "snapshot": output("lack_mem_after", "snapshot"),
            "metric": "prefill_lack_mem",
            "op": "eq",
            "expected": 0,
            "baseline": output("lack_mem_baseline", "snapshot"),
        },
    )
    case.step(
        "target_receives_traffic",
        "recovery_pump",
        timeout_s=90,
        params={"selection": output("target", "selection"), "duration_s": 15},
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 30},
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params={
            "snapshot": output("recovery_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step("cleanup", "teardown")


def crash_after(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prefills", "recovery_select", params={"targets": ["prefill-0", "prefill-1"]}
    )
    case.step(
        "arm_first_enqueue",
        "recovery_crash_arm",
        params={"selection": output("prefills", "selection"), "mode": "first_enqueue"},
    )
    case.step(
        "trigger",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 12, "output_len": 10},
    )
    case.step(
        "trigger_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("trigger", "requests")},
    )
    case.step(
        "trigger_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("trigger", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "crash_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("prefills", "selection"),
            "include": ["mock"],
        },
    )
    case.step(
        "crashed",
        "recovery_partition",
        params={
            "selection": output("prefills", "selection"),
            "snapshot": output("crash_state", "snapshot"),
            "stopped": True,
        },
    )
    case.step(
        "survivors",
        "recovery_partition",
        params={
            "selection": output("prefills", "selection"),
            "snapshot": output("crash_state", "snapshot"),
            "stopped": False,
        },
    )
    case.step(
        "exactly_one_crashed",
        "check",
        params={"actual": output("crashed", "count"), "op": "eq", "expected": 1},
    )
    case.step(
        "disarm_survivors",
        "recovery_crash_arm",
        params={"selection": output("survivors", "selection"), "mode": "disarm"},
    )
    case.step(
        "alive_dropped",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "le", "value": 1},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "master_observed_loss",
        "recovery_check",
        params={
            "snapshot": output("alive_dropped", "snapshot"),
            "metric": "alive_prefill",
            "op": "le",
            "expected": 1,
        },
    )
    case.step(
        "routable_set_settle", "recovery_pause", timeout_s=30, params={"duration_s": 2}
    )
    case.step(
        "takeover",
        "recovery_prepare",
        params={"count": 5, "concurrency": 5, "stream_timeout_s": 12, "output_len": 10},
    )
    case.step(
        "takeover_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("takeover", "requests")},
    )
    case.step(
        "takeover_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("takeover", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "survivor_serves_sixty_percent",
        "recovery_check",
        params={
            "snapshot": output("takeover_state", "snapshot"),
            "metric": "success",
            "op": "ge",
            "expected": 3,
        },
    )
    case.step(
        "restart_crashed",
        "recovery_engine_control",
        params={"selection": output("crashed", "selection"), "operation": "start"},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "crashed_engine_rediscovers",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "recovery",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 12, "output_len": 10},
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params={
            "snapshot": output("recovery_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step(
        "residue_window",
        "recovery_residue",
        timeout_s=40,
        params={
            "requests": output("takeover", "requests"),
            "base_residue": 1,
            "settle_s": 20,
            "stable_s": 8,
        },
    )
    case.step(
        "prefill_engines_clean",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "target_engine_clean_total", "op": "eq", "value": 0},
            "selection": output("prefills", "selection"),
            "include": ["mock"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "prefill_engines_clean_inflight",
        "recovery_check",
        params={
            "snapshot": output("prefill_engines_clean", "snapshot"),
            "metric": "target_inflight",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "prefill_engines_clean_leaks",
        "recovery_check",
        params={
            "snapshot": output("prefill_engines_clean", "snapshot"),
            "metric": "target_leaks",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


def no_resurrect(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "prefills", "recovery_select", params={"targets": ["prefill-0", "prefill-1"]}
    )
    case.step(
        "slow_prefill",
        "recovery_engine_control",
        params={
            "selection": output("prefills", "selection"),
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 2000},
            "restore_perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("perf_sync", "recovery_pause", timeout_s=30, params={"duration_s": 1.5})
    case.step(
        "fired",
        "recovery_prepare",
        params={
            "count": 8,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "manual",
            "stream_timeout_s": 60,
            "generate_payload": "legacy_default",
        },
    )
    case.step(
        "fired_dispatch",
        "recovery_dispatch",
        timeout_s=300,
        params={"requests": output("fired", "requests")},
    )
    case.step(
        "fired_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("fired", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "routed_payload_minimum",
        "recovery_check",
        params={
            "snapshot": output("fired_state", "snapshot"),
            "metric": "schedule_ok",
            "op": "ge",
            "expected": 4,
        },
    )
    case.step(
        "batch_dispatch", "recovery_pause", timeout_s=30, params={"duration_s": 0.5}
    )
    case.step(
        "targets",
        "recovery_select_routed",
        params={"requests": output("fired", "requests")},
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("targets", "selection")},
    )
    case.step(
        "arm_next_enqueue",
        "recovery_crash_arm",
        params={"selection": output("targets", "selection"), "mode": "next_enqueue"},
    )
    case.step(
        "trigger_all_targets",
        "recovery_crash_trigger",
        timeout_s=50,
        params={"selection": output("targets", "selection"), "duration_s": 15},
    )
    case.step(
        "all_targets_crashed",
        "recovery_all_targets_check",
        params={
            "snapshot": output("trigger_all_targets", "snapshot"),
            "metric": "target_stopped",
        },
    )
    case.step(
        "retire_all",
        "recovery_retire_all",
        timeout_s=90,
        params={
            "selection": output("targets", "selection"),
            "log": output("log_mark", "snapshot"),
            "per_target_s": 30,
        },
    )
    case.step(
        "restart_targets",
        "recovery_engine_control",
        params={"selection": output("targets", "selection"), "operation": "start"},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "all_prefills_alive",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "pretraffic_reset",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("targets", "selection"),
            "include": ["inflight", "mock"],
        },
    )
    case.step(
        "new_prefill_batches_zero",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "new_prefill_members_zero",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_prefill_requests",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "wiped_running",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_running",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "wiped_inflight",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_inflight",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "wiped_cache_keys",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_cache_keys",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "wiped_held_blocks",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_held_blocks",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "wiped_accepted",
        "recovery_check",
        params={
            "snapshot": output("pretraffic_reset", "snapshot"),
            "metric": "target_accepted",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "consume_old_payload",
        "recovery_consume",
        timeout_s=180,
        params={"requests": output("fired", "requests"), "wait_s": 2},
    )
    case.step(
        "old_requests_never_complete",
        "recovery_check",
        params={
            "snapshot": output("consume_old_payload", "snapshot"),
            "metric": "observed_completed",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "engine_drain",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 10,
            "until": {"metric": "target_engine_clean_total", "op": "eq", "value": 0},
            "selection": output("targets", "selection"),
            "include": ["mock"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "engine_drain_inflight",
        "recovery_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "target_inflight",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "engine_drain_leaks",
        "recovery_check",
        params={
            "snapshot": output("engine_drain", "snapshot"),
            "metric": "target_leaks",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_drain",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "master_drain_scheduler",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_drain_decode_load",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 30},
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params={
            "snapshot": output("recovery_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step("cleanup", "teardown")


def status_gap_long_retire(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step("target", "recovery_select", params={"targets": ["prefill-0"]})
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "selection": output("target", "selection"),
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "prefills", "recovery_select", params={"targets": ["prefill-0", "prefill-1"]}
    )
    case.step(
        "slow_prefill",
        "recovery_engine_control",
        params={
            "selection": output("prefills", "selection"),
            "operation": "set_perf",
            "perf": {"prefill_fixed_ms": 2000},
            "restore_perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("perf_sync", "recovery_pause", timeout_s=30, params={"duration_s": 1.5})
    case.step(
        "fired",
        "recovery_prepare",
        params={
            "count": 8,
            "concurrency": 1,
            "input_len": 512,
            "output_len": 2,
            "consume": "manual",
            "stream_timeout_s": 60,
            "generate_payload": "legacy_default",
        },
    )
    case.step(
        "fired_dispatch",
        "recovery_dispatch",
        timeout_s=300,
        params={"requests": output("fired", "requests")},
    )
    case.step(
        "fired_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("fired", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "batch_dispatch", "recovery_pause", timeout_s=30, params={"duration_s": 0.5}
    )
    case.step(
        "long_gap",
        "status_control",
        params={"fault": "status_no_respond", "selection": "first"},
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 15,
            "until": {"metric": "retired", "op": "ge", "value": 1},
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
            "interval_s": 0.2,
        },
    )
    case.step(
        "long_gap_retires",
        "recovery_check",
        params={
            "snapshot": output("retirement", "snapshot"),
            "metric": "retired",
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "post_retire_hold", "recovery_pause", timeout_s=30, params={"duration_s": 1}
    )
    case.step(
        "resume_status",
        "status_control",
        params={"fault": "status_no_respond", "selection": "first", "enabled": False},
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=45,
        params={
            "duration_s": 30,
            "until": {"metric": "alive_prefill", "op": "ge", "value": 2},
            "include": ["info"],
            "interval_s": 0.5,
        },
    )
    case.step(
        "prefill_alive_back",
        "recovery_check",
        params={
            "snapshot": output("alive_restored", "snapshot"),
            "metric": "alive_prefill",
            "op": "ge",
            "expected": 2,
        },
    )
    case.step("reconnect", "recovery_pause", timeout_s=30, params={"duration_s": 3})
    case.step(
        "generation_after",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "log": output("log_mark", "snapshot"),
            "include": ["log"],
        },
    )
    case.step(
        "long_gap_creates_generation",
        "recovery_check",
        params={
            "snapshot": output("generation_after", "snapshot"),
            "metric": "created",
            "op": "ge",
            "expected": 1,
            "baseline": output("generation_before", "snapshot"),
        },
    )
    case.step(
        "consume_fence_payload_observed",
        "recovery_consume",
        timeout_s=210,
        params={"requests": output("fired", "requests"), "wait_s": 5},
    )
    case.step(
        "master_drain",
        "recovery_observe",
        timeout_s=110,
        params={
            "duration_s": 95,
            "until": {"metric": "cleanup_inflight", "op": "eq", "value": 0},
            "interval_s": 0.5,
        },
    )
    case.step(
        "master_drain_scheduler",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "scheduler",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_drain_prefill_batches",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "prefill_batches",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "master_drain_decode_load",
        "recovery_check",
        params={
            "snapshot": output("master_drain", "snapshot"),
            "metric": "decode_total_load",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params={"count": 1, "concurrency": 1, "stream_timeout_s": 30},
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=120,
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=30,
        params={
            "duration_s": 0,
            "requests": output("recovery", "requests"),
            "include": ["client_records"],
        },
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params={
            "snapshot": output("recovery_state", "snapshot"),
            "metric": "success",
            "op": "eq",
            "expected": 1,
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "generation_bump": {
        "build": generation_bump,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "status_gap_no_bump": {
        "build": status_gap_no_bump,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "down_phases": {
        "build": down_phases,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "flap": {
        "build": flap,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "kv_resync": {
        "build": kv_resync,
        "profiles": ["batch-window", "single-batch", "window-nonbatch"],
        "metadata": {},
    },
    "kv_usage_reset": {
        "build": kv_usage_reset,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "crash_after": {
        "build": crash_after,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "no_resurrect": {
        "build": no_resurrect,
        "profiles": ["batch-window", "single-batch"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "status_gap_long_retire": {
        "build": status_gap_long_retire,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
