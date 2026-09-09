"""Nine engine recovery contracts with explicit generation, cache, transport and resource-owner evidence."""

from ..case_config import output


def generation_bump(case):
    case.step("setup", "setup", timeout_s=case.value("generation_bump.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("generation_bump.prior_drain_observed_timeout_s"),
        params=case.value("generation_bump.prior_drain_observed"),
    )
    case.step("target", "recovery_select", params=case.value("generation_bump.target"))
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("generation_bump.generation_before_timeout_s"),
        params=case.params(
            "generation_bump.generation_before",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params=case.value("generation_bump.baseline"),
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("generation_bump.baseline_dispatch_timeout_s"),
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=case.value("generation_bump.baseline_state_timeout_s"),
        params=case.params(
            "generation_bump.baseline_state",
            {"requests": output("baseline", "requests")},
        ),
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params=case.params(
            "generation_bump.baseline_succeeds",
            {"snapshot": output("baseline_state", "snapshot")},
        ),
    )
    case.step(
        "outage",
        "engine_control",
        params=case.value("generation_bump.outage"),
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=case.value("generation_bump.retirement_timeout_s"),
        params=case.params(
            "generation_bump.retirement", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "transport_retired",
        "recovery_check",
        params=case.params(
            "generation_bump.transport_retired",
            {"snapshot": output("retirement", "snapshot")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.value("generation_bump.restart"),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("generation_bump.alive_restored_timeout_s"),
        params=case.value("generation_bump.alive_restored"),
    )
    case.step(
        "alive_back",
        "recovery_check",
        params=case.params(
            "generation_bump.alive_back",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("generation_bump.reconnect_timeout_s"),
        params=case.value("generation_bump.reconnect"),
    )
    case.step(
        "recovered_generation",
        "recovery_observe",
        timeout_s=case.value("generation_bump.recovered_generation_timeout_s"),
        params=case.params(
            "generation_bump.recovered_generation",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "generation_is_new",
        "recovery_check",
        params=case.params(
            "generation_bump.generation_is_new",
            {
                "snapshot": output("recovered_generation", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "recovered_prefill_batch_ledger_zero",
        "recovery_check",
        params=case.params(
            "generation_bump.recovered_prefill_batch_ledger_zero",
            {"snapshot": output("recovered_generation", "snapshot")},
        ),
    )
    case.step(
        "recovered_prefill_member_ledger_zero",
        "recovery_check",
        params=case.params(
            "generation_bump.recovered_prefill_member_ledger_zero",
            {"snapshot": output("recovered_generation", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("generation_bump.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("generation_bump.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("generation_bump.recovery_state_timeout_s"),
        params=case.params(
            "generation_bump.recovery_state",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "generation_bump.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def status_gap_no_bump(case):
    case.step(
        "setup", "setup", timeout_s=case.value("status_gap_no_bump.setup_timeout_s")
    )
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("status_gap_no_bump.prior_drain_observed_timeout_s"),
        params=case.value("status_gap_no_bump.prior_drain_observed"),
    )
    case.step(
        "target", "recovery_select", params=case.value("status_gap_no_bump.target")
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("status_gap_no_bump.generation_before_timeout_s"),
        params=case.params(
            "status_gap_no_bump.generation_before",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params=case.value("status_gap_no_bump.baseline"),
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("status_gap_no_bump.baseline_dispatch_timeout_s"),
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=case.value("status_gap_no_bump.baseline_state_timeout_s"),
        params=case.params(
            "status_gap_no_bump.baseline_state",
            {"requests": output("baseline", "requests")},
        ),
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params=case.params(
            "status_gap_no_bump.baseline_succeeds",
            {"snapshot": output("baseline_state", "snapshot")},
        ),
    )
    case.step(
        "short_gap",
        "status_control",
        params=case.value("status_gap_no_bump.short_gap"),
    )
    case.step(
        "two_poll_ticks",
        "recovery_pause",
        timeout_s=case.value("status_gap_no_bump.two_poll_ticks_timeout_s"),
        params=case.value("status_gap_no_bump.two_poll_ticks"),
    )
    case.step(
        "resume_status",
        "status_control",
        params=case.value("status_gap_no_bump.resume_status"),
    )
    case.step(
        "hung_rpc_deadline_and_resume",
        "recovery_pause",
        timeout_s=case.value(
            "status_gap_no_bump.hung_rpc_deadline_and_resume_timeout_s"
        ),
        params=case.value("status_gap_no_bump.hung_rpc_deadline_and_resume"),
    )
    case.step(
        "post_gap_generation",
        "recovery_observe",
        timeout_s=case.value("status_gap_no_bump.post_gap_generation_timeout_s"),
        params=case.params(
            "status_gap_no_bump.post_gap_generation",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "no_new_generation",
        "recovery_check",
        params=case.params(
            "status_gap_no_bump.no_new_generation",
            {
                "snapshot": output("post_gap_generation", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "discovered_intact",
        "recovery_check",
        params=case.params(
            "status_gap_no_bump.discovered_intact",
            {"snapshot": output("post_gap_generation", "snapshot")},
        ),
    )
    case.step(
        "alive_intact",
        "recovery_check",
        params=case.params(
            "status_gap_no_bump.alive_intact",
            {"snapshot": output("post_gap_generation", "snapshot")},
        ),
    )
    case.step(
        "post_gap",
        "recovery_prepare",
        params=case.value("status_gap_no_bump.post_gap"),
    )
    case.step(
        "post_gap_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("status_gap_no_bump.post_gap_dispatch_timeout_s"),
        params={"requests": output("post_gap", "requests")},
    )
    case.step(
        "post_gap_state",
        "recovery_observe",
        timeout_s=case.value("status_gap_no_bump.post_gap_state_timeout_s"),
        params=case.params(
            "status_gap_no_bump.post_gap_state",
            {"requests": output("post_gap", "requests")},
        ),
    )
    case.step(
        "post_gap_succeeds",
        "recovery_check",
        params=case.params(
            "status_gap_no_bump.post_gap_succeeds",
            {"snapshot": output("post_gap_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def down_phases(case):
    case.step("setup", "setup", timeout_s=case.value("down_phases.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("down_phases.prior_drain_observed_timeout_s"),
        params=case.value("down_phases.prior_drain_observed"),
    )
    case.step(
        "baseline",
        "recovery_prepare",
        params=case.value("down_phases.baseline"),
    )
    case.step(
        "baseline_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("down_phases.baseline_dispatch_timeout_s"),
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_state",
        "recovery_observe",
        timeout_s=case.value("down_phases.baseline_state_timeout_s"),
        params=case.params(
            "down_phases.baseline_state", {"requests": output("baseline", "requests")}
        ),
    )
    case.step(
        "baseline_succeeds",
        "recovery_check",
        params=case.params(
            "down_phases.baseline_succeeds",
            {"snapshot": output("baseline_state", "snapshot")},
        ),
    )
    case.step(
        "baseline_health",
        "recovery_observe",
        timeout_s=case.value("down_phases.baseline_health_timeout_s"),
        params=case.value("down_phases.baseline_health"),
    )
    case.step(
        "baseline_health_http_200",
        "recovery_check",
        params=case.params(
            "down_phases.baseline_health_http_200",
            {"snapshot": output("baseline_health", "snapshot")},
        ),
    )
    case.step(
        "outage",
        "engine_control",
        params=case.value("down_phases.outage"),
    )
    case.step(
        "eviction",
        "recovery_observe",
        timeout_s=case.value("down_phases.eviction_timeout_s"),
        params=case.value("down_phases.eviction"),
    )
    case.step(
        "survivor_only_alive",
        "recovery_check",
        params=case.params(
            "down_phases.survivor_only_alive",
            {"snapshot": output("eviction", "snapshot")},
        ),
    )
    case.step(
        "takeover",
        "recovery_prepare",
        params=case.value("down_phases.takeover"),
    )
    case.step(
        "takeover_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("down_phases.takeover_dispatch_timeout_s"),
        params={"requests": output("takeover", "requests")},
    )
    case.step(
        "takeover_state",
        "recovery_observe",
        timeout_s=case.value("down_phases.takeover_state_timeout_s"),
        params=case.params(
            "down_phases.takeover_state", {"requests": output("takeover", "requests")}
        ),
    )
    case.step(
        "downtime_health",
        "recovery_observe",
        timeout_s=case.value("down_phases.downtime_health_timeout_s"),
        params=case.value("down_phases.downtime_health"),
    )
    case.step(
        "downtime_health_http_200",
        "recovery_check",
        params=case.params(
            "down_phases.downtime_health_http_200",
            {"snapshot": output("downtime_health", "snapshot")},
        ),
    )
    case.step(
        "takeover_at_least_ninety_percent",
        "recovery_check",
        params=case.params(
            "down_phases.takeover_at_least_ninety_percent",
            {"snapshot": output("takeover_state", "snapshot")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.value("down_phases.restart"),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("down_phases.alive_restored_timeout_s"),
        params=case.value("down_phases.alive_restored"),
    )
    case.step(
        "alive_back",
        "recovery_check",
        params=case.params(
            "down_phases.alive_back", {"snapshot": output("alive_restored", "snapshot")}
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("down_phases.reconnect_timeout_s"),
        params=case.value("down_phases.reconnect"),
    )
    case.step(
        "recovery_batch",
        "recovery_prepare",
        params=case.value("down_phases.recovery_batch"),
    )
    case.step(
        "recovery_batch_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("down_phases.recovery_batch_dispatch_timeout_s"),
        params={"requests": output("recovery_batch", "requests")},
    )
    case.step(
        "recovery_batch_state",
        "recovery_observe",
        timeout_s=case.value("down_phases.recovery_batch_state_timeout_s"),
        params=case.params(
            "down_phases.recovery_batch_state",
            {"requests": output("recovery_batch", "requests")},
        ),
    )
    case.step(
        "recovery_at_least_ninety_five_percent",
        "recovery_check",
        params=case.params(
            "down_phases.recovery_at_least_ninety_five_percent",
            {"snapshot": output("recovery_batch_state", "snapshot")},
        ),
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
        timeout_s=case.value("down_phases.recovered_health_timeout_s"),
        params=case.value("down_phases.recovered_health"),
    )
    case.step(
        "recovered_health_http_200",
        "recovery_check",
        params=case.params(
            "down_phases.recovered_health_http_200",
            {"snapshot": output("recovered_health", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "recovery_observe",
        timeout_s=case.value("down_phases.closing_drain_timeout_s"),
        params=case.value("down_phases.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "recovery_check",
        params=case.params(
            "down_phases.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "recovery_check",
        params=case.params(
            "down_phases.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "recovery_check",
        params=case.params(
            "down_phases.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def flap(case):
    case.step("setup", "setup", timeout_s=case.value("flap.setup_timeout_s"))
    case.step("background_flow", "elastic_cold_flow")
    case.step(
        "flow_ramp",
        "recovery_pause",
        timeout_s=case.value("flap.flow_ramp_timeout_s"),
        params=case.value("flap.flow_ramp"),
    )
    case.step(
        "stop_1",
        "engine_control",
        params=case.value("flap.stop_1"),
    )
    case.step(
        "down_window_1",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_1_timeout_s"),
        params=case.value("flap.down_window_1"),
    )
    case.step(
        "alive_during_1",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_1_timeout_s"),
        params=case.value("flap.alive_during_1"),
    )
    case.step(
        "master_cycle_1",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_1_timeout_s"),
        params=case.value("flap.master_cycle_1"),
    )
    case.step(
        "master_cycle_1_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_1_http_200",
            {"snapshot": output("master_cycle_1", "snapshot")},
        ),
    )
    case.step(
        "start_1",
        "engine_control",
        params=case.value("flap.start_1"),
    )
    case.step(
        "up_gap_1",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_1_timeout_s"),
        params=case.value("flap.up_gap_1"),
    )
    case.step(
        "stop_2",
        "engine_control",
        params=case.value("flap.stop_2"),
    )
    case.step(
        "down_window_2",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_2_timeout_s"),
        params=case.value("flap.down_window_2"),
    )
    case.step(
        "alive_during_2",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_2_timeout_s"),
        params=case.value("flap.alive_during_2"),
    )
    case.step(
        "master_cycle_2",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_2_timeout_s"),
        params=case.value("flap.master_cycle_2"),
    )
    case.step(
        "master_cycle_2_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_2_http_200",
            {"snapshot": output("master_cycle_2", "snapshot")},
        ),
    )
    case.step(
        "start_2",
        "engine_control",
        params=case.value("flap.start_2"),
    )
    case.step(
        "up_gap_2",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_2_timeout_s"),
        params=case.value("flap.up_gap_2"),
    )
    case.step(
        "stop_3",
        "engine_control",
        params=case.value("flap.stop_3"),
    )
    case.step(
        "down_window_3",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_3_timeout_s"),
        params=case.value("flap.down_window_3"),
    )
    case.step(
        "alive_during_3",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_3_timeout_s"),
        params=case.value("flap.alive_during_3"),
    )
    case.step(
        "master_cycle_3",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_3_timeout_s"),
        params=case.value("flap.master_cycle_3"),
    )
    case.step(
        "master_cycle_3_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_3_http_200",
            {"snapshot": output("master_cycle_3", "snapshot")},
        ),
    )
    case.step(
        "start_3",
        "engine_control",
        params=case.value("flap.start_3"),
    )
    case.step(
        "up_gap_3",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_3_timeout_s"),
        params=case.value("flap.up_gap_3"),
    )
    case.step(
        "stop_4",
        "engine_control",
        params=case.value("flap.stop_4"),
    )
    case.step(
        "down_window_4",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_4_timeout_s"),
        params=case.value("flap.down_window_4"),
    )
    case.step(
        "alive_during_4",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_4_timeout_s"),
        params=case.value("flap.alive_during_4"),
    )
    case.step(
        "master_cycle_4",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_4_timeout_s"),
        params=case.value("flap.master_cycle_4"),
    )
    case.step(
        "master_cycle_4_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_4_http_200",
            {"snapshot": output("master_cycle_4", "snapshot")},
        ),
    )
    case.step(
        "start_4",
        "engine_control",
        params=case.value("flap.start_4"),
    )
    case.step(
        "up_gap_4",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_4_timeout_s"),
        params=case.value("flap.up_gap_4"),
    )
    case.step(
        "stop_5",
        "engine_control",
        params=case.value("flap.stop_5"),
    )
    case.step(
        "down_window_5",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_5_timeout_s"),
        params=case.value("flap.down_window_5"),
    )
    case.step(
        "alive_during_5",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_5_timeout_s"),
        params=case.value("flap.alive_during_5"),
    )
    case.step(
        "master_cycle_5",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_5_timeout_s"),
        params=case.value("flap.master_cycle_5"),
    )
    case.step(
        "master_cycle_5_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_5_http_200",
            {"snapshot": output("master_cycle_5", "snapshot")},
        ),
    )
    case.step(
        "start_5",
        "engine_control",
        params=case.value("flap.start_5"),
    )
    case.step(
        "up_gap_5",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_5_timeout_s"),
        params=case.value("flap.up_gap_5"),
    )
    case.step(
        "stop_6",
        "engine_control",
        params=case.value("flap.stop_6"),
    )
    case.step(
        "down_window_6",
        "recovery_pause",
        timeout_s=case.value("flap.down_window_6_timeout_s"),
        params=case.value("flap.down_window_6"),
    )
    case.step(
        "alive_during_6",
        "recovery_observe",
        timeout_s=case.value("flap.alive_during_6_timeout_s"),
        params=case.value("flap.alive_during_6"),
    )
    case.step(
        "master_cycle_6",
        "recovery_observe",
        timeout_s=case.value("flap.master_cycle_6_timeout_s"),
        params=case.value("flap.master_cycle_6"),
    )
    case.step(
        "master_cycle_6_http_200",
        "recovery_check",
        params=case.params(
            "flap.master_cycle_6_http_200",
            {"snapshot": output("master_cycle_6", "snapshot")},
        ),
    )
    case.step(
        "start_6",
        "engine_control",
        params=case.value("flap.start_6"),
    )
    case.step(
        "up_gap_6",
        "recovery_pause",
        timeout_s=case.value("flap.up_gap_6_timeout_s"),
        params=case.value("flap.up_gap_6"),
    )
    case.step(
        "stop_flow",
        "recovery_flow_stop",
        timeout_s=case.value("flap.stop_flow_timeout_s"),
        params={"flow": output("background_flow", "flow")},
    )
    case.step(
        "flow_availability",
        "recovery_flow_assert",
        params=case.params(
            "flap.flow_availability", {"result": output("stop_flow", "result")}
        ),
    )
    case.step(
        "topology",
        "recovery_observe",
        timeout_s=case.value("flap.topology_timeout_s"),
        params=case.value("flap.topology"),
    )
    case.step(
        "topology_discovered",
        "recovery_check",
        params=case.params(
            "flap.topology_discovered", {"snapshot": output("topology", "snapshot")}
        ),
    )
    case.step(
        "topology_alive",
        "recovery_check",
        params=case.params(
            "flap.topology_alive", {"snapshot": output("topology", "snapshot")}
        ),
    )
    case.step(
        "post_flap",
        "recovery_prepare",
        params=case.value("flap.post_flap"),
    )
    case.step(
        "post_flap_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("flap.post_flap_dispatch_timeout_s"),
        params={"requests": output("post_flap", "requests")},
    )
    case.step(
        "post_flap_state",
        "recovery_observe",
        timeout_s=case.value("flap.post_flap_state_timeout_s"),
        params=case.params(
            "flap.post_flap_state", {"requests": output("post_flap", "requests")}
        ),
    )
    case.step(
        "post_flap_recovers",
        "recovery_check",
        params=case.params(
            "flap.post_flap_recovers",
            {"snapshot": output("post_flap_state", "snapshot")},
        ),
    )
    case.step(
        "closing_drain",
        "recovery_observe",
        timeout_s=case.value("flap.closing_drain_timeout_s"),
        params=case.value("flap.closing_drain"),
    )
    case.step(
        "closing_drain_scheduler",
        "recovery_check",
        params=case.params(
            "flap.closing_drain_scheduler",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_prefill_batches",
        "recovery_check",
        params=case.params(
            "flap.closing_drain_prefill_batches",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step(
        "closing_drain_decode_load",
        "recovery_check",
        params=case.params(
            "flap.closing_drain_decode_load",
            {"snapshot": output("closing_drain", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def kv_resync(case):
    case.step("setup", "setup", timeout_s=case.value("kv_resync.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("kv_resync.prior_drain_observed_timeout_s"),
        params=case.value("kv_resync.prior_drain_observed"),
    )
    case.step(
        "seed",
        "recovery_prepare",
        params=case.value("kv_resync.seed"),
    )
    case.step(
        "seed_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("kv_resync.seed_dispatch_timeout_s"),
        params={"requests": output("seed", "requests")},
    )
    case.step(
        "seed_state",
        "recovery_observe",
        timeout_s=case.value("kv_resync.seed_state_timeout_s"),
        params=case.params(
            "kv_resync.seed_state", {"requests": output("seed", "requests")}
        ),
    )
    case.step(
        "seed_succeeds",
        "recovery_check",
        params=case.params(
            "kv_resync.seed_succeeds", {"snapshot": output("seed_state", "snapshot")}
        ),
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
        timeout_s=case.value("kv_resync.seed_ownership_timeout_s"),
        params=case.params(
            "kv_resync.seed_ownership", {"selection": output("holder", "selection")}
        ),
    )
    case.step(
        "holder_has_whole_family",
        "recovery_check",
        params=case.params(
            "kv_resync.holder_has_whole_family",
            {"snapshot": output("seed_ownership", "snapshot")},
        ),
    )
    case.step(
        "cache_sync_before_outage",
        "recovery_pause",
        timeout_s=case.value("kv_resync.cache_sync_before_outage_timeout_s"),
        params=case.value("kv_resync.cache_sync_before_outage"),
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("holder", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("kv_resync.generation_before_timeout_s"),
        params=case.params(
            "kv_resync.generation_before", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "stop_holder",
        "engine_control",
        params=case.params(
            "kv_resync.stop_holder", {"targets": [output("holder", "engine")]}
        ),
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=case.value("kv_resync.retirement_timeout_s"),
        params=case.params(
            "kv_resync.retirement", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "holder_retired",
        "recovery_check",
        params=case.params(
            "kv_resync.holder_retired", {"snapshot": output("retirement", "snapshot")}
        ),
    )
    case.step(
        "restart_holder",
        "engine_control",
        params=case.params(
            "kv_resync.restart_holder", {"targets": [output("holder", "engine")]}
        ),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("kv_resync.alive_restored_timeout_s"),
        params=case.value("kv_resync.alive_restored"),
    )
    case.step(
        "holder_alive_back",
        "recovery_check",
        params=case.params(
            "kv_resync.holder_alive_back",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("kv_resync.reconnect_timeout_s"),
        params=case.value("kv_resync.reconnect"),
    )
    case.step(
        "new_generation",
        "recovery_observe",
        timeout_s=case.value("kv_resync.new_generation_timeout_s"),
        params=case.params(
            "kv_resync.new_generation", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "holder_generation_bumped",
        "recovery_check",
        params=case.params(
            "kv_resync.holder_generation_bumped",
            {
                "snapshot": output("new_generation", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "intact_wave",
        "recovery_prepare",
        params=case.value("kv_resync.intact_wave"),
    )
    case.step(
        "intact_wave_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("kv_resync.intact_wave_dispatch_timeout_s"),
        params={"requests": output("intact_wave", "requests")},
    )
    case.step(
        "intact_wave_state",
        "recovery_observe",
        timeout_s=case.value("kv_resync.intact_wave_state_timeout_s"),
        params=case.params(
            "kv_resync.intact_wave_state",
            {"requests": output("intact_wave", "requests")},
        ),
    )
    case.step(
        "intact_routes",
        "recovery_observe",
        timeout_s=case.value("kv_resync.intact_routes_timeout_s"),
        params=case.params(
            "kv_resync.intact_routes",
            {
                "selection": output("holder", "selection"),
                "requests": output("intact_wave", "requests"),
            },
        ),
    )
    case.step(
        "memory_intact_holder_survives",
        "recovery_check",
        params=case.params(
            "kv_resync.memory_intact_holder_survives",
            {"snapshot": output("intact_routes", "snapshot")},
        ),
    )
    case.step(
        "wipe_family",
        "recovery_cache_control",
        params=case.params(
            "kv_resync.wipe_family", {"selection": output("holder", "selection")}
        ),
    )
    case.step(
        "family_gone",
        "recovery_observe",
        timeout_s=case.value("kv_resync.family_gone_timeout_s"),
        params=case.params(
            "kv_resync.family_gone", {"selection": output("holder", "selection")}
        ),
    )
    case.step(
        "wiped_family_absent",
        "recovery_check",
        params=case.params(
            "kv_resync.wiped_family_absent",
            {"snapshot": output("family_gone", "snapshot")},
        ),
    )
    case.step(
        "cache_sync_after_wipe",
        "recovery_pause",
        timeout_s=case.value("kv_resync.cache_sync_after_wipe_timeout_s"),
        params=case.value("kv_resync.cache_sync_after_wipe"),
    )
    case.step(
        "wiped_wave",
        "recovery_prepare",
        params=case.value("kv_resync.wiped_wave"),
    )
    case.step(
        "wiped_wave_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("kv_resync.wiped_wave_dispatch_timeout_s"),
        params={"requests": output("wiped_wave", "requests")},
    )
    case.step(
        "wiped_wave_state",
        "recovery_observe",
        timeout_s=case.value("kv_resync.wiped_wave_state_timeout_s"),
        params=case.params(
            "kv_resync.wiped_wave_state", {"requests": output("wiped_wave", "requests")}
        ),
    )
    case.step(
        "wiped_routes",
        "recovery_observe",
        timeout_s=case.value("kv_resync.wiped_routes_timeout_s"),
        params=case.params(
            "kv_resync.wiped_routes",
            {
                "selection": output("holder", "selection"),
                "requests": output("wiped_wave", "requests"),
            },
        ),
    )
    case.step(
        "memory_lost_old_holder_spreads",
        "recovery_check",
        params=case.params(
            "kv_resync.memory_lost_old_holder_spreads",
            {"snapshot": output("wiped_routes", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def kv_usage_reset(case):
    case.step("setup", "setup", timeout_s=case.value("kv_usage_reset.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.prior_drain_observed_timeout_s"),
        params=case.value("kv_usage_reset.prior_drain_observed"),
    )
    case.step("target", "recovery_select", params=case.value("kv_usage_reset.target"))
    case.step(
        "evict_current_cache",
        "recovery_cache_control",
        params=case.params(
            "kv_usage_reset.evict_current_cache",
            {"selection": output("target", "selection")},
        ),
    )
    case.step(
        "empty_cache_sync",
        "recovery_pause",
        timeout_s=case.value("kv_usage_reset.empty_cache_sync_timeout_s"),
        params=case.value("kv_usage_reset.empty_cache_sync"),
    )
    case.step(
        "empty_usage",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.empty_usage_timeout_s"),
        params=case.params(
            "kv_usage_reset.empty_usage", {"selection": output("target", "selection")}
        ),
    )
    case.step(
        "baseline_usage_zero",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.baseline_usage_zero",
            {"snapshot": output("empty_usage", "snapshot")},
        ),
    )
    case.step(
        "set_pressure",
        "recovery_cache_control",
        params=case.params(
            "kv_usage_reset.set_pressure", {"selection": output("target", "selection")}
        ),
    )
    case.step(
        "pressure_reported",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.pressure_reported_timeout_s"),
        params=case.params(
            "kv_usage_reset.pressure_reported",
            {"selection": output("target", "selection")},
        ),
    )
    case.step(
        "pressure_construction_observed",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.pressure_construction_observed",
            {"snapshot": output("pressure_reported", "snapshot")},
        ),
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.generation_before_timeout_s"),
        params=case.params(
            "kv_usage_reset.generation_before", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "outage",
        "engine_control",
        params=case.value("kv_usage_reset.outage"),
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.retirement_timeout_s"),
        params=case.params(
            "kv_usage_reset.retirement", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "pressure_generation_retired",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.pressure_generation_retired",
            {"snapshot": output("retirement", "snapshot")},
        ),
    )
    case.step(
        "restart",
        "engine_control",
        params=case.value("kv_usage_reset.restart"),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.alive_restored_timeout_s"),
        params=case.value("kv_usage_reset.alive_restored"),
    )
    case.step(
        "alive_back",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.alive_back",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("kv_usage_reset.reconnect_timeout_s"),
        params=case.value("kv_usage_reset.reconnect"),
    )
    case.step(
        "recovered_generation",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.recovered_generation_timeout_s"),
        params=case.params(
            "kv_usage_reset.recovered_generation",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "new_capacity_generation",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.new_capacity_generation",
            {
                "snapshot": output("recovered_generation", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "lack_mem_baseline",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.lack_mem_baseline_timeout_s"),
        params=case.value("kv_usage_reset.lack_mem_baseline"),
    )
    case.step(
        "first_wave",
        "recovery_prepare",
        params=case.value("kv_usage_reset.first_wave"),
    )
    case.step(
        "first_wave_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("kv_usage_reset.first_wave_dispatch_timeout_s"),
        params={"requests": output("first_wave", "requests")},
    )
    case.step(
        "first_wave_state",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.first_wave_state_timeout_s"),
        params=case.params(
            "kv_usage_reset.first_wave_state",
            {"requests": output("first_wave", "requests")},
        ),
    )
    case.step(
        "first_wave_succeeds",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.first_wave_succeeds",
            {"snapshot": output("first_wave_state", "snapshot")},
        ),
    )
    case.step(
        "lack_mem_after",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.lack_mem_after_timeout_s"),
        params=case.value("kv_usage_reset.lack_mem_after"),
    )
    case.step(
        "first_wave_no_lack_mem",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.first_wave_no_lack_mem",
            {
                "snapshot": output("lack_mem_after", "snapshot"),
                "baseline": output("lack_mem_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "target_receives_traffic",
        "recovery_pump",
        timeout_s=case.value("kv_usage_reset.target_receives_traffic_timeout_s"),
        params=case.params(
            "kv_usage_reset.target_receives_traffic",
            {"selection": output("target", "selection")},
        ),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("kv_usage_reset.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("kv_usage_reset.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("kv_usage_reset.recovery_state_timeout_s"),
        params=case.params(
            "kv_usage_reset.recovery_state",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "kv_usage_reset.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def crash_after(case):
    case.step("setup", "setup", timeout_s=case.value("crash_after.setup_timeout_s"))
    case.step("prefills", "recovery_select", params=case.value("crash_after.prefills"))
    case.step(
        "arm_first_enqueue",
        "recovery_crash_arm",
        params=case.params(
            "crash_after.arm_first_enqueue",
            {"selection": output("prefills", "selection")},
        ),
    )
    case.step(
        "trigger",
        "recovery_prepare",
        params=case.value("crash_after.trigger"),
    )
    case.step(
        "trigger_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("crash_after.trigger_dispatch_timeout_s"),
        params={"requests": output("trigger", "requests")},
    )
    case.step(
        "trigger_state",
        "recovery_observe",
        timeout_s=case.value("crash_after.trigger_state_timeout_s"),
        params=case.params(
            "crash_after.trigger_state", {"requests": output("trigger", "requests")}
        ),
    )
    case.step(
        "crash_state",
        "recovery_observe",
        timeout_s=case.value("crash_after.crash_state_timeout_s"),
        params=case.params(
            "crash_after.crash_state", {"selection": output("prefills", "selection")}
        ),
    )
    case.step(
        "crashed",
        "recovery_partition",
        params=case.params(
            "crash_after.crashed",
            {
                "selection": output("prefills", "selection"),
                "snapshot": output("crash_state", "snapshot"),
            },
        ),
    )
    case.step(
        "survivors",
        "recovery_partition",
        params=case.params(
            "crash_after.survivors",
            {
                "selection": output("prefills", "selection"),
                "snapshot": output("crash_state", "snapshot"),
            },
        ),
    )
    case.step(
        "exactly_one_crashed",
        "check",
        params=case.params(
            "crash_after.exactly_one_crashed", {"actual": output("crashed", "count")}
        ),
    )
    case.step(
        "disarm_survivors",
        "recovery_crash_arm",
        params=case.params(
            "crash_after.disarm_survivors",
            {"selection": output("survivors", "selection")},
        ),
    )
    case.step(
        "alive_dropped",
        "recovery_observe",
        timeout_s=case.value("crash_after.alive_dropped_timeout_s"),
        params=case.value("crash_after.alive_dropped"),
    )
    case.step(
        "master_observed_loss",
        "recovery_check",
        params=case.params(
            "crash_after.master_observed_loss",
            {"snapshot": output("alive_dropped", "snapshot")},
        ),
    )
    case.step(
        "routable_set_settle",
        "recovery_pause",
        timeout_s=case.value("crash_after.routable_set_settle_timeout_s"),
        params=case.value("crash_after.routable_set_settle"),
    )
    case.step(
        "takeover",
        "recovery_prepare",
        params=case.value("crash_after.takeover"),
    )
    case.step(
        "takeover_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("crash_after.takeover_dispatch_timeout_s"),
        params={"requests": output("takeover", "requests")},
    )
    case.step(
        "takeover_state",
        "recovery_observe",
        timeout_s=case.value("crash_after.takeover_state_timeout_s"),
        params=case.params(
            "crash_after.takeover_state", {"requests": output("takeover", "requests")}
        ),
    )
    case.step(
        "survivor_serves_sixty_percent",
        "recovery_check",
        params=case.params(
            "crash_after.survivor_serves_sixty_percent",
            {"snapshot": output("takeover_state", "snapshot")},
        ),
    )
    case.step(
        "restart_crashed",
        "recovery_engine_control",
        params=case.params(
            "crash_after.restart_crashed", {"selection": output("crashed", "selection")}
        ),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("crash_after.alive_restored_timeout_s"),
        params=case.value("crash_after.alive_restored"),
    )
    case.step(
        "crashed_engine_rediscovers",
        "recovery_check",
        params=case.params(
            "crash_after.crashed_engine_rediscovers",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("crash_after.reconnect_timeout_s"),
        params=case.value("crash_after.reconnect"),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("crash_after.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("crash_after.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("crash_after.recovery_state_timeout_s"),
        params=case.params(
            "crash_after.recovery_state", {"requests": output("recovery", "requests")}
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "crash_after.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step(
        "residue_window",
        "recovery_residue",
        timeout_s=case.value("crash_after.residue_window_timeout_s"),
        params=case.params(
            "crash_after.residue_window", {"requests": output("takeover", "requests")}
        ),
    )
    case.step(
        "prefill_engines_clean",
        "recovery_observe",
        timeout_s=case.value("crash_after.prefill_engines_clean_timeout_s"),
        params=case.params(
            "crash_after.prefill_engines_clean",
            {"selection": output("prefills", "selection")},
        ),
    )
    case.step(
        "prefill_engines_clean_inflight",
        "recovery_check",
        params=case.params(
            "crash_after.prefill_engines_clean_inflight",
            {"snapshot": output("prefill_engines_clean", "snapshot")},
        ),
    )
    case.step(
        "prefill_engines_clean_leaks",
        "recovery_check",
        params=case.params(
            "crash_after.prefill_engines_clean_leaks",
            {"snapshot": output("prefill_engines_clean", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def no_resurrect(case):
    case.step("setup", "setup", timeout_s=case.value("no_resurrect.setup_timeout_s"))
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.prior_drain_observed_timeout_s"),
        params=case.value("no_resurrect.prior_drain_observed"),
    )
    case.step("prefills", "recovery_select", params=case.value("no_resurrect.prefills"))
    case.step(
        "slow_prefill",
        "recovery_engine_control",
        params=case.params(
            "no_resurrect.slow_prefill", {"selection": output("prefills", "selection")}
        ),
    )
    case.step(
        "perf_sync",
        "recovery_pause",
        timeout_s=case.value("no_resurrect.perf_sync_timeout_s"),
        params=case.value("no_resurrect.perf_sync"),
    )
    case.step(
        "fired",
        "recovery_prepare",
        params=case.value("no_resurrect.fired"),
    )
    case.step(
        "fired_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("no_resurrect.fired_dispatch_timeout_s"),
        params={"requests": output("fired", "requests")},
    )
    case.step(
        "fired_state",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.fired_state_timeout_s"),
        params=case.params(
            "no_resurrect.fired_state", {"requests": output("fired", "requests")}
        ),
    )
    case.step(
        "routed_payload_minimum",
        "recovery_check",
        params=case.params(
            "no_resurrect.routed_payload_minimum",
            {"snapshot": output("fired_state", "snapshot")},
        ),
    )
    case.step(
        "batch_dispatch",
        "recovery_pause",
        timeout_s=case.value("no_resurrect.batch_dispatch_timeout_s"),
        params=case.value("no_resurrect.batch_dispatch"),
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
        params=case.params(
            "no_resurrect.arm_next_enqueue",
            {"selection": output("targets", "selection")},
        ),
    )
    case.step(
        "trigger_all_targets",
        "recovery_crash_trigger",
        timeout_s=case.value("no_resurrect.trigger_all_targets_timeout_s"),
        params=case.params(
            "no_resurrect.trigger_all_targets",
            {"selection": output("targets", "selection")},
        ),
    )
    case.step(
        "all_targets_crashed",
        "recovery_all_targets_check",
        params=case.params(
            "no_resurrect.all_targets_crashed",
            {"snapshot": output("trigger_all_targets", "snapshot")},
        ),
    )
    case.step(
        "retire_all",
        "recovery_retire_all",
        timeout_s=case.value("no_resurrect.retire_all_timeout_s"),
        params=case.params(
            "no_resurrect.retire_all",
            {
                "selection": output("targets", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "restart_targets",
        "recovery_engine_control",
        params=case.params(
            "no_resurrect.restart_targets",
            {"selection": output("targets", "selection")},
        ),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.alive_restored_timeout_s"),
        params=case.value("no_resurrect.alive_restored"),
    )
    case.step(
        "all_prefills_alive",
        "recovery_check",
        params=case.params(
            "no_resurrect.all_prefills_alive",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("no_resurrect.reconnect_timeout_s"),
        params=case.value("no_resurrect.reconnect"),
    )
    case.step(
        "pretraffic_reset",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.pretraffic_reset_timeout_s"),
        params=case.params(
            "no_resurrect.pretraffic_reset",
            {"selection": output("targets", "selection")},
        ),
    )
    case.step(
        "new_prefill_batches_zero",
        "recovery_check",
        params=case.params(
            "no_resurrect.new_prefill_batches_zero",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "new_prefill_members_zero",
        "recovery_check",
        params=case.params(
            "no_resurrect.new_prefill_members_zero",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "wiped_running",
        "recovery_check",
        params=case.params(
            "no_resurrect.wiped_running",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "wiped_inflight",
        "recovery_check",
        params=case.params(
            "no_resurrect.wiped_inflight",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "wiped_cache_keys",
        "recovery_check",
        params=case.params(
            "no_resurrect.wiped_cache_keys",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "wiped_held_blocks",
        "recovery_check",
        params=case.params(
            "no_resurrect.wiped_held_blocks",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "wiped_accepted",
        "recovery_check",
        params=case.params(
            "no_resurrect.wiped_accepted",
            {"snapshot": output("pretraffic_reset", "snapshot")},
        ),
    )
    case.step(
        "consume_old_payload",
        "recovery_consume",
        timeout_s=case.value("no_resurrect.consume_old_payload_timeout_s"),
        params=case.params(
            "no_resurrect.consume_old_payload",
            {"requests": output("fired", "requests")},
        ),
    )
    case.step(
        "old_requests_never_complete",
        "recovery_check",
        params=case.params(
            "no_resurrect.old_requests_never_complete",
            {"snapshot": output("consume_old_payload", "snapshot")},
        ),
    )
    case.step(
        "engine_drain",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.engine_drain_timeout_s"),
        params=case.params(
            "no_resurrect.engine_drain", {"selection": output("targets", "selection")}
        ),
    )
    case.step(
        "engine_drain_inflight",
        "recovery_check",
        params=case.params(
            "no_resurrect.engine_drain_inflight",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "engine_drain_leaks",
        "recovery_check",
        params=case.params(
            "no_resurrect.engine_drain_leaks",
            {"snapshot": output("engine_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.master_drain_timeout_s"),
        params=case.value("no_resurrect.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "recovery_check",
        params=case.params(
            "no_resurrect.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "recovery_check",
        params=case.params(
            "no_resurrect.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "recovery_check",
        params=case.params(
            "no_resurrect.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("no_resurrect.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("no_resurrect.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("no_resurrect.recovery_state_timeout_s"),
        params=case.params(
            "no_resurrect.recovery_state", {"requests": output("recovery", "requests")}
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "no_resurrect.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def status_gap_long_retire(case):
    case.step(
        "setup", "setup", timeout_s=case.value("status_gap_long_retire.setup_timeout_s")
    )
    case.step(
        "prior_drain_observed",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.prior_drain_observed_timeout_s"),
        params=case.value("status_gap_long_retire.prior_drain_observed"),
    )
    case.step(
        "target", "recovery_select", params=case.value("status_gap_long_retire.target")
    )
    case.step(
        "log_mark",
        "recovery_log_mark",
        params={"selection": output("target", "selection")},
    )
    case.step(
        "generation_before",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.generation_before_timeout_s"),
        params=case.params(
            "status_gap_long_retire.generation_before",
            {
                "selection": output("target", "selection"),
                "log": output("log_mark", "snapshot"),
            },
        ),
    )
    case.step(
        "prefills",
        "recovery_select",
        params=case.value("status_gap_long_retire.prefills"),
    )
    case.step(
        "slow_prefill",
        "recovery_engine_control",
        params=case.params(
            "status_gap_long_retire.slow_prefill",
            {"selection": output("prefills", "selection")},
        ),
    )
    case.step(
        "perf_sync",
        "recovery_pause",
        timeout_s=case.value("status_gap_long_retire.perf_sync_timeout_s"),
        params=case.value("status_gap_long_retire.perf_sync"),
    )
    case.step(
        "fired",
        "recovery_prepare",
        params=case.value("status_gap_long_retire.fired"),
    )
    case.step(
        "fired_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("status_gap_long_retire.fired_dispatch_timeout_s"),
        params={"requests": output("fired", "requests")},
    )
    case.step(
        "fired_state",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.fired_state_timeout_s"),
        params=case.params(
            "status_gap_long_retire.fired_state",
            {"requests": output("fired", "requests")},
        ),
    )
    case.step(
        "batch_dispatch",
        "recovery_pause",
        timeout_s=case.value("status_gap_long_retire.batch_dispatch_timeout_s"),
        params=case.value("status_gap_long_retire.batch_dispatch"),
    )
    case.step(
        "long_gap",
        "status_control",
        params=case.value("status_gap_long_retire.long_gap"),
    )
    case.step(
        "retirement",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.retirement_timeout_s"),
        params=case.params(
            "status_gap_long_retire.retirement", {"log": output("log_mark", "snapshot")}
        ),
    )
    case.step(
        "long_gap_retires",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.long_gap_retires",
            {"snapshot": output("retirement", "snapshot")},
        ),
    )
    case.step(
        "post_retire_hold",
        "recovery_pause",
        timeout_s=case.value("status_gap_long_retire.post_retire_hold_timeout_s"),
        params=case.value("status_gap_long_retire.post_retire_hold"),
    )
    case.step(
        "resume_status",
        "status_control",
        params=case.value("status_gap_long_retire.resume_status"),
    )
    case.step(
        "alive_restored",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.alive_restored_timeout_s"),
        params=case.value("status_gap_long_retire.alive_restored"),
    )
    case.step(
        "prefill_alive_back",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.prefill_alive_back",
            {"snapshot": output("alive_restored", "snapshot")},
        ),
    )
    case.step(
        "reconnect",
        "recovery_pause",
        timeout_s=case.value("status_gap_long_retire.reconnect_timeout_s"),
        params=case.value("status_gap_long_retire.reconnect"),
    )
    case.step(
        "generation_after",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.generation_after_timeout_s"),
        params=case.params(
            "status_gap_long_retire.generation_after",
            {"log": output("log_mark", "snapshot")},
        ),
    )
    case.step(
        "long_gap_creates_generation",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.long_gap_creates_generation",
            {
                "snapshot": output("generation_after", "snapshot"),
                "baseline": output("generation_before", "snapshot"),
            },
        ),
    )
    case.step(
        "consume_fence_payload_observed",
        "recovery_consume",
        timeout_s=case.value(
            "status_gap_long_retire.consume_fence_payload_observed_timeout_s"
        ),
        params=case.params(
            "status_gap_long_retire.consume_fence_payload_observed",
            {"requests": output("fired", "requests")},
        ),
    )
    case.step(
        "master_drain",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.master_drain_timeout_s"),
        params=case.value("status_gap_long_retire.master_drain"),
    )
    case.step(
        "master_drain_scheduler",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.master_drain_scheduler",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_prefill_batches",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.master_drain_prefill_batches",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "master_drain_decode_load",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.master_drain_decode_load",
            {"snapshot": output("master_drain", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "recovery_prepare",
        params=case.value("status_gap_long_retire.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "recovery_dispatch",
        timeout_s=case.value("status_gap_long_retire.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_state",
        "recovery_observe",
        timeout_s=case.value("status_gap_long_retire.recovery_state_timeout_s"),
        params=case.params(
            "status_gap_long_retire.recovery_state",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "recovery_succeeds",
        "recovery_check",
        params=case.params(
            "status_gap_long_retire.recovery_succeeds",
            {"snapshot": output("recovery_state", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")
