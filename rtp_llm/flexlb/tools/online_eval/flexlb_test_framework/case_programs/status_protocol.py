"""Explicit status protocol boundary programs; old whole-case expected_fail is narrowed to reviewed checks only."""

from ..case_config import output


def inflight_ttl_cleanup(case):
    case.step(
        "setup", "setup", timeout_s=case.value("inflight_ttl_cleanup.setup_timeout_s")
    )
    case.step(
        "slow_prefill",
        "status_perf",
        params=case.value("inflight_ttl_cleanup.slow_prefill"),
    )
    case.step(
        "silent",
        "status_prepare",
        params=case.value("inflight_ttl_cleanup.silent"),
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=case.value("inflight_ttl_cleanup.metrics_ready_timeout_s"),
        params=case.value("inflight_ttl_cleanup.metrics_ready"),
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.ttl_before_timeout_s"),
        params=case.value("inflight_ttl_cleanup.ttl_before"),
    )
    case.step(
        "p_suppress",
        "status_control",
        params=case.params(
            "inflight_ttl_cleanup.p_suppress",
            {"requests": output("silent", "requests")},
        ),
    )
    case.step(
        "d_suppress",
        "status_control",
        params=case.params(
            "inflight_ttl_cleanup.d_suppress",
            {"requests": output("silent", "requests")},
        ),
    )
    case.step(
        "silent_dispatch",
        "status_dispatch",
        timeout_s=case.value("inflight_ttl_cleanup.silent_dispatch_timeout_s"),
        params={"requests": output("silent", "requests")},
    )
    case.step(
        "accepted_window",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.accepted_window_timeout_s"),
        params=case.value("inflight_ttl_cleanup.accepted_window"),
    )
    case.step(
        "six_prefill_accepted",
        "status_check",
        params=case.params(
            "inflight_ttl_cleanup.six_prefill_accepted",
            {
                "snapshot": output("accepted_window", "snapshot"),
                "baseline": output("ttl_before", "snapshot"),
            },
        ),
    )
    case.step(
        "active_at_twelve_seconds",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.active_at_twelve_seconds_timeout_s"),
        params=case.value("inflight_ttl_cleanup.active_at_twelve_seconds"),
    )
    case.step(
        "ledger_nonempty_after_twelve",
        "status_check",
        params=case.params(
            "inflight_ttl_cleanup.ledger_nonempty_after_twelve",
            {"snapshot": output("active_at_twelve_seconds", "snapshot")},
        ),
    )
    case.step(
        "ttl_drain",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.ttl_drain_timeout_s"),
        params=case.value("inflight_ttl_cleanup.ttl_drain"),
    )
    case.step(
        "scheduler_ttl_drain",
        "status_check",
        params=case.params(
            "inflight_ttl_cleanup.scheduler_ttl_drain",
            {"snapshot": output("ttl_drain", "snapshot")},
        ),
    )
    case.step(
        "ttl_event_window",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.ttl_event_window_timeout_s"),
        params=case.value("inflight_ttl_cleanup.ttl_event_window"),
    )
    case.step(
        "six_ttl_eviction_events",
        "status_check",
        params=case.params(
            "inflight_ttl_cleanup.six_ttl_eviction_events",
            {
                "snapshot": output("ttl_event_window", "snapshot"),
                "baseline": output("ttl_before", "snapshot"),
            },
        ),
    )
    case.step(
        "p_clear",
        "status_control",
        params=case.params(
            "inflight_ttl_cleanup.p_clear", {"requests": output("silent", "requests")}
        ),
    )
    case.step(
        "d_clear",
        "status_control",
        params=case.params(
            "inflight_ttl_cleanup.d_clear", {"requests": output("silent", "requests")}
        ),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("inflight_ttl_cleanup.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("inflight_ttl_cleanup.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("inflight_ttl_cleanup.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "inflight_ttl_cleanup.recovery_success",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("inflight_ttl_cleanup.final_health_timeout_s"),
        params=case.value("inflight_ttl_cleanup.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "inflight_ttl_cleanup.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def prefill_suppress_all(case):
    case.step(
        "setup", "setup", timeout_s=case.value("prefill_suppress_all.setup_timeout_s")
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=case.value("prefill_suppress_all.metrics_ready_timeout_s"),
        params=case.value("prefill_suppress_all.metrics_ready"),
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.ttl_before_timeout_s"),
        params=case.value("prefill_suppress_all.ttl_before"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("prefill_suppress_all.traffic"),
    )
    case.step(
        "on_0",
        "status_control",
        params=case.value("prefill_suppress_all.on_0"),
    )
    case.step(
        "on_1",
        "status_control",
        params=case.value("prefill_suppress_all.on_1"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("prefill_suppress_all.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("prefill_suppress_all.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params=case.params(
            "prefill_suppress_all.legal_request_terminals",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "scheduler_window",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.scheduler_window_timeout_s"),
        params=case.value("prefill_suppress_all.scheduler_window"),
    )
    case.step(
        "scheduler_retires",
        "status_check",
        params=case.params(
            "prefill_suppress_all.scheduler_retires",
            {"snapshot": output("scheduler_window", "snapshot")},
        ),
    )
    case.step(
        "prefill_window",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.prefill_window_timeout_s"),
        params=case.value("prefill_suppress_all.prefill_window"),
    )
    case.step(
        "prefill_retires",
        "status_check",
        params=case.params(
            "prefill_suppress_all.prefill_retires",
            {"snapshot": output("prefill_window", "snapshot")},
        ),
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.ttl_channel_after_timeout_s"),
        params=case.value("prefill_suppress_all.ttl_channel_after"),
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params=case.params(
            "prefill_suppress_all.scheduler_ttl_channel_reachable",
            {"snapshot": output("ttl_channel_after", "snapshot")},
        ),
    )
    case.step(
        "prefill_ttl_channel_reachable",
        "status_check",
        params=case.params(
            "prefill_suppress_all.prefill_ttl_channel_reachable",
            {"snapshot": output("ttl_channel_after", "snapshot")},
        ),
    )
    case.step(
        "off_0",
        "status_control",
        params=case.value("prefill_suppress_all.off_0"),
    )
    case.step(
        "off_1",
        "status_control",
        params=case.value("prefill_suppress_all.off_1"),
    )
    case.step(
        "final_owners",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.final_owners_timeout_s"),
        params=case.value("prefill_suppress_all.final_owners"),
    )
    case.step(
        "final_scheduler_zero",
        "status_check",
        params=case.params(
            "prefill_suppress_all.final_scheduler_zero",
            {"snapshot": output("final_owners", "snapshot")},
        ),
    )
    case.step(
        "final_prefill_batches_zero",
        "status_check",
        params=case.params(
            "prefill_suppress_all.final_prefill_batches_zero",
            {"snapshot": output("final_owners", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("prefill_suppress_all.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("prefill_suppress_all.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("prefill_suppress_all.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "prefill_suppress_all.recovery_success",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("prefill_suppress_all.final_health_timeout_s"),
        params=case.value("prefill_suppress_all.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "prefill_suppress_all.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def prefill_suppress_finished(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("prefill_suppress_finished.setup_timeout_s"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("prefill_suppress_finished.traffic"),
    )
    case.step(
        "on_0",
        "status_control",
        params=case.value("prefill_suppress_finished.on_0"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("prefill_suppress_finished.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("prefill_suppress_finished.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params=case.params(
            "prefill_suppress_finished.legal_request_terminals",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "off_0",
        "status_control",
        params=case.value("prefill_suppress_finished.off_0"),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("prefill_suppress_finished.after_clear_timeout_s"),
        params=case.value("prefill_suppress_finished.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "prefill_suppress_finished.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "prefill_suppress_finished.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "prefill_suppress_finished.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("prefill_suppress_finished.final_health_timeout_s"),
        params=case.value("prefill_suppress_finished.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "prefill_suppress_finished.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def decode_suppress_finished(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_suppress_finished.setup_timeout_s"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("decode_suppress_finished.traffic"),
    )
    case.step(
        "on_0",
        "status_control",
        params=case.value("decode_suppress_finished.on_0"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("decode_suppress_finished.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("decode_suppress_finished.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "legal_request_terminals",
        "status_outcomes",
        params=case.params(
            "decode_suppress_finished.legal_request_terminals",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "p_finished_window",
        "status_sample",
        timeout_s=case.value("decode_suppress_finished.p_finished_window_timeout_s"),
        params=case.value("decode_suppress_finished.p_finished_window"),
    )
    case.step(
        "prefill_finishes_independently",
        "status_check",
        params=case.params(
            "decode_suppress_finished.prefill_finishes_independently",
            {"snapshot": output("p_finished_window", "snapshot")},
        ),
    )
    case.step(
        "off_0",
        "status_control",
        params=case.value("decode_suppress_finished.off_0"),
    )
    case.step(
        "decode_recovery_window",
        "status_sample",
        timeout_s=case.value(
            "decode_suppress_finished.decode_recovery_window_timeout_s"
        ),
        params=case.value("decode_suppress_finished.decode_recovery_window"),
    )
    case.step(
        "decode_retires_after_clear",
        "status_check",
        params=case.params(
            "decode_suppress_finished.decode_retires_after_clear",
            {"snapshot": output("decode_recovery_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("decode_suppress_finished.final_health_timeout_s"),
        params=case.value("decode_suppress_finished.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "decode_suppress_finished.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def no_respond(case):
    case.step("setup", "setup", timeout_s=case.value("no_respond.setup_timeout_s"))
    case.step(
        "slow_prefill",
        "status_perf",
        params=case.value("no_respond.slow_prefill"),
    )
    case.step(
        "live",
        "status_prepare",
        params=case.value("no_respond.live"),
    )
    case.step(
        "live_dispatch",
        "status_dispatch",
        timeout_s=case.value("no_respond.live_dispatch_timeout_s"),
        params={"requests": output("live", "requests")},
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=case.value("no_respond.metrics_ready_timeout_s"),
        params=case.value("no_respond.metrics_ready"),
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=case.value("no_respond.ttl_before_timeout_s"),
        params=case.value("no_respond.ttl_before"),
    )
    case.step(
        "retire_on",
        "status_control",
        params=case.value("no_respond.retire_on"),
    )
    case.step(
        "some_retired_window",
        "status_sample",
        timeout_s=case.value("no_respond.some_retired_window_timeout_s"),
        params=case.value("no_respond.some_retired_window"),
    )
    case.step(
        "at_least_one_prefill_retired",
        "status_check",
        params=case.params(
            "no_respond.at_least_one_prefill_retired",
            {"snapshot": output("some_retired_window", "snapshot")},
        ),
    )
    case.step(
        "all_retired_window",
        "status_sample",
        timeout_s=case.value("no_respond.all_retired_window_timeout_s"),
        params=case.value("no_respond.all_retired_window"),
    )
    case.step(
        "all_prefill_retired",
        "status_check",
        params=case.params(
            "no_respond.all_prefill_retired",
            {"snapshot": output("all_retired_window", "snapshot")},
        ),
    )
    case.step(
        "retired_ledger_window",
        "status_sample",
        timeout_s=case.value("no_respond.retired_ledger_window_timeout_s"),
        params=case.value("no_respond.retired_ledger_window"),
    )
    case.step(
        "retired_ledger_drained",
        "status_check",
        params=case.params(
            "no_respond.retired_ledger_drained",
            {"snapshot": output("retired_ledger_window", "snapshot")},
        ),
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=case.value("no_respond.ttl_channel_after_timeout_s"),
        params=case.value("no_respond.ttl_channel_after"),
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params=case.params(
            "no_respond.scheduler_ttl_channel_reachable",
            {"snapshot": output("ttl_channel_after", "snapshot")},
        ),
    )
    case.step(
        "retire_off",
        "status_control",
        params=case.value("no_respond.retire_off"),
    )
    case.step(
        "final_scheduler",
        "status_sample",
        timeout_s=case.value("no_respond.final_scheduler_timeout_s"),
        params=case.value("no_respond.final_scheduler"),
    )
    case.step(
        "final_scheduler_zero",
        "status_check",
        params=case.params(
            "no_respond.final_scheduler_zero",
            {"snapshot": output("final_scheduler", "snapshot")},
        ),
    )
    case.step(
        "topology_recovery",
        "status_sample",
        timeout_s=case.value("no_respond.topology_recovery_timeout_s"),
        params=case.value("no_respond.topology_recovery"),
    )
    case.step(
        "topology_recovers",
        "status_check",
        params=case.params(
            "no_respond.topology_recovers",
            {"snapshot": output("topology_recovery", "snapshot")},
        ),
    )
    case.step(
        "reconnect_window",
        "status_sample",
        timeout_s=case.value("no_respond.reconnect_window_timeout_s"),
        params=case.value("no_respond.reconnect_window"),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("no_respond.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("no_respond.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("no_respond.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "no_respond.recovery_success", {"requests": output("recovery", "requests")}
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("no_respond.final_health_timeout_s"),
        params=case.value("no_respond.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "no_respond.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def version_regress(case):
    case.step("setup", "setup", timeout_s=case.value("version_regress.setup_timeout_s"))
    case.step(
        "slow_prefill",
        "status_perf",
        params=case.value("version_regress.slow_prefill"),
    )
    case.step(
        "live",
        "status_prepare",
        params=case.value("version_regress.live"),
    )
    case.step(
        "live_dispatch",
        "status_dispatch",
        timeout_s=case.value("version_regress.live_dispatch_timeout_s"),
        params={"requests": output("live", "requests")},
    )
    case.step(
        "metrics_ready",
        "status_metrics_ready",
        timeout_s=case.value("version_regress.metrics_ready_timeout_s"),
        params=case.value("version_regress.metrics_ready"),
    )
    case.step(
        "ttl_before",
        "status_sample",
        timeout_s=case.value("version_regress.ttl_before_timeout_s"),
        params=case.value("version_regress.ttl_before"),
    )
    case.step(
        "retire_on",
        "status_control",
        params=case.value("version_regress.retire_on"),
    )
    case.step(
        "some_retired_window",
        "status_sample",
        timeout_s=case.value("version_regress.some_retired_window_timeout_s"),
        params=case.value("version_regress.some_retired_window"),
    )
    case.step(
        "at_least_one_prefill_retired",
        "status_check",
        params=case.params(
            "version_regress.at_least_one_prefill_retired",
            {"snapshot": output("some_retired_window", "snapshot")},
        ),
    )
    case.step(
        "retired_ledger_window",
        "status_sample",
        timeout_s=case.value("version_regress.retired_ledger_window_timeout_s"),
        params=case.value("version_regress.retired_ledger_window"),
    )
    case.step(
        "retired_ledger_drained",
        "status_check",
        params=case.params(
            "version_regress.retired_ledger_drained",
            {"snapshot": output("retired_ledger_window", "snapshot")},
        ),
    )
    case.step(
        "ttl_channel_after",
        "status_sample",
        timeout_s=case.value("version_regress.ttl_channel_after_timeout_s"),
        params=case.value("version_regress.ttl_channel_after"),
    )
    case.step(
        "scheduler_ttl_channel_reachable",
        "status_check",
        params=case.params(
            "version_regress.scheduler_ttl_channel_reachable",
            {"snapshot": output("ttl_channel_after", "snapshot")},
        ),
    )
    case.step(
        "retire_off",
        "status_control",
        params=case.value("version_regress.retire_off"),
    )
    case.step(
        "final_scheduler",
        "status_sample",
        timeout_s=case.value("version_regress.final_scheduler_timeout_s"),
        params=case.value("version_regress.final_scheduler"),
    )
    case.step(
        "final_scheduler_zero",
        "status_check",
        params=case.params(
            "version_regress.final_scheduler_zero",
            {"snapshot": output("final_scheduler", "snapshot")},
        ),
    )
    case.step(
        "topology_recovery",
        "status_sample",
        timeout_s=case.value("version_regress.topology_recovery_timeout_s"),
        params=case.value("version_regress.topology_recovery"),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("version_regress.final_health_timeout_s"),
        params=case.value("version_regress.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "version_regress.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unknown_rid_finished(case):
    case.step(
        "setup", "setup", timeout_s=case.value("unknown_rid_finished.setup_timeout_s")
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("unknown_rid_finished.clean_baseline_timeout_s"),
        params=case.value("unknown_rid_finished.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "unknown_rid_finished.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "unknown_rid_finished.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "unknown_rid_finished.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "ghost",
        "status_prepare",
        params=case.value("unknown_rid_finished.ghost"),
    )
    case.step(
        "ghost_on",
        "status_control",
        params=case.params(
            "unknown_rid_finished.ghost_on", {"requests": output("ghost", "requests")}
        ),
    )
    case.step(
        "ghost_window",
        "status_sample",
        timeout_s=case.value("unknown_rid_finished.ghost_window_timeout_s"),
        params=case.value("unknown_rid_finished.ghost_window"),
    )
    case.step(
        "ghost_off",
        "status_control",
        params=case.value("unknown_rid_finished.ghost_off"),
    )
    case.step(
        "unknown_terminal_ignored_after_clear",
        "status_sample",
        timeout_s=case.value(
            "unknown_rid_finished.unknown_terminal_ignored_after_clear_timeout_s"
        ),
        params=case.value("unknown_rid_finished.unknown_terminal_ignored_after_clear"),
    )
    case.step(
        "unknown_terminal_ignored",
        "status_check",
        params=case.params(
            "unknown_rid_finished.unknown_terminal_ignored",
            {
                "snapshot": output("unknown_terminal_ignored_after_clear", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("unknown_rid_finished.final_health_timeout_s"),
        params=case.value("unknown_rid_finished.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "unknown_rid_finished.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unknown_rid_running(case):
    case.step(
        "setup", "setup", timeout_s=case.value("unknown_rid_running.setup_timeout_s")
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("unknown_rid_running.clean_baseline_timeout_s"),
        params=case.value("unknown_rid_running.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "unknown_rid_running.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "unknown_rid_running.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "unknown_rid_running.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "ghost",
        "status_prepare",
        params=case.value("unknown_rid_running.ghost"),
    )
    case.step(
        "ghost_on",
        "status_control",
        params=case.params(
            "unknown_rid_running.ghost_on", {"requests": output("ghost", "requests")}
        ),
    )
    case.step(
        "ghost_window",
        "status_sample",
        timeout_s=case.value("unknown_rid_running.ghost_window_timeout_s"),
        params=case.value("unknown_rid_running.ghost_window"),
    )
    case.step(
        "ghost_off",
        "status_control",
        params=case.value("unknown_rid_running.ghost_off"),
    )
    case.step(
        "unknown_active_clear_window",
        "status_sample",
        timeout_s=case.value(
            "unknown_rid_running.unknown_active_clear_window_timeout_s"
        ),
        params=case.value("unknown_rid_running.unknown_active_clear_window"),
    )
    case.step(
        "unknown_active_retires_after_clear",
        "status_check",
        params=case.params(
            "unknown_rid_running.unknown_active_retires_after_clear",
            {"snapshot": output("unknown_active_clear_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("unknown_rid_running.final_health_timeout_s"),
        params=case.value("unknown_rid_running.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "unknown_rid_running.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unknown_batchid(case):
    case.step("setup", "setup", timeout_s=case.value("unknown_batchid.setup_timeout_s"))
    case.step(
        "slow_prefill",
        "status_perf",
        params=case.value("unknown_batchid.slow_prefill"),
    )
    case.step(
        "baseline_request",
        "status_prepare",
        params=case.value("unknown_batchid.baseline_request"),
    )
    case.step(
        "baseline_request_dispatch",
        "status_dispatch",
        timeout_s=case.value("unknown_batchid.baseline_request_dispatch_timeout_s"),
        params={"requests": output("baseline_request", "requests")},
    )
    case.step(
        "baseline_request_wait",
        "wait",
        timeout_s=case.value("unknown_batchid.baseline_request_wait_timeout_s"),
        params={"requests": output("baseline_request", "requests")},
    )
    case.step(
        "baseline_request_success",
        "status_outcomes",
        params=case.params(
            "unknown_batchid.baseline_request_success",
            {"requests": output("baseline_request", "requests")},
        ),
    )
    case.step(
        "real",
        "status_prepare",
        params=case.value("unknown_batchid.real"),
    )
    case.step(
        "real_dispatch",
        "status_dispatch",
        timeout_s=case.value("unknown_batchid.real_dispatch_timeout_s"),
        params={"requests": output("real", "requests")},
    )
    case.step(
        "legacy_effective_batch_zero",
        "status_control",
        params=case.params(
            "unknown_batchid.legacy_effective_batch_zero",
            {"requests": output("real", "requests")},
        ),
    )
    case.step(
        "real_wait",
        "wait",
        timeout_s=case.value("unknown_batchid.real_wait_timeout_s"),
        params={"requests": output("real", "requests")},
    )
    case.step(
        "real_request_unaffected",
        "status_outcomes",
        params=case.params(
            "unknown_batchid.real_request_unaffected",
            {"requests": output("real", "requests")},
        ),
    )
    case.step(
        "fake_clear",
        "status_control",
        params=case.value("unknown_batchid.fake_clear"),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("unknown_batchid.after_clear_timeout_s"),
        params=case.value("unknown_batchid.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "unknown_batchid.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "unknown_batchid.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "unknown_batchid.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("unknown_batchid.final_health_timeout_s"),
        params=case.value("unknown_batchid.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "unknown_batchid.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def special_ids(case):
    case.step("setup", "setup", timeout_s=case.value("special_ids.setup_timeout_s"))
    case.step(
        "slow_prefill",
        "status_perf",
        params=case.value("special_ids.slow_prefill"),
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("special_ids.clean_baseline_timeout_s"),
        params=case.value("special_ids.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "special_ids.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "special_ids.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "special_ids.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "negative_rid_on",
        "status_control",
        params=case.value("special_ids.negative_rid_on"),
    )
    case.step(
        "negative_rid_window",
        "status_sample",
        timeout_s=case.value("special_ids.negative_rid_window_timeout_s"),
        params=case.value("special_ids.negative_rid_window"),
    )
    case.step(
        "negative_rid_clear",
        "status_control",
        params=case.value("special_ids.negative_rid_clear"),
    )
    case.step(
        "negative_rid_ignored_after_clear",
        "status_sample",
        timeout_s=case.value("special_ids.negative_rid_ignored_after_clear_timeout_s"),
        params=case.value("special_ids.negative_rid_ignored_after_clear"),
    )
    case.step(
        "negative_rid_ignored",
        "status_check",
        params=case.params(
            "special_ids.negative_rid_ignored",
            {
                "snapshot": output("negative_rid_ignored_after_clear", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "zero_rid_channel_probe",
        "status_control",
        params=case.value("special_ids.zero_rid_channel_probe"),
    )
    case.step(
        "zero_rid_clear",
        "status_control",
        params=case.value("special_ids.zero_rid_clear"),
    )
    case.step(
        "zero_real",
        "status_prepare",
        params=case.value("special_ids.zero_real"),
    )
    case.step(
        "zero_dispatch",
        "status_dispatch",
        timeout_s=case.value("special_ids.zero_dispatch_timeout_s"),
        params={"requests": output("zero_real", "requests")},
    )
    case.step(
        "zero_batch_on",
        "status_control",
        params=case.params(
            "special_ids.zero_batch_on", {"requests": output("zero_real", "requests")}
        ),
    )
    case.step(
        "zero_wait",
        "wait",
        timeout_s=case.value("special_ids.zero_wait_timeout_s"),
        params={"requests": output("zero_real", "requests")},
    )
    case.step(
        "zero_batch_unaffected",
        "status_outcomes",
        params=case.params(
            "special_ids.zero_batch_unaffected",
            {"requests": output("zero_real", "requests")},
        ),
    )
    case.step(
        "zero_clear",
        "status_control",
        params=case.value("special_ids.zero_clear"),
    )
    case.step(
        "negative_real",
        "status_prepare",
        params=case.value("special_ids.negative_real"),
    )
    case.step(
        "negative_dispatch",
        "status_dispatch",
        timeout_s=case.value("special_ids.negative_dispatch_timeout_s"),
        params={"requests": output("negative_real", "requests")},
    )
    case.step(
        "negative_batch_on",
        "status_control",
        params=case.params(
            "special_ids.negative_batch_on",
            {"requests": output("negative_real", "requests")},
        ),
    )
    case.step(
        "negative_wait",
        "wait",
        timeout_s=case.value("special_ids.negative_wait_timeout_s"),
        params={"requests": output("negative_real", "requests")},
    )
    case.step(
        "negative_batch_unaffected",
        "status_outcomes",
        params=case.params(
            "special_ids.negative_batch_unaffected",
            {"requests": output("negative_real", "requests")},
        ),
    )
    case.step(
        "negative_clear",
        "status_control",
        params=case.value("special_ids.negative_clear"),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("special_ids.after_clear_timeout_s"),
        params=case.value("special_ids.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "special_ids.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "special_ids.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "special_ids.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("special_ids.final_health_timeout_s"),
        params=case.value("special_ids.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "special_ids.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def unbatched_single_request(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("unbatched_single_request.setup_timeout_s"),
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.clean_baseline_timeout_s"),
        params=case.value("unbatched_single_request.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "unbatched_single_request.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "unbatched_single_request.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "unbatched_single_request.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "omitted_running",
        "status_prepare",
        params=case.value("unbatched_single_request.omitted_running"),
    )
    case.step(
        "omitted_running_before",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_running_before_timeout_s"
        ),
        params=case.value("unbatched_single_request.omitted_running_before"),
    )
    case.step(
        "omitted_running_on",
        "status_control",
        params=case.params(
            "unbatched_single_request.omitted_running_on",
            {"requests": output("omitted_running", "requests")},
        ),
    )
    case.step(
        "omitted_running_window",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_running_window_timeout_s"
        ),
        params=case.value("unbatched_single_request.omitted_running_window"),
    )
    case.step(
        "omitted_running_off",
        "status_control",
        params=case.value("unbatched_single_request.omitted_running_off"),
    )
    case.step(
        "omitted_running_ignored_after_clear",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_running_ignored_after_clear_timeout_s"
        ),
        params=case.value(
            "unbatched_single_request.omitted_running_ignored_after_clear"
        ),
    )
    case.step(
        "omitted_running_ignored",
        "status_check",
        params=case.params(
            "unbatched_single_request.omitted_running_ignored",
            {
                "snapshot": output("omitted_running_ignored_after_clear", "snapshot"),
                "baseline": output("omitted_running_before", "snapshot"),
            },
        ),
    )
    case.step(
        "omitted_finished",
        "status_prepare",
        params=case.value("unbatched_single_request.omitted_finished"),
    )
    case.step(
        "omitted_finished_before",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_finished_before_timeout_s"
        ),
        params=case.value("unbatched_single_request.omitted_finished_before"),
    )
    case.step(
        "omitted_finished_on",
        "status_control",
        params=case.params(
            "unbatched_single_request.omitted_finished_on",
            {"requests": output("omitted_finished", "requests")},
        ),
    )
    case.step(
        "omitted_finished_window",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_finished_window_timeout_s"
        ),
        params=case.value("unbatched_single_request.omitted_finished_window"),
    )
    case.step(
        "omitted_finished_off",
        "status_control",
        params=case.value("unbatched_single_request.omitted_finished_off"),
    )
    case.step(
        "omitted_finished_ignored_after_clear",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.omitted_finished_ignored_after_clear_timeout_s"
        ),
        params=case.value(
            "unbatched_single_request.omitted_finished_ignored_after_clear"
        ),
    )
    case.step(
        "omitted_finished_ignored",
        "status_check",
        params=case.params(
            "unbatched_single_request.omitted_finished_ignored",
            {
                "snapshot": output("omitted_finished_ignored_after_clear", "snapshot"),
                "baseline": output("omitted_finished_before", "snapshot"),
            },
        ),
    )
    case.step(
        "zero_running",
        "status_prepare",
        params=case.value("unbatched_single_request.zero_running"),
    )
    case.step(
        "zero_running_before",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.zero_running_before_timeout_s"),
        params=case.value("unbatched_single_request.zero_running_before"),
    )
    case.step(
        "zero_running_on",
        "status_control",
        params=case.params(
            "unbatched_single_request.zero_running_on",
            {"requests": output("zero_running", "requests")},
        ),
    )
    case.step(
        "zero_running_window",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.zero_running_window_timeout_s"),
        params=case.value("unbatched_single_request.zero_running_window"),
    )
    case.step(
        "zero_running_off",
        "status_control",
        params=case.value("unbatched_single_request.zero_running_off"),
    )
    case.step(
        "zero_running_ignored_after_clear",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.zero_running_ignored_after_clear_timeout_s"
        ),
        params=case.value("unbatched_single_request.zero_running_ignored_after_clear"),
    )
    case.step(
        "zero_running_ignored",
        "status_check",
        params=case.params(
            "unbatched_single_request.zero_running_ignored",
            {
                "snapshot": output("zero_running_ignored_after_clear", "snapshot"),
                "baseline": output("zero_running_before", "snapshot"),
            },
        ),
    )
    case.step(
        "zero_finished",
        "status_prepare",
        params=case.value("unbatched_single_request.zero_finished"),
    )
    case.step(
        "zero_finished_before",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.zero_finished_before_timeout_s"),
        params=case.value("unbatched_single_request.zero_finished_before"),
    )
    case.step(
        "zero_finished_on",
        "status_control",
        params=case.params(
            "unbatched_single_request.zero_finished_on",
            {"requests": output("zero_finished", "requests")},
        ),
    )
    case.step(
        "zero_finished_window",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.zero_finished_window_timeout_s"),
        params=case.value("unbatched_single_request.zero_finished_window"),
    )
    case.step(
        "zero_finished_off",
        "status_control",
        params=case.value("unbatched_single_request.zero_finished_off"),
    )
    case.step(
        "zero_finished_ignored_after_clear",
        "status_sample",
        timeout_s=case.value(
            "unbatched_single_request.zero_finished_ignored_after_clear_timeout_s"
        ),
        params=case.value("unbatched_single_request.zero_finished_ignored_after_clear"),
    )
    case.step(
        "zero_finished_ignored",
        "status_check",
        params=case.params(
            "unbatched_single_request.zero_finished_ignored",
            {
                "snapshot": output("zero_finished_ignored_after_clear", "snapshot"),
                "baseline": output("zero_finished_before", "snapshot"),
            },
        ),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.after_clear_timeout_s"),
        params=case.value("unbatched_single_request.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "unbatched_single_request.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "unbatched_single_request.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "unbatched_single_request.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("unbatched_single_request.final_health_timeout_s"),
        params=case.value("unbatched_single_request.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "unbatched_single_request.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def foreign_batchid(case):
    case.step("setup", "setup", timeout_s=case.value("foreign_batchid.setup_timeout_s"))
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("foreign_batchid.clean_baseline_timeout_s"),
        params=case.value("foreign_batchid.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "foreign_batchid.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "foreign_batchid.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "foreign_batchid.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "ghost",
        "status_prepare",
        params=case.value("foreign_batchid.ghost"),
    )
    case.step(
        "foreign_on",
        "status_control",
        params=case.params(
            "foreign_batchid.foreign_on", {"requests": output("ghost", "requests")}
        ),
    )
    case.step(
        "foreign_ghost_window",
        "status_sample",
        timeout_s=case.value("foreign_batchid.foreign_ghost_window_timeout_s"),
        params=case.value("foreign_batchid.foreign_ghost_window"),
    )
    case.step(
        "foreign_terminal_ignored",
        "status_check",
        params=case.params(
            "foreign_batchid.foreign_terminal_ignored",
            {
                "snapshot": output("foreign_ghost_window", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "real_traffic",
        "status_prepare",
        params=case.value("foreign_batchid.real_traffic"),
    )
    case.step(
        "real_traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("foreign_batchid.real_traffic_dispatch_timeout_s"),
        params={"requests": output("real_traffic", "requests")},
    )
    case.step(
        "real_traffic_wait",
        "wait",
        timeout_s=case.value("foreign_batchid.real_traffic_wait_timeout_s"),
        params={"requests": output("real_traffic", "requests")},
    )
    case.step(
        "real_traffic_success",
        "status_outcomes",
        params=case.params(
            "foreign_batchid.real_traffic_success",
            {"requests": output("real_traffic", "requests")},
        ),
    )
    case.step(
        "foreign_off",
        "status_control",
        params=case.value("foreign_batchid.foreign_off"),
    )
    case.step(
        "after_foreign",
        "status_sample",
        timeout_s=case.value("foreign_batchid.after_foreign_timeout_s"),
        params=case.value("foreign_batchid.after_foreign"),
    )
    case.step(
        "after_foreign_scheduler",
        "status_check",
        params=case.params(
            "foreign_batchid.after_foreign_scheduler",
            {"snapshot": output("after_foreign", "snapshot")},
        ),
    )
    case.step(
        "after_foreign_prefill_batches",
        "status_check",
        params=case.params(
            "foreign_batchid.after_foreign_prefill_batches",
            {"snapshot": output("after_foreign", "snapshot")},
        ),
    )
    case.step(
        "after_foreign_decode_load",
        "status_check",
        params=case.params(
            "foreign_batchid.after_foreign_decode_load",
            {"snapshot": output("after_foreign", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("foreign_batchid.final_health_timeout_s"),
        params=case.value("foreign_batchid.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "foreign_batchid.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def duplicate_finished(case):
    case.step(
        "setup", "setup", timeout_s=case.value("duplicate_finished.setup_timeout_s")
    )
    case.step(
        "duplicate_on",
        "status_control",
        params=case.value("duplicate_finished.duplicate_on"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("duplicate_finished.traffic"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("duplicate_finished.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("duplicate_finished.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_success",
        "status_outcomes",
        params=case.params(
            "duplicate_finished.traffic_success",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "initial_drain",
        "status_sample",
        timeout_s=case.value("duplicate_finished.initial_drain_timeout_s"),
        params=case.value("duplicate_finished.initial_drain"),
    )
    case.step(
        "duplicate_replay_window",
        "status_sample",
        timeout_s=case.value("duplicate_finished.duplicate_replay_window_timeout_s"),
        params=case.value("duplicate_finished.duplicate_replay_window"),
    )
    case.step(
        "duplicate_off",
        "status_control",
        params=case.value("duplicate_finished.duplicate_off"),
    )
    case.step(
        "terminal_replay_is_idempotent",
        "status_check",
        params=case.params(
            "duplicate_finished.terminal_replay_is_idempotent",
            {
                "snapshot": output("duplicate_replay_window", "snapshot"),
                "baseline": output("initial_drain", "snapshot"),
            },
        ),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("duplicate_finished.after_clear_timeout_s"),
        params=case.value("duplicate_finished.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "duplicate_finished.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "duplicate_finished.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "duplicate_finished.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("duplicate_finished.final_health_timeout_s"),
        params=case.value("duplicate_finished.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "duplicate_finished.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def cursor_regress(case):
    case.step("setup", "setup", timeout_s=case.value("cursor_regress.setup_timeout_s"))
    case.step(
        "history",
        "status_prepare",
        params=case.value("cursor_regress.history"),
    )
    case.step(
        "history_dispatch",
        "status_dispatch",
        timeout_s=case.value("cursor_regress.history_dispatch_timeout_s"),
        params={"requests": output("history", "requests")},
    )
    case.step(
        "history_wait",
        "wait",
        timeout_s=case.value("cursor_regress.history_wait_timeout_s"),
        params={"requests": output("history", "requests")},
    )
    case.step(
        "history_success",
        "status_outcomes",
        params=case.params(
            "cursor_regress.history_success",
            {"requests": output("history", "requests")},
        ),
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("cursor_regress.clean_baseline_timeout_s"),
        params=case.value("cursor_regress.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "cursor_regress.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "cursor_regress.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "cursor_regress.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "cursor_on",
        "status_control",
        params=case.value("cursor_regress.cursor_on"),
    )
    case.step(
        "replay_before",
        "status_sample",
        timeout_s=case.value("cursor_regress.replay_before_timeout_s"),
        params=case.value("cursor_regress.replay_before"),
    )
    case.step(
        "replay_window",
        "status_sample",
        timeout_s=case.value("cursor_regress.replay_window_timeout_s"),
        params=case.value("cursor_regress.replay_window"),
    )
    case.step(
        "cursor_off",
        "status_control",
        params=case.value("cursor_regress.cursor_off"),
    )
    case.step(
        "cursor_replay_is_idempotent",
        "status_check",
        params=case.params(
            "cursor_regress.cursor_replay_is_idempotent",
            {
                "snapshot": output("replay_window", "snapshot"),
                "baseline": output("replay_before", "snapshot"),
            },
        ),
    )
    case.step(
        "after_replay",
        "status_sample",
        timeout_s=case.value("cursor_regress.after_replay_timeout_s"),
        params=case.value("cursor_regress.after_replay"),
    )
    case.step(
        "after_replay_scheduler",
        "status_check",
        params=case.params(
            "cursor_regress.after_replay_scheduler",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "after_replay_prefill_batches",
        "status_check",
        params=case.params(
            "cursor_regress.after_replay_prefill_batches",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "after_replay_decode_load",
        "status_check",
        params=case.params(
            "cursor_regress.after_replay_decode_load",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("cursor_regress.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("cursor_regress.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("cursor_regress.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "cursor_regress.recovery_success",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("cursor_regress.final_health_timeout_s"),
        params=case.value("cursor_regress.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "cursor_regress.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def finished_then_running(case):
    case.step(
        "setup", "setup", timeout_s=case.value("finished_then_running.setup_timeout_s")
    )
    case.step(
        "settled",
        "status_prepare",
        params=case.value("finished_then_running.settled"),
    )
    case.step(
        "settled_dispatch",
        "status_dispatch",
        timeout_s=case.value("finished_then_running.settled_dispatch_timeout_s"),
        params={"requests": output("settled", "requests")},
    )
    case.step(
        "settled_wait",
        "wait",
        timeout_s=case.value("finished_then_running.settled_wait_timeout_s"),
        params={"requests": output("settled", "requests")},
    )
    case.step(
        "settled_success",
        "status_outcomes",
        params=case.params(
            "finished_then_running.settled_success",
            {"requests": output("settled", "requests")},
        ),
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("finished_then_running.clean_baseline_timeout_s"),
        params=case.value("finished_then_running.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "finished_then_running.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "finished_then_running.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "finished_then_running.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "terminal_replay",
        "status_control",
        params=case.params(
            "finished_then_running.terminal_replay",
            {"requests": output("settled", "requests")},
        ),
    )
    case.step(
        "terminal_window",
        "status_sample",
        timeout_s=case.value("finished_then_running.terminal_window_timeout_s"),
        params=case.value("finished_then_running.terminal_window"),
    )
    case.step(
        "terminal_off",
        "status_control",
        params=case.value("finished_then_running.terminal_off"),
    )
    case.step(
        "active_replay",
        "status_control",
        params=case.params(
            "finished_then_running.active_replay",
            {"requests": output("settled", "requests")},
        ),
    )
    case.step(
        "active_window",
        "status_sample",
        timeout_s=case.value("finished_then_running.active_window_timeout_s"),
        params=case.value("finished_then_running.active_window"),
    )
    case.step(
        "active_off",
        "status_control",
        params=case.value("finished_then_running.active_off"),
    )
    case.step(
        "terminal_cannot_resurrect",
        "status_check",
        params=case.params(
            "finished_then_running.terminal_cannot_resurrect",
            {
                "snapshot": output("active_window", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "after_replay",
        "status_sample",
        timeout_s=case.value("finished_then_running.after_replay_timeout_s"),
        params=case.value("finished_then_running.after_replay"),
    )
    case.step(
        "after_replay_scheduler",
        "status_check",
        params=case.params(
            "finished_then_running.after_replay_scheduler",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "after_replay_prefill_batches",
        "status_check",
        params=case.params(
            "finished_then_running.after_replay_prefill_batches",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "after_replay_decode_load",
        "status_check",
        params=case.params(
            "finished_then_running.after_replay_decode_load",
            {"snapshot": output("after_replay", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("finished_then_running.final_health_timeout_s"),
        params=case.value("finished_then_running.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "finished_then_running.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def zombie_completed_running(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("zombie_completed_running.setup_timeout_s"),
    )
    case.step(
        "decode_zombie_on",
        "status_control",
        params=case.value("zombie_completed_running.decode_zombie_on"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("zombie_completed_running.traffic"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("zombie_completed_running.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("zombie_completed_running.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_success",
        "status_outcomes",
        params=case.params(
            "zombie_completed_running.traffic_success",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "zombie_window",
        "status_sample",
        timeout_s=case.value("zombie_completed_running.zombie_window_timeout_s"),
        params=case.value("zombie_completed_running.zombie_window"),
    )
    case.step(
        "decode_zombie_off",
        "status_control",
        params=case.value("zombie_completed_running.decode_zombie_off"),
    )
    case.step(
        "after_zombie",
        "status_sample",
        timeout_s=case.value("zombie_completed_running.after_zombie_timeout_s"),
        params=case.value("zombie_completed_running.after_zombie"),
    )
    case.step(
        "after_zombie_scheduler",
        "status_check",
        params=case.params(
            "zombie_completed_running.after_zombie_scheduler",
            {"snapshot": output("after_zombie", "snapshot")},
        ),
    )
    case.step(
        "after_zombie_prefill_batches",
        "status_check",
        params=case.params(
            "zombie_completed_running.after_zombie_prefill_batches",
            {"snapshot": output("after_zombie", "snapshot")},
        ),
    )
    case.step(
        "after_zombie_decode_load",
        "status_check",
        params=case.params(
            "zombie_completed_running.after_zombie_decode_load",
            {"snapshot": output("after_zombie", "snapshot")},
        ),
    )
    case.step(
        "decode_load_zero",
        "status_check",
        params=case.params(
            "zombie_completed_running.decode_load_zero",
            {"snapshot": output("after_zombie", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("zombie_completed_running.final_health_timeout_s"),
        params=case.value("zombie_completed_running.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "zombie_completed_running.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def zombie_fake_running(case):
    case.step(
        "setup", "setup", timeout_s=case.value("zombie_fake_running.setup_timeout_s")
    )
    case.step(
        "clean_baseline",
        "status_sample",
        timeout_s=case.value("zombie_fake_running.clean_baseline_timeout_s"),
        params=case.value("zombie_fake_running.clean_baseline"),
    )
    case.step(
        "clean_baseline_scheduler",
        "status_check",
        params=case.params(
            "zombie_fake_running.clean_baseline_scheduler",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_prefill_batches",
        "status_check",
        params=case.params(
            "zombie_fake_running.clean_baseline_prefill_batches",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "clean_baseline_decode_load",
        "status_check",
        params=case.params(
            "zombie_fake_running.clean_baseline_decode_load",
            {"snapshot": output("clean_baseline", "snapshot")},
        ),
    )
    case.step(
        "ghosts",
        "status_prepare",
        params=case.value("zombie_fake_running.ghosts"),
    )
    case.step(
        "ghosts_on",
        "status_control",
        params=case.params(
            "zombie_fake_running.ghosts_on", {"requests": output("ghosts", "requests")}
        ),
    )
    case.step(
        "active_ghost_window",
        "status_sample",
        timeout_s=case.value("zombie_fake_running.active_ghost_window_timeout_s"),
        params=case.value("zombie_fake_running.active_ghost_window"),
    )
    case.step(
        "resident_growth_bounded",
        "status_check",
        params=case.params(
            "zombie_fake_running.resident_growth_bounded",
            {
                "snapshot": output("active_ghost_window", "snapshot"),
                "baseline": output("clean_baseline", "snapshot"),
            },
        ),
    )
    case.step(
        "master_healthy_during_active",
        "status_check",
        params=case.params(
            "zombie_fake_running.master_healthy_during_active",
            {"snapshot": output("active_ghost_window", "snapshot")},
        ),
    )
    case.step(
        "ghosts_clear",
        "status_control",
        params=case.value("zombie_fake_running.ghosts_clear"),
    )
    case.step(
        "clear_retirement_window",
        "status_sample",
        timeout_s=case.value("zombie_fake_running.clear_retirement_window_timeout_s"),
        params=case.value("zombie_fake_running.clear_retirement_window"),
    )
    case.step(
        "ghosts_retire_after_clear",
        "status_check",
        params=case.params(
            "zombie_fake_running.ghosts_retire_after_clear",
            {"snapshot": output("clear_retirement_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("zombie_fake_running.final_health_timeout_s"),
        params=case.value("zombie_fake_running.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "zombie_fake_running.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def decode_before_prefill(case):
    case.step(
        "setup", "setup", timeout_s=case.value("decode_before_prefill.setup_timeout_s")
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("decode_before_prefill.traffic"),
    )
    case.step(
        "p_terminals_off",
        "status_control",
        params=case.params(
            "decode_before_prefill.p_terminals_off",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value("decode_before_prefill.traffic_dispatch_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "traffic_wait",
        "wait",
        timeout_s=case.value("decode_before_prefill.traffic_wait_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "decode_completes_requests",
        "status_outcomes",
        params=case.params(
            "decode_before_prefill.decode_completes_requests",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "event_driven_window",
        "status_sample",
        timeout_s=case.value("decode_before_prefill.event_driven_window_timeout_s"),
        params=case.value("decode_before_prefill.event_driven_window"),
    )
    case.step(
        "decode_terminal_retires_prefill_promptly",
        "status_check",
        params=case.params(
            "decode_before_prefill.decode_terminal_retires_prefill_promptly",
            {"snapshot": output("event_driven_window", "snapshot")},
        ),
    )
    case.step(
        "fallback_retirement_window",
        "status_sample",
        timeout_s=case.value(
            "decode_before_prefill.fallback_retirement_window_timeout_s"
        ),
        params=case.value("decode_before_prefill.fallback_retirement_window"),
    )
    case.step(
        "p_terminal_restore",
        "status_control",
        params=case.params(
            "decode_before_prefill.p_terminal_restore",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "prefill_after_clear",
        "status_sample",
        timeout_s=case.value("decode_before_prefill.prefill_after_clear_timeout_s"),
        params=case.value("decode_before_prefill.prefill_after_clear"),
    )
    case.step(
        "prefill_zero_after_clear",
        "status_check",
        params=case.params(
            "decode_before_prefill.prefill_zero_after_clear",
            {"snapshot": output("prefill_after_clear", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("decode_before_prefill.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("decode_before_prefill.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("decode_before_prefill.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "decode_before_prefill.recovery_success",
            {"requests": output("recovery", "requests")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("decode_before_prefill.final_health_timeout_s"),
        params=case.value("decode_before_prefill.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "decode_before_prefill.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def decode_running_before_prefill(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_running_before_prefill.setup_timeout_s"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("decode_running_before_prefill.traffic"),
    )
    case.step(
        "p_suppress",
        "status_control",
        params=case.params(
            "decode_running_before_prefill.p_suppress",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_finished_suppress",
        "status_control",
        params=case.value("decode_running_before_prefill.d_finished_suppress"),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value(
            "decode_running_before_prefill.traffic_dispatch_timeout_s"
        ),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "nonempty_window",
        "status_sample",
        timeout_s=case.value("decode_running_before_prefill.nonempty_window_timeout_s"),
        params=case.value("decode_running_before_prefill.nonempty_window"),
    )
    case.step(
        "batch_really_dispatched",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.batch_really_dispatched",
            {"snapshot": output("nonempty_window", "snapshot")},
        ),
    )
    case.step(
        "intermediate_hold_window",
        "status_sample",
        timeout_s=case.value(
            "decode_running_before_prefill.intermediate_hold_window_timeout_s"
        ),
        params=case.value("decode_running_before_prefill.intermediate_hold_window"),
    )
    case.step(
        "intermediate_cannot_retire_prefill",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.intermediate_cannot_retire_prefill",
            {"snapshot": output("intermediate_hold_window", "snapshot")},
        ),
    )
    case.step(
        "intermediate_cannot_retire_scheduler",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.intermediate_cannot_retire_scheduler",
            {"snapshot": output("intermediate_hold_window", "snapshot")},
        ),
    )
    case.step(
        "p_clear",
        "status_control",
        params=case.params(
            "decode_running_before_prefill.p_clear",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_finished_clear",
        "status_control",
        params=case.value("decode_running_before_prefill.d_finished_clear"),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("decode_running_before_prefill.after_clear_timeout_s"),
        params=case.value("decode_running_before_prefill.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("decode_running_before_prefill.final_health_timeout_s"),
        params=case.value("decode_running_before_prefill.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "decode_running_before_prefill.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def decode_waiting_before_prefill(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_waiting_before_prefill.setup_timeout_s"),
    )
    case.step(
        "traffic",
        "status_prepare",
        params=case.value("decode_waiting_before_prefill.traffic"),
    )
    case.step(
        "p_suppress",
        "status_control",
        params=case.params(
            "decode_waiting_before_prefill.p_suppress",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_suppress",
        "status_control",
        params=case.params(
            "decode_waiting_before_prefill.d_suppress",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_waiting_on",
        "status_control",
        params=case.params(
            "decode_waiting_before_prefill.d_waiting_on",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "traffic_dispatch",
        "status_dispatch",
        timeout_s=case.value(
            "decode_waiting_before_prefill.traffic_dispatch_timeout_s"
        ),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "nonempty_window",
        "status_sample",
        timeout_s=case.value("decode_waiting_before_prefill.nonempty_window_timeout_s"),
        params=case.value("decode_waiting_before_prefill.nonempty_window"),
    )
    case.step(
        "batch_really_dispatched",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.batch_really_dispatched",
            {"snapshot": output("nonempty_window", "snapshot")},
        ),
    )
    case.step(
        "intermediate_hold_window",
        "status_sample",
        timeout_s=case.value(
            "decode_waiting_before_prefill.intermediate_hold_window_timeout_s"
        ),
        params=case.value("decode_waiting_before_prefill.intermediate_hold_window"),
    )
    case.step(
        "intermediate_cannot_retire_prefill",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.intermediate_cannot_retire_prefill",
            {"snapshot": output("intermediate_hold_window", "snapshot")},
        ),
    )
    case.step(
        "intermediate_cannot_retire_scheduler",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.intermediate_cannot_retire_scheduler",
            {"snapshot": output("intermediate_hold_window", "snapshot")},
        ),
    )
    case.step(
        "p_clear",
        "status_control",
        params=case.params(
            "decode_waiting_before_prefill.p_clear",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_clear",
        "status_control",
        params=case.params(
            "decode_waiting_before_prefill.d_clear",
            {"requests": output("traffic", "requests")},
        ),
    )
    case.step(
        "d_waiting_clear",
        "status_control",
        params=case.value("decode_waiting_before_prefill.d_waiting_clear"),
    )
    case.step(
        "after_clear",
        "status_sample",
        timeout_s=case.value("decode_waiting_before_prefill.after_clear_timeout_s"),
        params=case.value("decode_waiting_before_prefill.after_clear"),
    )
    case.step(
        "after_clear_scheduler",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.after_clear_scheduler",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_prefill_batches",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.after_clear_prefill_batches",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "after_clear_decode_load",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.after_clear_decode_load",
            {"snapshot": output("after_clear", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("decode_waiting_before_prefill.final_health_timeout_s"),
        params=case.value("decode_waiting_before_prefill.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "decode_waiting_before_prefill.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def fetch_error(case):
    case.step("setup", "setup", timeout_s=case.value("fetch_error.setup_timeout_s"))
    case.step(
        "faulted",
        "status_prepare",
        params=case.value("fetch_error.faulted"),
    )
    case.step(
        "fetch_fault_on",
        "status_control",
        params=case.value("fetch_error.fetch_fault_on"),
    )
    case.step(
        "faulted_dispatch",
        "status_dispatch",
        timeout_s=case.value("fetch_error.faulted_dispatch_timeout_s"),
        params={"requests": output("faulted", "requests")},
    )
    case.step(
        "faulted_wait",
        "wait",
        timeout_s=case.value("fetch_error.faulted_wait_timeout_s"),
        params={"requests": output("faulted", "requests")},
    )
    case.step(
        "fetch_fault_surfaces",
        "status_outcomes",
        params=case.params(
            "fetch_error.fetch_fault_surfaces",
            {"requests": output("faulted", "requests")},
        ),
    )
    case.step(
        "fetch_fault_off",
        "status_control",
        params=case.value("fetch_error.fetch_fault_off"),
    )
    case.step(
        "fresh_request",
        "status_prepare",
        params=case.value("fetch_error.fresh_request"),
    )
    case.step(
        "fresh_request_dispatch",
        "status_dispatch",
        timeout_s=case.value("fetch_error.fresh_request_dispatch_timeout_s"),
        params={"requests": output("fresh_request", "requests")},
    )
    case.step(
        "fresh_request_wait",
        "wait",
        timeout_s=case.value("fetch_error.fresh_request_wait_timeout_s"),
        params={"requests": output("fresh_request", "requests")},
    )
    case.step(
        "fresh_request_success",
        "status_outcomes",
        params=case.params(
            "fetch_error.fresh_request_success",
            {"requests": output("fresh_request", "requests")},
        ),
    )
    case.step(
        "after_fetch_error",
        "status_sample",
        timeout_s=case.value("fetch_error.after_fetch_error_timeout_s"),
        params=case.value("fetch_error.after_fetch_error"),
    )
    case.step(
        "after_fetch_error_scheduler",
        "status_check",
        params=case.params(
            "fetch_error.after_fetch_error_scheduler",
            {"snapshot": output("after_fetch_error", "snapshot")},
        ),
    )
    case.step(
        "after_fetch_error_prefill_batches",
        "status_check",
        params=case.params(
            "fetch_error.after_fetch_error_prefill_batches",
            {"snapshot": output("after_fetch_error", "snapshot")},
        ),
    )
    case.step(
        "after_fetch_error_decode_load",
        "status_check",
        params=case.params(
            "fetch_error.after_fetch_error_decode_load",
            {"snapshot": output("after_fetch_error", "snapshot")},
        ),
    )
    case.step(
        "engine_after_fetch",
        "status_sample",
        timeout_s=case.value("fetch_error.engine_after_fetch_timeout_s"),
        params=case.value("fetch_error.engine_after_fetch"),
    )
    case.step(
        "prefill_engine_drained",
        "status_check",
        params=case.params(
            "fetch_error.prefill_engine_drained",
            {"snapshot": output("engine_after_fetch", "snapshot")},
        ),
    )
    case.step(
        "recovery",
        "status_prepare",
        params=case.value("fetch_error.recovery"),
    )
    case.step(
        "recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("fetch_error.recovery_dispatch_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_wait",
        "wait",
        timeout_s=case.value("fetch_error.recovery_wait_timeout_s"),
        params={"requests": output("recovery", "requests")},
    )
    case.step(
        "recovery_success",
        "status_outcomes",
        params=case.params(
            "fetch_error.recovery_success", {"requests": output("recovery", "requests")}
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("fetch_error.final_health_timeout_s"),
        params=case.value("fetch_error.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "fetch_error.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def debug_snapshot(case):
    case.step("setup", "setup", timeout_s=case.value("debug_snapshot.setup_timeout_s"))
    case.step(
        "debug_before",
        "status_sample",
        timeout_s=case.value("debug_snapshot.debug_before_timeout_s"),
        params=case.value("debug_snapshot.debug_before"),
    )
    case.step(
        "completed",
        "status_prepare",
        params=case.value("debug_snapshot.completed"),
    )
    case.step(
        "completed_dispatch",
        "status_dispatch",
        timeout_s=case.value("debug_snapshot.completed_dispatch_timeout_s"),
        params={"requests": output("completed", "requests")},
    )
    case.step(
        "completed_wait",
        "wait",
        timeout_s=case.value("debug_snapshot.completed_wait_timeout_s"),
        params={"requests": output("completed", "requests")},
    )
    case.step(
        "completed_success",
        "status_outcomes",
        params=case.params(
            "debug_snapshot.completed_success",
            {"requests": output("completed", "requests")},
        ),
    )
    case.step(
        "tombstone_window",
        "status_sample",
        timeout_s=case.value("debug_snapshot.tombstone_window_timeout_s"),
        params=case.params(
            "debug_snapshot.tombstone_window",
            {"requests": output("completed", "requests")},
        ),
    )
    case.step(
        "resource_free_queryable_tombstone",
        "status_check",
        params=case.params(
            "debug_snapshot.resource_free_queryable_tombstone",
            {"snapshot": output("tombstone_window", "snapshot")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("debug_snapshot.final_health_timeout_s"),
        params=case.value("debug_snapshot.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "debug_snapshot.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")


def normal_no_fetch(case):
    case.step("setup", "setup", timeout_s=case.value("normal_no_fetch.setup_timeout_s"))
    case.step(
        "fresh_baseline",
        "status_sample",
        timeout_s=case.value("normal_no_fetch.fresh_baseline_timeout_s"),
        params=case.value("normal_no_fetch.fresh_baseline"),
    )
    case.step(
        "fresh_accepted_zero",
        "status_check",
        params=case.params(
            "normal_no_fetch.fresh_accepted_zero",
            {"snapshot": output("fresh_baseline", "snapshot")},
        ),
    )
    case.step(
        "fresh_fetch_zero",
        "status_check",
        params=case.params(
            "normal_no_fetch.fresh_fetch_zero",
            {"snapshot": output("fresh_baseline", "snapshot")},
        ),
    )
    case.step(
        "unfetched",
        "status_prepare",
        params=case.value("normal_no_fetch.unfetched"),
    )
    case.step(
        "schedule_only",
        "status_dispatch",
        timeout_s=case.value("normal_no_fetch.schedule_only_timeout_s"),
        params={"requests": output("unfetched", "requests")},
    )
    case.step(
        "prefill_completion_window",
        "status_sample",
        timeout_s=case.value("normal_no_fetch.prefill_completion_window_timeout_s"),
        params=case.params(
            "normal_no_fetch.prefill_completion_window",
            {"requests": output("unfetched", "requests")},
        ),
    )
    case.step(
        "prefill_completed",
        "status_check",
        params=case.params(
            "normal_no_fetch.prefill_completed",
            {"snapshot": output("prefill_completion_window", "snapshot")},
        ),
    )
    case.step(
        "master_enqueued",
        "status_check",
        params=case.params(
            "normal_no_fetch.master_enqueued",
            {"snapshot": output("prefill_completion_window", "snapshot")},
        ),
    )
    case.step(
        "completion_window_no_fetch",
        "status_check",
        params=case.params(
            "normal_no_fetch.completion_window_no_fetch",
            {"snapshot": output("prefill_completion_window", "snapshot")},
        ),
    )
    case.step(
        "post_completion_window",
        "status_sample",
        timeout_s=case.value("normal_no_fetch.post_completion_window_timeout_s"),
        params=case.params(
            "normal_no_fetch.post_completion_window",
            {"requests": output("unfetched", "requests")},
        ),
    )
    case.step(
        "prefill_stays_completed",
        "status_check",
        params=case.params(
            "normal_no_fetch.prefill_stays_completed",
            {"snapshot": output("post_completion_window", "snapshot")},
        ),
    )
    case.step(
        "post_completion_no_fetch",
        "status_check",
        params=case.params(
            "normal_no_fetch.post_completion_no_fetch",
            {"snapshot": output("post_completion_window", "snapshot")},
        ),
    )
    case.step(
        "client_never_fetches",
        "status_check",
        params=case.params(
            "normal_no_fetch.client_never_fetches",
            {"snapshot": output("post_completion_window", "snapshot")},
        ),
    )
    case.step(
        "one_prefill_acceptance",
        "status_check",
        params=case.params(
            "normal_no_fetch.one_prefill_acceptance",
            {"snapshot": output("post_completion_window", "snapshot")},
        ),
    )
    case.step(
        "separate_recovery",
        "status_prepare",
        params=case.value("normal_no_fetch.separate_recovery"),
    )
    case.step(
        "separate_recovery_dispatch",
        "status_dispatch",
        timeout_s=case.value("normal_no_fetch.separate_recovery_dispatch_timeout_s"),
        params={"requests": output("separate_recovery", "requests")},
    )
    case.step(
        "separate_recovery_wait",
        "wait",
        timeout_s=case.value("normal_no_fetch.separate_recovery_wait_timeout_s"),
        params={"requests": output("separate_recovery", "requests")},
    )
    case.step(
        "separate_recovery_success",
        "status_outcomes",
        params=case.params(
            "normal_no_fetch.separate_recovery_success",
            {"requests": output("separate_recovery", "requests")},
        ),
    )
    case.step(
        "final_health",
        "status_sample",
        timeout_s=case.value("normal_no_fetch.final_health_timeout_s"),
        params=case.value("normal_no_fetch.final_health"),
    )
    case.step(
        "master_http_200",
        "status_check",
        params=case.params(
            "normal_no_fetch.master_http_200",
            {"snapshot": output("final_health", "snapshot")},
        ),
    )
    case.step("cleanup", "teardown")
