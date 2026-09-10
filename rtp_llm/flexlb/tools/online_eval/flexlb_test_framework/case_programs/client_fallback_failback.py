"""Explicit client routing programs: all-master outage, wraparound recovery, no fallback for business/deadline errors, and direct GenerateStreamCall fault recovery."""

from ..case_config import output


def all_masters_down(case):
    case.step(
        "setup", "setup", timeout_s=case.value("all_masters_down.setup_timeout_s")
    )
    case.step(
        "flow",
        "master_client_start",
        params=case.value("all_masters_down.flow"),
    )
    case.step(
        "kill_a_time", "master_mark", params=case.value("all_masters_down.kill_a_time")
    )
    case.step("kill_a", "master_fault", params=case.value("all_masters_down.kill_a"))
    case.step(
        "kill_b_time", "master_mark", params=case.value("all_masters_down.kill_b_time")
    )
    case.step("kill_b", "master_fault", params=case.value("all_masters_down.kill_b"))
    case.step(
        "outage_end", "master_mark", params=case.value("all_masters_down.outage_end")
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("all_masters_down.finish_timeout_s"),
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
        params=case.params(
            "all_masters_down.fallback_rows",
            {
                "rows": output("finish", "rows"),
                "from": output("kill_b_time", "epoch_s"),
                "until": output("outage_end", "epoch_s"),
            },
        ),
    )
    case.step(
        "steady_master",
        "master_client_check",
        params=case.params(
            "all_masters_down.steady_master", {"rows": output("steady", "rows")}
        ),
    )
    case.step(
        "fallback_success",
        "master_client_check",
        params=case.params(
            "all_masters_down.fallback_success",
            {"rows": output("fallback_rows", "rows")},
        ),
    )
    case.step(
        "fallback_share",
        "master_client_check",
        params=case.params(
            "all_masters_down.fallback_share", {"rows": output("outage", "rows")}
        ),
    )
    case.step(
        "no_master_during_outage",
        "master_client_check",
        params=case.params(
            "all_masters_down.no_master_during_outage",
            {"rows": output("outage", "rows")},
        ),
    )
    case.step(
        "outage_errors",
        "master_client_check",
        params=case.params(
            "all_masters_down.outage_errors", {"rows": output("outage", "rows")}
        ),
    )
    case.step(
        "unique_requests",
        "master_client_check",
        params=case.params(
            "all_masters_down.unique_requests", {"rows": output("finish", "rows")}
        ),
    )
    case.step("cleanup", "teardown")


def wraparound(case):
    window = case.number("checkpoint_window_s")
    owner_limit = case.number("max_owner_load")
    flow = output("flow", "client")

    def checkpoint(name, target, *, fault=None, baseline=True):
        params = case.params(
            "wraparound.params",
            {
                "client": flow,
                "target": target,
                "duration_s": window,
                "max_owner_load": owner_limit,
            },
        )
        if fault:
            params.update(
                fault=output(fault, "fault"),
                min_target_share=case.value("wraparound.checkpoint.min_target_share"),
                min_success=case.value("wraparound.checkpoint.min_success"),
                max_prefill_share=case.value("wraparound.checkpoint.max_prefill_share"),
            )
        if baseline:
            params["baseline"] = output("baseline_a", "snapshot")
        case.step(
            name,
            "master_client_checkpoint",
            timeout_s=window + case.value("wraparound.checkpoint_timeout_margin_s"),
            params=params,
        )

    case.step("setup", "setup", timeout_s=case.value("wraparound.setup_timeout_s"))
    case.step(
        "flow",
        "master_client_start",
        params=case.value("wraparound.flow"),
    )
    checkpoint("baseline_a", "A", baseline=False)
    case.step("kill_a", "master_fault", params=case.value("wraparound.kill_a"))
    checkpoint("switch_to_b", "B", fault="kill_a")
    checkpoint("steady_b", "B")
    case.step(
        "restart_a",
        "master_restore",
        timeout_s=case.value("wraparound.restart_a_timeout_s"),
        params={"fault": output("kill_a", "fault")},
    )
    case.step(
        "ready_a",
        "master_topology_ready",
        timeout_s=case.value("wraparound.ready_a_timeout_s"),
        params=case.value("wraparound.ready_a"),
    )
    case.step(
        "recovery_a",
        "master_request_batch",
        timeout_s=case.value("wraparound.recovery_a_timeout_s"),
        params=case.value("wraparound.recovery_a"),
    )
    case.step(
        "recovery_distribution",
        "master_probe_distribution",
        params=case.params(
            "wraparound.recovery_distribution",
            {"requests": output("recovery_a", "requests")},
        ),
    )
    # A owns no active requests here; B still drives their shared engines.
    case.step(
        "idle_a",
        "master_owner_idle",
        timeout_s=case.value("wraparound.idle_a_timeout_s"),
        params=case.value("wraparound.idle_a"),
    )
    checkpoint("coexist_sticky_b", "B")
    case.step("kill_b", "master_fault", params=case.value("wraparound.kill_b"))
    checkpoint("switch_to_a", "A", fault="kill_b")
    checkpoint("steady_a", "A")
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("wraparound.finish_timeout_s"),
        params={"client": flow},
    )
    case.step(
        "all_requests_accounted",
        "master_client_reconcile",
        params={"client": flow, "rows": output("finish", "rows")},
    )
    case.step(
        "overall_success",
        "master_client_check",
        params=case.params(
            "wraparound.overall_success", {"rows": output("finish", "rows")}
        ),
    )
    case.step(
        "clean_a",
        "master_inflight_clean",
        timeout_s=case.value("wraparound.clean_a_timeout_s"),
        params=case.value("wraparound.clean_a"),
    )
    case.step(
        "rotation_probe",
        "master_request_batch",
        timeout_s=case.value("wraparound.rotation_probe_timeout_s"),
        params=case.value("wraparound.rotation_probe"),
    )
    case.step(
        "rotation_distribution",
        "master_probe_distribution",
        params=case.params(
            "wraparound.rotation_distribution",
            {"requests": output("rotation_probe", "requests")},
        ),
    )
    case.step("cleanup", "teardown")


def negative_errorcode(case):
    case.step(
        "setup", "setup", timeout_s=case.value("negative_errorcode.setup_timeout_s")
    )
    case.step(
        "flow",
        "master_client_start",
        params=case.value("negative_errorcode.flow"),
    )
    case.step(
        "freeze_begin",
        "master_mark",
        params=case.value("negative_errorcode.freeze_begin"),
    )
    case.step(
        "freeze_a", "master_fault", params=case.value("negative_errorcode.freeze_a")
    )
    case.step(
        "freeze_end", "master_mark", params=case.value("negative_errorcode.freeze_end")
    )
    case.step("thaw_a", "master_restore", params={"fault": output("freeze_a", "fault")})
    case.step("settled", "master_mark", params=case.value("negative_errorcode.settled"))
    case.step(
        "business_fault",
        "engine_inject",
        params=case.value("negative_errorcode.business_fault"),
    )
    case.step(
        "business_begin",
        "master_mark",
        params=case.value("negative_errorcode.business_begin"),
    )
    case.step(
        "business_end",
        "master_mark",
        params=case.value("negative_errorcode.business_end"),
    )
    case.step(
        "clear_business",
        "engine_clear",
        params={"fault": output("business_fault", "fault")},
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("negative_errorcode.finish_timeout_s"),
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
        params=case.params(
            "negative_errorcode.schedule_errors",
            {
                "rows": output("finish", "rows"),
                "from": output("business_begin", "epoch_s"),
                "until": output("business_end", "epoch_s"),
            },
        ),
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
        params=case.params(
            "negative_errorcode.deadlines",
            {
                "rows": output("finish", "rows"),
                "from": output("freeze_begin", "epoch_s"),
                "until": output("freeze_end", "epoch_s"),
            },
        ),
    )
    case.step(
        "business_errors_seen",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_errors_seen",
            {"rows": output("schedule_errors", "rows")},
        ),
    )
    case.step(
        "business_code",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_code",
            {"rows": output("schedule_errors", "rows")},
        ),
    )
    case.step(
        "business_route",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_route", {"rows": output("business", "rows")}
        ),
    )
    case.step(
        "business_no_fallback",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_no_fallback",
            {"rows": output("business", "rows")},
        ),
    )
    case.step(
        "business_no_failed",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_no_failed",
            {"rows": output("business", "rows")},
        ),
    )
    case.step(
        "business_no_retry",
        "master_client_check",
        params=case.params(
            "negative_errorcode.business_no_retry", {"rows": output("business", "rows")}
        ),
    )
    case.step(
        "deadlines_seen",
        "master_client_check",
        params=case.params(
            "negative_errorcode.deadlines_seen", {"rows": output("deadlines", "rows")}
        ),
    )
    case.step(
        "deadline_route",
        "master_client_check",
        params=case.params(
            "negative_errorcode.deadline_route", {"rows": output("deadlines", "rows")}
        ),
    )
    case.step(
        "deadline_no_retry",
        "master_client_check",
        params=case.params(
            "negative_errorcode.deadline_no_retry",
            {"rows": output("deadlines", "rows")},
        ),
    )
    case.step(
        "deadline_no_fallback",
        "master_client_check",
        params=case.params(
            "negative_errorcode.deadline_no_fallback",
            {"rows": output("deadline_window", "rows")},
        ),
    )
    case.step(
        "tail_settle",
        "master_wait_inflight",
        timeout_s=case.value("negative_errorcode.tail_settle_timeout_s"),
        params=case.value("negative_errorcode.tail_settle"),
    )
    case.step("cleanup", "teardown")


def direct_generate_error(case):
    case.step(
        "setup", "setup", timeout_s=case.value("direct_generate_error.setup_timeout_s")
    )
    case.step(
        "baseline",
        "master_direct_request",
        timeout_s=case.value("direct_generate_error.baseline_timeout_s"),
        params=case.value("direct_generate_error.baseline"),
    )
    case.step(
        "baseline_finished",
        "check",
        params=case.params(
            "direct_generate_error.baseline_finished",
            {"actual": output("baseline", "finished")},
        ),
    )
    case.step(
        "baseline_no_error",
        "check",
        params=case.params(
            "direct_generate_error.baseline_no_error",
            {"actual": output("baseline", "error")},
        ),
    )
    case.step(
        "generate_fault",
        "engine_inject",
        params=case.value("direct_generate_error.generate_fault"),
    )
    case.step(
        "injected",
        "master_direct_request",
        timeout_s=case.value("direct_generate_error.injected_timeout_s"),
        params=case.value("direct_generate_error.injected"),
    )
    case.step(
        "injected_error",
        "check",
        params=case.params(
            "direct_generate_error.injected_error",
            {"actual": output("injected", "error")},
        ),
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
        timeout_s=case.value("direct_generate_error.recovery_timeout_s"),
        params=case.value("direct_generate_error.recovery"),
    )
    case.step(
        "recovered_finished",
        "check",
        params=case.params(
            "direct_generate_error.recovered_finished",
            {"actual": output("recovery", "finished")},
        ),
    )
    case.step(
        "recovered_no_error",
        "check",
        params=case.params(
            "direct_generate_error.recovered_no_error",
            {"actual": output("recovery", "error")},
        ),
    )
    case.step("recovered_clean", "master_direct_clean")
    case.step("cleanup", "teardown")
