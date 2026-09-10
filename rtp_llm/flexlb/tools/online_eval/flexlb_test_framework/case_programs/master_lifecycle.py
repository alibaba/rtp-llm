"""Cold restart of the owned master, clean scheduler state, topology and request recovery."""

from ..case_config import output


def kill_single(case):
    case.step("setup", "setup", timeout_s=case.value("kill_single.setup_timeout_s"))
    case.step("baseline", "request", params=case.value("kill_single.baseline"))
    case.step(
        "baseline_done", "wait", params={"requests": output("baseline", "requests")}
    )
    case.step(
        "baseline_complete",
        "check",
        params=case.params(
            "kill_single.baseline_complete",
            {"actual": output("baseline_done", "completed")},
        ),
    )
    case.step(
        "baseline_no_errors",
        "check",
        params=case.params(
            "kill_single.baseline_no_errors",
            {"actual": output("baseline_done", "error_count")},
        ),
    )
    case.step("kill", "master_fault", params=case.value("kill_single.kill"))
    case.step(
        "restart",
        "master_restore",
        timeout_s=case.value("kill_single.restart_timeout_s"),
        params={"fault": output("kill", "fault")},
    )
    case.step(
        "restored_topology",
        "master_ready",
        timeout_s=case.value("kill_single.restored_topology_timeout_s"),
        params=case.value("kill_single.restored_topology"),
    )
    case.step(
        "restored_inflight",
        "master_ready",
        timeout_s=case.value("kill_single.restored_inflight_timeout_s"),
        params=case.value("kill_single.restored_inflight"),
    )
    case.step("recovery", "request", params=case.value("kill_single.recovery"))
    case.step(
        "recovery_done", "wait", params={"requests": output("recovery", "requests")}
    )
    case.step(
        "recovery_complete",
        "check",
        params=case.params(
            "kill_single.recovery_complete",
            {"actual": output("recovery_done", "completed")},
        ),
    )
    case.step(
        "recovery_no_errors",
        "check",
        params=case.params(
            "kill_single.recovery_no_errors",
            {"actual": output("recovery_done", "error_count")},
        ),
    )
    case.step("cleanup", "teardown")


def freeze_short_long(case):
    case.step(
        "setup", "setup", timeout_s=case.value("freeze_short_long.setup_timeout_s")
    )
    case.step(
        "flow", "master_client_start", params=case.value("freeze_short_long.flow")
    )
    case.step(
        "short_begin", "master_mark", params=case.value("freeze_short_long.short_begin")
    )
    case.step(
        "short_freeze",
        "master_fault",
        params=case.value("freeze_short_long.short_freeze"),
    )
    case.step(
        "short_end", "master_mark", params=case.value("freeze_short_long.short_end")
    )
    case.step(
        "short_restore",
        "master_restore",
        params={"fault": output("short_freeze", "fault")},
    )
    case.step(
        "post_short_end",
        "master_mark",
        params=case.value("freeze_short_long.post_short_end"),
    )
    case.step("before", "master_state", params=case.value("freeze_short_long.before"))
    case.step(
        "long_begin", "master_mark", params=case.value("freeze_short_long.long_begin")
    )
    case.step(
        "long_freeze",
        "master_fault",
        params=case.value("freeze_short_long.long_freeze"),
    )
    # The HA flow may succeed through A; use a no-fallback request to frozen B
    # to exercise the deadline path deterministically.
    case.step(
        "deadline_probe",
        "master_request_batch",
        timeout_s=case.value("freeze_short_long.deadline_probe_timeout_s"),
        params=case.value("freeze_short_long.deadline_probe"),
    )
    case.step(
        "long_end", "master_mark", params=case.value("freeze_short_long.long_end")
    )
    case.step(
        "long_restore",
        "master_restore",
        params={"fault": output("long_freeze", "fault")},
    )
    case.step(
        "after", "master_scheduler_state", params=case.value("freeze_short_long.after")
    )
    case.step(
        "post_long_end",
        "master_mark",
        params=case.value("freeze_short_long.post_long_end"),
    )
    case.step(
        "ready_b",
        "master_topology_state",
        params=case.value("freeze_short_long.ready_b"),
    )
    case.step(
        "continuity",
        "master_continuity",
        params={
            "before": output("before", "state"),
            "after": output("after", "state"),
            "settled": output("ready_b", "state"),
        },
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("freeze_short_long.finish_timeout_s"),
        params={"client": output("flow", "client")},
    )
    case.step(
        "short_hang",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("short_begin", "epoch_s"),
            "until": output("short_end", "epoch_s"),
        },
    )
    case.step(
        "short_post",
        "master_client_window",
        params=case.params(
            "freeze_short_long.short_post",
            {
                "rows": output("finish", "rows"),
                "from": output("short_end", "epoch_s"),
                "until": output("post_short_end", "epoch_s"),
            },
        ),
    )
    case.step(
        "short_burst",
        "master_client_window",
        params=case.params(
            "freeze_short_long.short_burst",
            {
                "rows": output("finish", "rows"),
                "from": output("short_end", "epoch_s"),
                "until": output("long_begin", "epoch_s"),
            },
        ),
    )
    case.step(
        "short_verdict",
        "master_short_hang_check",
        params=case.params(
            "freeze_short_long.short_verdict",
            {
                "hang": output("short_hang", "rows"),
                "burst": output("short_burst", "rows"),
                "post": output("short_post", "rows"),
            },
        ),
    )
    case.step(
        "judged",
        "master_client_window",
        params=case.params(
            "freeze_short_long.judged",
            {
                "rows": output("finish", "rows"),
                "from": output("long_begin", "epoch_s"),
                "until": output("long_end", "epoch_s"),
            },
        ),
    )
    case.step(
        "pre_freeze",
        "master_client_window",
        params=case.params(
            "freeze_short_long.pre_freeze",
            {
                "rows": output("finish", "rows"),
                "from": output("long_begin", "epoch_s"),
                "until": output("long_begin", "epoch_s"),
            },
        ),
    )
    case.step(
        "deadline_straddle",
        "master_client_window",
        params=case.params(
            "freeze_short_long.deadline_straddle",
            {
                "rows": output("finish", "rows"),
                "from": output("long_begin", "epoch_s"),
                "until": output("long_end", "epoch_s"),
            },
        ),
    )
    case.step(
        "post_long",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("long_end", "epoch_s"),
            "until": output("post_long_end", "epoch_s"),
        },
    )
    case.step(
        "retry_seen",
        "master_client_check",
        params=case.params(
            "freeze_short_long.retry_seen",
            {"rows": output("deadline_straddle", "rows")},
        ),
    )
    case.step(
        "switch_to_a",
        "master_client_check",
        params=case.params(
            "freeze_short_long.switch_to_a", {"rows": output("judged", "rows")}
        ),
    )
    case.step(
        "visible_terminal",
        "master_client_check",
        params=case.params(
            "freeze_short_long.visible_terminal", {"rows": output("pre_freeze", "rows")}
        ),
    )
    case.step(
        "deadline_visible",
        "master_deadline_probe_check",
        params={"snapshot": output("deadline_probe", "snapshot")},
    )
    case.observe(
        "post_success",
        "master_client_check",
        params=case.params(
            "freeze_short_long.post_success", {"rows": output("post_long", "rows")}
        ),
    )
    case.observe(
        "post_on_a",
        "master_client_check",
        params=case.params(
            "freeze_short_long.post_on_a", {"rows": output("post_long", "rows")}
        ),
    )
    case.observe(
        "unique_requests",
        "master_client_check",
        params=case.params(
            "freeze_short_long.unique_requests", {"rows": output("finish", "rows")}
        ),
    )
    case.step("cleanup", "teardown")


def kill_dual_b_to_a(case):
    case.step(
        "setup", "setup", timeout_s=case.value("kill_dual_b_to_a.setup_timeout_s")
    )
    case.step("flow", "master_client_start", params=case.value("kill_dual_b_to_a.flow"))
    case.step(
        "kill_time", "master_mark", params=case.value("kill_dual_b_to_a.kill_time")
    )
    case.step("kill_b", "master_fault", params=case.value("kill_dual_b_to_a.kill_b"))
    case.step("switched", "master_mark", params=case.value("kill_dual_b_to_a.switched"))
    case.step(
        "restart_b",
        "master_restore",
        timeout_s=case.value("kill_dual_b_to_a.restart_b_timeout_s"),
        params={"fault": output("kill_b", "fault")},
    )
    case.step(
        "ready_b",
        "master_ready",
        timeout_s=case.value("kill_dual_b_to_a.ready_b_timeout_s"),
        params=case.value("kill_dual_b_to_a.ready_b"),
    )
    case.step(
        "clean_b",
        "master_ready",
        timeout_s=case.value("kill_dual_b_to_a.clean_b_timeout_s"),
        params=case.params(
            "kill_dual_b_to_a.clean_b", {"client": output("flow", "client")}
        ),
    )
    case.step(
        "recovery_a",
        "master_request_batch",
        timeout_s=case.value("kill_dual_b_to_a.recovery_a_timeout_s"),
        params=case.value("kill_dual_b_to_a.recovery_a"),
    )
    case.step(
        "recovery_rate",
        "check",
        params=case.params(
            "kill_dual_b_to_a.recovery_rate",
            {"actual": output("recovery_a", "success_rate")},
        ),
    )
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("kill_dual_b_to_a.finish_timeout_s"),
        params={"client": output("flow", "client")},
    )
    case.step(
        "steady_plain",
        "master_client_window",
        params=case.params(
            "kill_dual_b_to_a.steady_plain",
            {"rows": output("finish", "rows"), "until": output("kill_time", "epoch_s")},
        ),
    )
    case.step(
        "straddle",
        "master_client_window",
        params=case.params(
            "kill_dual_b_to_a.straddle",
            {
                "rows": output("finish", "rows"),
                "from": output("kill_time", "epoch_s"),
                "until": output("switched", "epoch_s"),
            },
        ),
    )
    case.step(
        "switch",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("kill_time", "epoch_s"),
            "until": output("switched", "epoch_s"),
        },
    )
    case.step(
        "after",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("switched", "epoch_s"),
        },
    )
    case.observe(
        "steady_b",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.steady_b", {"rows": output("steady_plain", "rows")}
        ),
    )
    case.observe(
        "retry_seen",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.retry_seen", {"rows": output("straddle", "rows")}
        ),
    )
    case.observe(
        "switch_to_a",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.switch_to_a", {"rows": output("switch", "rows")}
        ),
    )
    case.observe(
        "switch_errors",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.switch_errors", {"rows": output("switch", "rows")}
        ),
    )
    case.observe(
        "after_a",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.after_a", {"rows": output("after", "rows")}
        ),
    )
    case.observe(
        "after_success",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.after_success", {"rows": output("after", "rows")}
        ),
    )
    case.observe(
        "unique_requests",
        "master_client_check",
        params=case.params(
            "kill_dual_b_to_a.unique_requests", {"rows": output("finish", "rows")}
        ),
    )
    case.step("cleanup", "teardown")
