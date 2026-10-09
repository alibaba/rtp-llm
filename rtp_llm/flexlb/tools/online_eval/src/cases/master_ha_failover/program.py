"""One continuous real-trace replay across a dual-master restart cycle."""

from cases.master_ha_failover.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from scenario.loader import ScenarioError


FLOW_PROGRAMS = ("non_rolling",)


def default(case):
    _cycle(case, "rolling")


def non_rolling(case):
    _cycle(case, "non_rolling")


def _cycle(case, restart_mode):
    root = "dual_master_cycle"
    if "restart_mode" in case.value(root):
        raise ScenarioError("restart_mode is owned by flow identity, not parameters")
    case.step("setup", "setup", timeout_s=case.value(f"{root}.setup_timeout_s"))
    case.step("flow", "master_client_start", params=case.value(f"{root}.flow"))

    # The producer keeps running while A and B are killed and restarted in order.
    case.step("baseline_end", "master_mark", params=case.value(f"{root}.baseline_wait"))
    case.step("kill_a", "master_fault", params=case.value(f"{root}.kill_a"))
    case.step("b_ready", "master_ready", params=case.value(f"{root}.b_ready"))
    case.step("b_start", "master_mark", params=case.value(f"{root}.settle"))
    case.step("b_end", "master_mark", params=case.value(f"{root}.survivor_wait"))
    if restart_mode == "non_rolling":
        case.step("kill_b", "master_fault", params=case.value(f"{root}.kill_b"))
        case.step("outage_start", "master_mark", params=case.value(f"{root}.settle"))
        case.step("outage_end", "master_mark", params=case.value(f"{root}.outage_wait"))
    case.step("restart_a", "master_restore", timeout_s=case.value(f"{root}.restart_timeout_s"),
              params={"fault": output("kill_a", "fault")})
    case.step("a_ready", "master_ready", params=case.value(f"{root}.a_ready"))
    if restart_mode == "rolling":
        case.step("kill_b", "master_fault", params=case.value(f"{root}.kill_b"))
    case.step("a_start", "master_mark", params=case.value(f"{root}.settle"))
    case.step("a_end", "master_mark", params=case.value(f"{root}.survivor_wait"))
    case.step("restart_b", "master_restore", timeout_s=case.value(f"{root}.restart_timeout_s"),
              params={"fault": output("kill_b", "fault")})
    case.step("a_ready_final", "master_ready", params=case.value(f"{root}.a_ready"))
    case.step("b_ready_final", "master_ready", params=case.value(f"{root}.b_ready"))
    case.step("both_start", "master_mark", params=case.value(f"{root}.settle"))
    case.step("both_end", "master_mark", params=case.value(f"{root}.both_wait"))
    case.step("finish", "master_client_finish", timeout_s=case.value(f"{root}.finish_timeout_s"),
              params={"client": output("flow", "client"), "stop_sending": True})

    windows = {
        "baseline": {"until": output("baseline_end", "epoch_s"), "until_offset_s": -2},
        "b_only": {"from": output("b_start", "epoch_s"), "until": output("b_end", "epoch_s")},
        "a_only": {"from": output("a_start", "epoch_s"), "until": output("a_end", "epoch_s")},
        "both": {"from": output("both_start", "epoch_s"), "until": output("both_end", "epoch_s")},
    }
    if restart_mode == "non_rolling":
        windows["outage"] = {"from": output("outage_start", "epoch_s"),
                              "until": output("outage_end", "epoch_s")}
    else:
        # The existing steady windows intentionally skip restart seams and
        # stop after 30 seconds of the final long-running traffic period.
        windows["a_handover"] = {"from": output("b_end", "epoch_s"),
                                  "until": output("a_start", "epoch_s")}
        windows["post_recovery"] = {"from": output("both_start", "epoch_s")}
        windows["all_requests"] = {}
    for name, boundaries in windows.items():
        case.step(name, "master_client_window", params={"rows": output("finish", "rows"), **boundaries})

    checks = {
        "baseline_success": "baseline",
        "b_success": "b_only",
        "b_route": "b_only",
        "b_balance": "b_only",
        "a_success": "a_only",
        "a_route": "a_only",
        "a_balance": "a_only",
        "both_success": "both",
        "both_balance": "both",
    }
    if restart_mode == "non_rolling":
        checks.update(outage_failures="outage", outage_no_master="outage",
                      outage_terminal="outage")
    else:
        checks.update(handover_errors="a_handover", late_errors="post_recovery",
                      rolling_errors="all_requests")
    for name, window in checks.items():
        params = case.params(f"{root}.checks.{name}", {"rows": output(window, "rows")})
        case.metric(params["metric"])
        case.observe(name, "master_client_check", params=params)
    params = case.params(f"{root}.checks.unique_requests", {"rows": output("finish", "rows")})
    case.metric(params["metric"])
    case.observe("unique_requests", "master_client_check", params=params)
    case.step("clean_a", "master_inflight_clean", params={"target": "A"})
    case.step("clean_b", "master_inflight_clean", params={"target": "B"})
    case.step("cleanup", "teardown")
