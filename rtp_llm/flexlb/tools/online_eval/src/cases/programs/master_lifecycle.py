"""Cold restart of the owned master, clean scheduler state, topology and request recovery."""

from cases.config import output


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
