"""Actual dual standalone masters: sticky A to B through same-request failover, with per-request route evidence."""

from ..case_config import output


def standalone_a_to_b(case):
    case.step(
        "setup", "setup", timeout_s=case.value("standalone_a_to_b.setup_timeout_s")
    )
    case.step(
        "flow", "master_client_start", params=case.value("standalone_a_to_b.flow")
    )
    case.step(
        "lookback", "master_mark", params=case.value("standalone_a_to_b.lookback")
    )
    case.step(
        "kill_time", "master_mark", params=case.value("standalone_a_to_b.kill_time")
    )
    case.step("kill_a", "master_fault", params=case.value("standalone_a_to_b.kill_a"))
    case.step(
        "switched", "master_mark", params=case.value("standalone_a_to_b.switched")
    )
    case.step("b_ready", "master_ready", params=case.value("standalone_a_to_b.b_ready"))
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=case.value("standalone_a_to_b.finish_timeout_s"),
        params={"client": output("flow", "client")},
    )
    case.step(
        "steady",
        "master_client_window",
        params=case.params(
            "standalone_a_to_b.steady",
            {
                "rows": output("finish", "rows"),
                "until": output("kill_time", "epoch_s"),
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
        "straddle",
        "master_client_window",
        params={
            "rows": output("finish", "rows"),
            "from": output("lookback", "epoch_s"),
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
        "steady_a",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.steady_a", {"rows": output("steady", "rows")}
        ),
    )
    case.observe(
        "failover_seen",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.failover_seen", {"rows": output("straddle", "rows")}
        ),
    )
    case.observe(
        "switch_to_b",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.switch_to_b", {"rows": output("switch", "rows")}
        ),
    )
    case.observe(
        "switch_errors",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.switch_errors", {"rows": output("switch", "rows")}
        ),
    )
    case.observe(
        "after_b",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.after_b", {"rows": output("after", "rows")}
        ),
    )
    case.observe(
        "after_success",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.after_success", {"rows": output("after", "rows")}
        ),
    )
    case.observe(
        "unique_requests",
        "master_client_check",
        params=case.params(
            "standalone_a_to_b.unique_requests", {"rows": output("finish", "rows")}
        ),
    )
    case.step("cleanup", "teardown")
