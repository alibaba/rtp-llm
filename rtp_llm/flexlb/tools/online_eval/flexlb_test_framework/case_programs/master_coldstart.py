"""Twenty requests with ten concurrent clients immediately after readiness, retaining topology samples and placement evidence."""

from ..case_config import output


def burst_twenty(case):
    case.step("setup", "setup", timeout_s=case.value("burst_twenty.setup_timeout_s"))
    case.step(
        "burst",
        "master_request_batch",
        params=case.value("burst_twenty.burst"),
    )
    case.step(
        "verdict",
        "master_coldstart_check",
        params={"snapshot": output("burst", "snapshot")},
    )
    case.step("cleanup", "teardown")
