"""Preserve terminal-window client records and independent mock source snapshots."""

from ..case_config import output


def deferred(case):
    case.step("setup", "setup", timeout_s=case.value("deferred.setup_timeout_s"))
    case.step(
        "baseline",
        "snapshot",
        params=case.value("deferred.baseline"),
    )
    case.step("submit", "request", params=case.value("deferred.submit"))
    case.step(
        "observing",
        "observe",
        params=case.params(
            "deferred.observing", {"requests": output("submit", "requests")}
        ),
    )
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    case.step(
        "frozen",
        "observe",
        params=case.params(
            "deferred.frozen", {"observation": output("observing", "observation")}
        ),
    )
    case.step(
        "completed",
        "check",
        params=case.params(
            "deferred.completed", {"actual": output("terminal", "completed")}
        ),
    )
    case.step(
        "no_errors",
        "check",
        params=case.params(
            "deferred.no_errors", {"actual": output("terminal", "error_count")}
        ),
    )
    case.step("cleanup", "teardown")
