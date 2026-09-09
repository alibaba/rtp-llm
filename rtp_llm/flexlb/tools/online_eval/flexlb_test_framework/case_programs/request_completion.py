"""Submitted requests reach a business terminal without stream errors."""

from ..case_config import output


def completion(case, *, deferred=False):
    """Both consumption modes share the same terminal and zero-error contract."""
    request = {
        "input_len": case.number("input_len"),
        "output_len": case.number("output_len"),
        "count": case.number("count"),
    }
    if deferred:
        request["consume"] = case.value("completion.deferred_consume")
    case.step("setup", "setup", timeout_s=case.value("completion.setup_timeout_s"))
    case.step("submit", "request", params=request)
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    for name, field in (
        ("completed", "completed"),
        ("no_errors", "error_count"),
    ):
        case.step(
            name,
            "check",
            params=case.params(
                "completion.step_5",
                {
                    "actual": output("terminal", field),
                    "expected": case.value(f"completion.expected.{name}"),
                },
            ),
        )
    case.step("cleanup", "teardown")


def immediate(case):
    completion(case)


def deferred_fetch(case):
    completion(case, deferred=True)


def client_no_fetch(case):
    shape = dict(
        input_len=case.number("input_len"),
        output_len=case.number("output_len"),
        observe_s=case.number("observe_s"),
    )
    case.step("setup", "setup", timeout_s=case.value("client_no_fetch.setup_timeout_s"))
    for mode in ("late", "missing"):
        case.step(
            mode,
            "client_fetch_probe",
            params=dict(shape, mode=mode),
            timeout_s=case.value("client_no_fetch.step_5_timeout_s"),
        )
    case.step(
        "automatic_environment",
        "environment_reconfigure",
        timeout_s=case.value("client_no_fetch.automatic_environment_timeout_s"),
        params=case.value("client_no_fetch.automatic_environment"),
    )
    case.step(
        "automatic",
        "client_auto_fetch_probe",
        params=dict(shape, mode=case.value("client_no_fetch.automatic.mode")),
        timeout_s=case.value("client_no_fetch.automatic_timeout_s"),
    )
    case.step("cleanup", "teardown")
