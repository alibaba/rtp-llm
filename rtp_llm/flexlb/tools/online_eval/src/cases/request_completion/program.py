"""Submitted requests reach a business terminal without stream errors."""

from cases.config import output


def default(case):
    """Submitted requests must reach a business terminal without stream errors."""
    request = {
        "input_len": case.number("input_len"),
        "output_len": case.number("output_len"),
        "count": case.number("count"),
    }
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
