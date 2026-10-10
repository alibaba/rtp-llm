"""Submitted requests reach a business terminal without stream errors."""

from cases.config import output
from cases.inputs import fields


def default(case):
    """Submitted requests must reach a business terminal without stream errors."""
    data = case.inputs(traffic={"input_len", "output_len", "count"},
                       procedure={"setup_timeout_s"}, checks={"completed", "no_errors"})
    request = {name: case.number("traffic." + name) for name in data.traffic}
    for name, criterion in data.checks.items():
        fields(criterion, {"op", "expected"}, f"parameters.checks.{name}")
    case.step("setup", "setup", timeout_s=data.procedure["setup_timeout_s"])
    case.step("submit", "request", params=request)
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    for name, field in (
        ("completed", "completed"),
        ("no_errors", "error_count"),
    ):
        case.step(
            name,
            "check",
            params=dict(data.checks[name], actual=output("terminal", field)),
        )
    case.step("cleanup", "teardown")
