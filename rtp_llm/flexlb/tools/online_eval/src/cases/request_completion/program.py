"""Submitted requests reach a business terminal without stream errors."""

from cases.config import output
from cases.inputs import fields
from traffic.contracts import driver
from cases.numeric_parameters import (
    COUNT, NONNEGATIVE, POSITIVE_COUNT, JAVA_LENGTH,
    number_fields,
)


NUMERIC_PARAMETERS = {
    **number_fields(COUNT,
        'checks.no_errors.expected',
    ),
    **number_fields(NONNEGATIVE,
        'procedure.setup_timeout_s',
    ),
    **number_fields(POSITIVE_COUNT,
        'traffic.count',
    ),
    **number_fields(JAVA_LENGTH,
        'traffic.input_len',
        'traffic.output_len',
    ),
}


def default(case):
    """Submitted requests must reach a business terminal without stream errors."""
    data = case.inputs(traffic={"kind", "input_len", "output_len", "count"},
                       procedure={"setup_timeout_s"}, checks={"completed", "no_errors"})
    request = {name: case.number("traffic." + name) for name in driver(data.traffic, "request_batch")}
    for name, criterion in data.checks.items():
        fields(criterion, {"window", "output", "unit", "op", "expected"}, f"parameters.checks.{name}")
        field = criterion["output"]
        unit = {"completed": "boolean", "error_count": "errors"}.get(field) if type(field) is str else None
        if criterion["window"] != "terminal" or unit is None or criterion["unit"] != unit:
            raise ValueError("completion checks require declared terminal outputs and their units")
        if (unit == "boolean" and type(criterion["expected"]) is not bool
                or unit == "errors" and type(criterion["expected"]) is not int):
            raise ValueError("completion threshold type does not match the output")
    case.step("setup", "setup", timeout_s=data.procedure["setup_timeout_s"])
    case.step("submit", "request", params=request)
    case.step("terminal", "wait", params={"requests": output("submit", "requests")})
    for name, criterion in data.checks.items():
        case.step(
            name,
            "check",
            params=dict(op=criterion["op"], expected=criterion["expected"],
                        actual=output(criterion["window"], criterion["output"])),
        )
    case.step("cleanup", "teardown")
