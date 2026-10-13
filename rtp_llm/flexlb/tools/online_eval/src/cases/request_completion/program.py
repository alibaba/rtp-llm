"""Submitted requests reach a business terminal without stream errors."""

from dataclasses import replace

from case_api.builder import output
from case_api.inputs import fields
from case_api.windows import check_windows
from traffic.contracts import driver
from case_api.numeric_parameters import COUNT
from case_api.numeric_parameters import NONNEGATIVE
from case_api.numeric_parameters import POSITIVE_COUNT
from case_api.numeric_parameters import JAVA_LENGTH



from case_api.input_contract import InputContract

INPUTS = InputContract.from_fields({
    'traffic': {'kind': None, 'count': replace(POSITIVE_COUNT, maximum=10000),
                'input_len': JAVA_LENGTH, 'output_len': JAVA_LENGTH},
    'procedure': {'setup_timeout_s': NONNEGATIVE},
    'checks': {'completed': None, 'no_errors': {'expected': COUNT}},
})


def default(case):
    """Submitted requests must reach a business terminal without stream errors."""
    data = case.bind(INPUTS)
    request = {name: case.number("traffic." + name) for name in driver(data.traffic, "request_batch")}
    for name, criterion in data.checks.items():
        fields(criterion, {"windows", "output", "unit", "op", "expected"}, f"parameters.checks.{name}")
        check_windows(criterion["windows"], available={"terminal"},
                      path=f"parameters.checks.{name}.windows", single=True)
        field = criterion["output"]
        unit = {"completed": "boolean", "error_count": "errors"}.get(field) if type(field) is str else None
        if unit is None or criterion["unit"] != unit:
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
                        actual=output(criterion["windows"][0], criterion["output"])),
        )
    case.step("cleanup", "teardown")


from case_api.registry import CaseDefinition

CASE = CaseDefinition(
    builders={"default": default},
    input_contract=INPUTS,
)
