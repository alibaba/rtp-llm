"""Submitted requests reach a business terminal without stream errors."""

from dataclasses import replace

from scenario.inputs import InputSchema
from scenario.numeric_parameters import COUNT, JAVA_LENGTH, NONNEGATIVE, POSITIVE_COUNT
from traffic.contracts import driver

INPUTS = InputSchema(
    {
        "traffic": {"kind": None, "count": replace(POSITIVE_COUNT, maximum=10000), "input_len": JAVA_LENGTH, "output_len": JAVA_LENGTH},
        "procedure": {"setup_timeout_s": NONNEGATIVE},
        "checks": {"completed": None, "no_errors": {"expected": COUNT}},
    }
)


def default(case):
    """Submitted requests must reach a business terminal without stream errors."""
    data = case.config
    request = {name: data["traffic"][name] for name in driver(data["traffic"], "request_batch")}
    validate_checks(data["checks"])
    case.step("setup", "setup", timeout_s=data["procedure"]["setup_timeout_s"])
    case.step(
        "terminal",
        "request_complete",
        params=dict(request, checks=data["checks"], operation_timeout_s=case.stage_timeout_s),
        timeout_s=2 * case.stage_timeout_s,
    )
    case.step("cleanup", "teardown")


from cases.request_completion.actions import COMPLETE, validate_checks
from scenario.registry import CaseDefinition

CASE = CaseDefinition(builders={"default": default}, input_schema=INPUTS, actions=(COMPLETE,))
