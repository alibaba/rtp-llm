"""一次执行请求批次，并检查业务终态；请求所有权仍由 backend 管理。"""

from analysis.checks import evaluate, validate_comparison
from scenario.contracts import StageHandler, StageOutput
from scenario.request_parameters import normalize_request_parameters
from scenario.validation import number


def validate_checks(checks):
    from scenario.inputs import fields
    from scenario.windows import check_windows

    fields(checks, {"completed", "no_errors"}, "completion checks")
    for name, criterion in checks.items():
        fields(criterion, {"windows", "output", "unit", "op", "expected"}, f"parameters.checks.{name}")
        check_windows(criterion["windows"], available={"terminal"}, path=f"parameters.checks.{name}.windows", single=True)
        field = criterion["output"]
        unit = {"completed": "boolean", "error_count": "errors"}.get(field) if type(field) is str else None
        if unit is None or criterion["unit"] != unit:
            raise ValueError("completion checks require declared terminal outputs and their units")
        if unit == "boolean" and type(criterion["expected"]) is not bool or unit == "errors" and type(criterion["expected"]) is not int:
            raise ValueError("completion threshold type does not match the output")
        validate_comparison(criterion["op"], criterion["expected"])


def validate(params, plan):
    params = dict(params)
    checks = params.pop("checks")
    budget = number(params.pop("operation_timeout_s"), plan.path, minimum=0.001)
    normalize_request_parameters(params, plan.path)
    validate_checks(checks)
    if params["consume"] == "deferred":
        for profile in plan.profiles:
            if "enqueue_batch" not in plan.env.render(profile)["effective_capabilities"]:
                raise ValueError("deferred consumption requires enqueue_batch")
    return dict(params, checks=checks, operation_timeout_s=budget)


def execute(ctx, params, deadline):
    request = {k: v for k, v in params.items() if k not in {"checks", "operation_timeout_s"}}
    with deadline.operation(params["operation_timeout_s"], ctx.enforce_deadlines) as operation:
        handle = ctx.backend.start_requests(ctx, request, operation)
    requests = ctx.resource(handle, "requests")
    with deadline.operation(params["operation_timeout_s"], ctx.enforce_deadlines) as operation:
        terminal = ctx.backend.wait_requests(ctx, requests, operation)
    return StageOutput(terminal, [evaluate(name, terminal[c["output"]], c["op"], c["expected"]) for name, c in params["checks"].items()])


COMPLETE = StageHandler(
    "request_complete", validate, execute, {"completed": "boolean", "error_count": "integer"}, checks=frozenset({"completed", "no_errors"})
)
