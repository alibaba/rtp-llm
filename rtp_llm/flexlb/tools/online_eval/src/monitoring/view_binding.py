"""Bind report views to declared monitoring metrics at the integration boundary."""
from monitoring import query_plan
from input_contract import fail

def validate_bindings(path, data, plan, fail):
    declared = query_plan.definitions(plan)
    for curve_id, binding in data.charts.curves.items():
        if binding.metric_id not in declared or binding.labels is None:
            fail(path, "curve must bind a declared metric id and labels: " + curve_id)


def validate_monitoring_policy(path, data, plan, fail):
    policy = data["metrics"]
    if "diagnostic_only" not in policy:
        fail(path, "selected views must classify diagnostic_only metrics explicitly")
    metrics = data["charts"]["curves"]
    if any(
        not {"unit", "color", "hidden"} <= set(style)
        for style in metrics.values()
    ):
        fail(path, "classified monitoring styles require unit, color and hidden")
    diagnostic = policy.get("diagnostic_only", [])

    known = {identity: definition for identity, definition in query_plan.definitions(plan).items()
             if not ("promql" in definition and identity.split('/')[-1] == 'up')}
    if not isinstance(diagnostic, list) or any(
        type(identity) is not str or identity not in known for identity in diagnostic
    ) or len(diagnostic) != len(set(diagnostic)):
        fail(path, "invalid diagnostic_only metrics")


def validate_view_metrics(path, data, definition):
    plan = query_plan.load_plan(data["metrics"]["query_plan"]) if "metrics" in data else None
    if "curves" in data.get("charts", {}):
        if plan is None:
            fail(path, "curves require metrics.query_plan")
        validate_bindings(path, definition, plan, fail)
    if definition.kind == "selected":
        validate_monitoring_policy(path, data, plan, fail)
