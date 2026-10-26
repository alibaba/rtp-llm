"""Minimal plan stubs for isolated actions; runner tests use compiled plans."""

from copy import deepcopy
from types import SimpleNamespace

from scenario.context import RuntimeContext
from scenario.stage_parameters import AdapterParameters


def plan(value=None):
    value = deepcopy(value or {})
    execution = value.get("execution", {})
    monitoring = value.get("workload_runtime", execution.get("monitoring", {}))
    return SimpleNamespace(
        **{
            key: child
            for key, child in value.items()
            if key not in {"environment", "resource_budget", "execution", "metadata", "implementation", "reporting", "reports"}
        },
        environment=SimpleNamespace(to_dict=lambda: deepcopy(value.get("environment", {}))),
        resource_budget=SimpleNamespace(to_metadata=lambda: value.setdefault("resource_budget", {})),
        execution=SimpleNamespace(
            collection=execution.get("collection", value.get("collection_profile")),
            monitoring=SimpleNamespace(to_dict=lambda: deepcopy(monitoring)),
        ),
        metadata=AdapterParameters(value.get("metadata", {"kind": value.get("test_kind", "functional")})),
        implementation=AdapterParameters(value["implementation"]) if "implementation" in value else None,
        reporting=AdapterParameters(value["reporting"]) if "reporting" in value else None,
        reports=value.get("reports"),
    )


def runtime_context(instance, *args, **kwargs):
    return RuntimeContext(plan(instance), *args, **kwargs)
