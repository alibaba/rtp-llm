"""Load case-selected views; YAML controls presentation, not analysis."""

import re
from pathlib import Path

from cases.registry import view_capabilities
from reporting.view_schema import (
    validate_bindings, validate_default, validate_monitoring_policy,
    validate_selected, validate_structure, validate_text,
)
from scenario.loader import load_document
from scenario.validation import fail

VIEWS = Path(__file__).resolve().parents[2] / "config/report_views"
DEFAULT_VIEW = "default.yaml"
_VALIDATORS = {"default": validate_default, "selected": validate_selected}


def view(name):
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
        fail("reports", "view must be a filename under config/report_views")
    path = VIEWS / name
    data = load_document(path)
    validate_structure(path, data, fail)
    kind = data.get("kind")
    validator = _VALIDATORS.get(kind) if type(kind) is str else None
    if validator is None:
        fail(path, "kind must be default or selected")
    if (name == DEFAULT_VIEW) != (kind == "default"):
        fail(path, "default kind is reserved for default.yaml")
    validator(path, data, fail)
    from reporting.events import validate_events
    validate_events(path, data.get("charts", {}), fail)
    query_plan = None
    capability = view_capabilities().get(name)
    if capability is not None:
        capability.validator(path, data, fail)
    if "metrics" in data:
        from monitoring.query_plan import load_plan

        query_plan = load_plan(data["metrics"]["query_plan"])
    if "curves" in data.get("charts", {}):
        if query_plan is None:
            fail(path, "curves require metrics.query_plan")
        validate_bindings(path, data, query_plan, fail)
    if kind == "selected":
        validate_monitoring_policy(path, data, query_plan, fail)
    validate_text(path, data, fail)
    return data


def declaration(value, *, kind, path="reports"):
    if kind != "workload":
        fail(path, "report views require metadata.kind=workload")
    if not isinstance(value, list) or not value or any(type(name) is not str for name in value):
        fail(path, "expected a nonempty list of view filenames")
    if len(set(value)) != len(value):
        fail(path, "duplicate view")
    for name in value:
        view(name)
    return list(value)
