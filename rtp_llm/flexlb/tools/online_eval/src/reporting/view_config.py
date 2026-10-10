"""Load case-selected views; YAML controls presentation, not analysis."""

import re
from pathlib import Path

from cases.registry import VIEW_KINDS, load_capability
from reporting.view_schema import (
    validate_bindings, validate_checks, validate_default, validate_monitoring_policy,
    validate_produced, validate_text,
)
from scenario.loader import load_document
from scenario.validation import fail

VIEWS = Path(__file__).resolve().parents[2] / "config/report_views"
DEFAULT_VIEW = "workload.yaml"
CHECKS_VIEW = "execution.yaml"
_BUILTIN_VALIDATORS = {CHECKS_VIEW: validate_checks, DEFAULT_VIEW: validate_default}


def view(name):
    if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
        fail("reports", "view must be a filename under config/report_views")
    path = VIEWS / name
    data = load_document(path)
    validator = _BUILTIN_VALIDATORS.get(name)
    if validator is None:
        kind = data.get("kind")
        capability = VIEW_KINDS.get(kind) if type(kind) is str else None
        validator = load_capability(capability["validator"]) if capability else validate_produced
    validator(path, data, fail)
    query_plan = None
    if "monitoring_query_plan" in data:
        from monitoring.query_plan import load_plan

        query_plan = load_plan(data["monitoring_query_plan"])
    if "curves" in data:
        if query_plan is None:
            fail(path, "curves require monitoring_query_plan")
        validate_bindings(path, data, query_plan, fail)
    if data["kind"] == "produced":
        validate_monitoring_policy(path, data, query_plan, fail)
    validate_text(path, data, fail)
    return data


def declaration(value, *, kind, path="reports"):
    if kind != "workload":
        fail(path, "report views require test.kind=workload")
    if not isinstance(value, list) or not value or any(type(name) is not str for name in value):
        fail(path, "expected a nonempty list of view filenames")
    if len(set(value)) != len(value):
        fail(path, "duplicate view")
    for name in value:
        view(name)
    return list(value)
