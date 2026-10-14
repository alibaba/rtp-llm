"""Load case-selected views; YAML controls presentation, not analysis."""

import re
from pathlib import Path

from case_api.registry import view_capabilities
from reporting.view_schema import (
    validate_default,
    validate_selected, validate_structure, validate_text, validate_panels,
)
from document_io import load_document, DocumentError
from input_contract import fail
from reporting.view_model import ViewDefinition
from reporting.events import validate_events

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
    validate_events(path, data.get("charts", {}), fail)
    validate_text(path, data, fail)
    capability = view_capabilities().get(name)
    try:
        definition = ViewDefinition.read(data, subtitle_fields=capability.subtitle_fields if capability else ())
    except (ValueError, TypeError) as exc:
        fail(path, str(exc))
    if kind == "selected":
        validate_panels(path, definition, fail)
    if capability is not None:
        try:
            capability.validator(path, definition, fail)
        except DocumentError:
            raise
        except (ValueError, TypeError) as exc:
            fail(path, str(exc))
    if "metrics" in data or kind == "selected":
        from monitoring.view_binding import validate_view_metrics

        validate_view_metrics(path, data, definition)
    return definition


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
