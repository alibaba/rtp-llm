"""Registered Python case programs. YAML can only select entries in this registry."""

from dataclasses import dataclass
from typing import Callable


PROGRAMS = {
    "cache_scale_in": "cases.cache_scale_in.program",
    "master_ha_failover": "cases.master_ha_failover.program",
    "master_performance": "cases.master_performance.program",
    "request_completion": "cases.request_completion.program",
}


@dataclass(frozen=True)
class ReportView:
    validator: Callable
    renderer: Callable


def view_capabilities():
    """Collect declarations from trusted programs; never import YAML-supplied paths."""
    from importlib import import_module
    import re

    views = {}
    for path in dict.fromkeys(PROGRAMS.values()):
        declared = getattr(import_module(path), "REPORT_VIEWS", {})
        if not isinstance(declared, dict):
            raise ValueError("REPORT_VIEWS must be a mapping: " + path)
        for name, capability in declared.items():
            if type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name):
                raise ValueError("invalid registered view filename: " + str(name))
            if name == "default.yaml":
                raise ValueError("reserved registered view: " + name)
            if (not isinstance(capability, ReportView) or not callable(capability.validator)
                    or not callable(capability.renderer)):
                raise ValueError("invalid registered view capability: " + name)
            if name in views and views[name] != capability:
                raise ValueError("conflicting registered view capability: " + name)
            views[name] = capability
    return views


def produce_gate_metrics(program, directory):
    """Project frozen gate values after telemetry export, before presentation."""
    from importlib import import_module
    producer = getattr(import_module(PROGRAMS[program]), "produce_gate_metrics", None)
    if producer is not None:
        if not callable(producer):
            raise ValueError("produce_gate_metrics must be callable")
        producer(directory)
