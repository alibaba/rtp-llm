"""Discover trusted case directories and validate their explicit capabilities."""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from reporting.view_model import ViewDefinition
    from scenario.inputs import InputSchema

from input_contract import validate_call
from monitoring.sources import EvidenceSource
from scenario.contracts import ExecutionRequirements, ProgramBuilder, ViewRenderer, ViewValidator


@dataclass(frozen=True)
class ReportView:
    validator: ViewValidator
    renderer: ViewRenderer
    subtitle_fields: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.subtitle_fields, (set, frozenset, tuple, list)):
            raise ValueError("subtitle_fields must be a collection of field names")
        fields = frozenset(self.subtitle_fields)
        if any(type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in fields):
            raise ValueError("invalid registered subtitle field")
        object.__setattr__(self, "subtitle_fields", fields)


@dataclass(frozen=True)
class CaseDefinition:
    builders: Mapping[str, ProgramBuilder]
    actions: tuple = ()
    report_views: Mapping[str, ReportView] = field(default_factory=dict)
    metrics: ModuleType | None = None
    sources: Mapping[str, EvidenceSource] = field(default_factory=dict)

    execution_requirements: ExecutionRequirements = field(default_factory=ExecutionRequirements)
    input_schema: InputSchema | None = None

    def __post_init__(self):
        if not isinstance(self.execution_requirements, ExecutionRequirements):
            raise ValueError("execution_requirements must be an ExecutionRequirements")
        if self.input_schema is not None:
            from scenario.inputs import InputSchema

            if not isinstance(self.input_schema, InputSchema):
                raise ValueError("input_schema must be an InputSchema")
        for name in ("builders", "report_views", "sources"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError("case " + name + " must be a mapping")
        for name in ("builders", "report_views", "sources"):
            value = getattr(self, name)
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "actions", tuple(self.actions))


@dataclass(frozen=True)
class RegisteredCase:
    name: str
    module: str
    path: Path
    definition: CaseDefinition


@dataclass(frozen=True)
class CaseRegistry:
    cases: Mapping[str, RegisteredCase]
    views: Mapping[str, ReportView]
    metric_contracts: Mapping[str, Callable[[str, str, dict | None], dict]]
    sources: Mapping[str, EvidenceSource]

    def capability(self, kind, name, *, case):
        """A case may consume only capabilities it explicitly declares, including shared ones."""
        fields = {"view": "report_views", "source": "sources"}
        if kind not in fields:
            raise ValueError("unknown capability kind: " + kind)
        if case not in self.cases:
            raise ValueError("unknown case: " + case)
        owned = getattr(self.cases[case].definition, fields[kind])
        if name not in owned:
            raise ValueError(f"{kind} {name!r} is not declared by case {case!r}")
        return owned[name]

    @classmethod
    def build(cls, cases):
        from scenario.contracts import StageHandler

        entries, views, contracts, sources = {}, {}, {}, {}

        def add(target, name, capability, kind):
            if name in target and target[name] != capability:
                raise ValueError("conflicting registered " + kind + " capability: " + name)
            target[name] = capability

        for case in cases:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", case.name) or case.name in entries:
                raise ValueError("invalid or duplicate case: " + case.name)
            definition = case.definition
            if not isinstance(definition, CaseDefinition):
                raise ValueError(case.module + ": CASE must be a CaseDefinition")
            if "default" not in definition.builders or any(
                not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not callable(build) for name, build in definition.builders.items()
            ):
                raise ValueError(case.module + ": invalid case builders")
            for name, builder in definition.builders.items():
                validate_call(builder, 1, case.module + ".builders." + name)
            for action in definition.actions:
                if not isinstance(action, StageHandler):
                    raise ValueError(case.module + ": actions must contain StageHandler capabilities")
                validate_call(action.validate, 2, case.module + ".actions." + action.name + ".validate")
                validate_call(action.execute, 3, case.module + ".actions." + action.name + ".execute")
            entries[case.name] = case
            for name, capability in definition.report_views.items():
                if (
                    not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", name)
                    or name == "default.yaml"
                    or not isinstance(capability, ReportView)
                    or not callable(capability.validator)
                    or not callable(capability.renderer)
                ):
                    raise ValueError("invalid registered view capability: " + name)
                validate_call(capability.validator, 3, case.module + ".views." + name + ".validator")
                validate_call(capability.renderer, 4, case.module + ".views." + name + ".renderer")
                add(views, name, capability, "view")
            if definition.metrics is not None:
                declared = getattr(definition.metrics, "CONTRACTS", None)
                if not isinstance(declared, Mapping):
                    raise ValueError(case.module + ": metric contracts must be a mapping")
                for name, describe in declared.items():
                    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                        raise ValueError("invalid metric source: " + str(name))
                    validate_call(describe, 3, case.module + ".metrics." + name)
                    add(contracts, name, describe, "metric source")
                validate_call(getattr(definition.metrics, "project", None), 3, case.module + ".metrics.project")
                if hasattr(definition.metrics, "finalize"):
                    validate_call(definition.metrics.finalize, 4, case.module + ".metrics.finalize")
            for name, capability in definition.sources.items():
                if (
                    not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                    or not isinstance(capability, EvidenceSource)
                    or not callable(capability.factory)
                    or not capability.required_fields <= capability.fields
                ):
                    raise ValueError("invalid registered source capability: " + name)
                validate_call(capability.factory, 2, case.module + ".sources." + name + ".factory")
                add(sources, name, capability, "source")
        return cls(*(MappingProxyType(v) for v in (entries, views, contracts, sources)))


def discover(root=None, *, package="cases", names=None):
    """Only immediate, owned package directories containing program.py are cases."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent / "cases"
    selected = None if names is None else frozenset(names)
    entries = []
    for path in sorted(root.iterdir()):
        if selected is not None and path.name not in selected:
            continue
        if path.is_dir() and (path / "program.py").is_file():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", path.name):
                raise ValueError("invalid case directory: " + path.name)
            module_name = package + "." + path.name + ".program"
            try:
                module = import_module(module_name)
            except Exception as exc:
                raise ValueError(f"{module_name}: case import failed: {exc}") from exc
            entries.append(RegisteredCase(path.name, module_name, Path(module.__file__), getattr(module, "CASE", None)))
    if selected is not None and selected != {entry.name for entry in entries}:
        raise ValueError("unknown selected cases: " + str(sorted(selected - {entry.name for entry in entries})))
    return CaseRegistry.build(entries)


# Discovery is lazy to avoid importing cases while their contracts are loading.
# A snapshot is immutable; a new process/explicit refresh sees new case folders.
_snapshot = None


def registry() -> CaseRegistry:
    global _snapshot
    if _snapshot is None:
        _snapshot = discover()
    return _snapshot


def view_capabilities():
    return registry().views


@contextmanager
def using_cases(names):
    """A frozen child imports only the cases admitted by its execution plan."""
    if names is None:
        yield
        return
    global _snapshot
    names = frozenset(names)
    available = _snapshot
    selected = (
        CaseRegistry.build([available.cases[name] for name in sorted(names)])
        if available is not None and names <= available.cases.keys()
        else discover(names=names)
    )
    _snapshot = selected
    try:
        yield
    finally:
        _snapshot = available


def load_metric_plan(name: str, *, capabilities: CaseRegistry | None = None) -> dict:
    """Bind a declarative plan to the currently admitted case capabilities."""
    # Bind after case discovery; case programs import this composition module.
    from monitoring.query_plan import load_plan

    current = registry() if capabilities is None else capabilities
    return load_plan(name, contracts=current.metric_contracts, sources=current.sources)


def view(name: str, *, capabilities: CaseRegistry | None = None) -> ViewDefinition:
    """Compose case-owned validation with the generic presentation reader."""
    from functools import partial

    from reporting.view_config import view as read_view

    current = registry() if capabilities is None else capabilities
    return read_view(name, capability=current.views.get(name), plan_loader=partial(load_metric_plan, capabilities=current))


def _selected_views(names):
    current = registry()
    return {name: (view(name, capabilities=current), current.views.get(name)) for name in names}


def write_views(directory, analysis, names=None, *, deadline=None, selected=None):
    """Compose selected case views before entering the generic delivery layer."""
    from reporting.workload import write_views as render

    names = names or ["default.yaml"]
    return render(directory, analysis, names, deadline=deadline, selected=_selected_views(names) if selected is None else selected)


def report_declaration(value, *, kind, path="reports"):
    from reporting.view_config import declaration

    return declaration(value, kind=kind, path=path, view_loader=view)
