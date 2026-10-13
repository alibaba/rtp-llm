"""Discover trusted case directories and validate their explicit capabilities."""

from __future__ import annotations

from case_api.extension_contracts import SnapshotFactory, RuntimeClientFactory, EvidenceExporter

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping

from case_api.extension_contracts import ProgramBuilder
from case_api.extension_contracts import ViewValidator
from case_api.extension_contracts import ViewRenderer
from case_api.extension_contracts import ProducerDescription
from case_api.extension_contracts import GateProducer
from case_api.extension_contracts import FinalizeProducer
from case_api.extension_contracts import SourceFactory
from case_api.extension_contracts import validate_call


class ProducerPhase(str, Enum):
    GATE = 'gate'
    FINALIZE = 'finalize'


@dataclass(frozen=True)
class ReportView:
    validator: ViewValidator
    renderer: ViewRenderer
    subtitle_fields: frozenset[str] = frozenset()
    deadline_renderer: object = None

    def __post_init__(self):
        if not isinstance(self.subtitle_fields, (set, frozenset, tuple, list)):
            raise ValueError("subtitle_fields must be a collection of field names")
        fields = frozenset(self.subtitle_fields)
        if any(type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", name) for name in fields):
            raise ValueError("invalid registered subtitle field")
        object.__setattr__(self, "subtitle_fields", fields)


@dataclass(frozen=True)
class MetricProducer:
    describe: ProducerDescription
    execute: GateProducer | FinalizeProducer
    phase: ProducerPhase

    @property
    def module(self):
        return self.describe.__module__


@dataclass(frozen=True)
class EvidenceSource:
    factory: SourceFactory
    fields: frozenset[str]
    label: str
    required_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class SnapshotSource:
    factory: SnapshotFactory


@dataclass(frozen=True)
class ClientFactory:
    factory: RuntimeClientFactory
    evidence: EvidenceExporter | None = None


@dataclass(frozen=True)
class CaseDefinition:
    builders: Mapping[str, ProgramBuilder]
    numeric_parameters: Mapping = field(default_factory=dict)
    actions: tuple = ()
    report_views: Mapping[str, ReportView] = field(default_factory=dict)
    producers: Mapping[str, MetricProducer] = field(default_factory=dict)
    sources: Mapping[str, EvidenceSource] = field(default_factory=dict)

    input_contract: object = None
    snapshot_sources: Mapping = field(default_factory=dict)
    clients: Mapping = field(default_factory=dict)
    gates: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.input_contract is not None:
            from case_api.input_contract import InputContract
            if not isinstance(self.input_contract, InputContract):
                raise ValueError("input_contract must be an InputContract")
            merged = dict(self.numeric_parameters)
            for path, rule in self.input_contract.numeric_parameters.items():
                if path in merged and merged[path] != rule:
                    raise ValueError("conflicting numeric input: " + path)
                merged[path] = rule
            object.__setattr__(self, "numeric_parameters", merged)
        for name in ('builders', 'numeric_parameters', 'report_views', 'producers', 'sources', 'snapshot_sources', 'clients', 'gates'):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError('case ' + name + ' must be a mapping')
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, 'actions', tuple(self.actions))


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
    producers: Mapping[str, MetricProducer]
    sources: Mapping[str, EvidenceSource]

    def capability(self, kind, name, *, case):
        """A case may consume only capabilities it explicitly declares, including shared ones."""
        fields = {'view': 'report_views', 'producer': 'producers', 'source': 'sources', 'snapshot': 'snapshot_sources', 'client': 'clients', 'gate': 'gates'}
        if kind not in fields:
            raise ValueError('unknown capability kind: ' + kind)
        if case not in self.cases:
            raise ValueError('unknown case: ' + case)
        owned = getattr(self.cases[case].definition, fields[kind])
        if name not in owned:
            raise ValueError(f'{kind} {name!r} is not declared by case {case!r}')
        return owned[name]

    @classmethod
    def build(cls, cases):
        from scenario.contracts import StageHandler

        entries, views, producers, sources = {}, {}, {}, {}
        def add(target, name, capability, kind):
            if name in target and target[name] != capability:
                raise ValueError('conflicting registered ' + kind + ' capability: ' + name)
            target[name] = capability
        for case in cases:
            if not re.fullmatch(r'[a-z][a-z0-9_]*', case.name) or case.name in entries:
                raise ValueError('invalid or duplicate case: ' + case.name)
            definition = case.definition
            if not isinstance(definition, CaseDefinition):
                raise ValueError(case.module + ': CASE must be a CaseDefinition')
            if 'default' not in definition.builders or any(
                not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not callable(build)
                for name, build in definition.builders.items()
            ):
                raise ValueError(case.module + ': invalid case builders')
            for name, builder in definition.builders.items():
                validate_call(builder, 1, case.module + '.builders.' + name)
            for action in definition.actions:
                if not isinstance(action, StageHandler):
                    raise ValueError(case.module + ': actions must contain StageHandler capabilities')
                validate_call(action.validate, 2, case.module + '.actions.' + action.name + '.validate')
                validate_call(action.execute, 3, case.module + '.actions.' + action.name + '.execute')
            for collection, expected, count in (("snapshot_sources", SnapshotSource, 2), ("clients", ClientFactory, 3)):
                for name, capability in getattr(definition, collection).items():
                    if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not isinstance(capability, expected):
                        raise ValueError("invalid " + collection + " capability: " + name)
                    validate_call(capability.factory, count, case.module + "." + collection + "." + name)
            for name, capability in definition.clients.items():
                if capability.evidence is not None:
                    validate_call(capability.evidence, 3, case.module + ".clients." + name + ".evidence")
            from case_api.gate_plugin import GatePlugin
            for name, capability in definition.gates.items():
                if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not isinstance(capability, GatePlugin):
                    raise ValueError("invalid gate capability: " + name)
            entries[case.name] = case
            for name, capability in definition.report_views.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*\.yaml', name) or name == 'default.yaml'
                        or not isinstance(capability, ReportView)
                        or not callable(capability.validator) or not callable(capability.renderer)):
                    raise ValueError('invalid registered view capability: ' + name)
                validate_call(capability.validator, 3, case.module + '.views.' + name + '.validator')
                validate_call(capability.renderer, 3, case.module + '.views.' + name + '.renderer')
                if capability.deadline_renderer is not None:
                    validate_call(capability.deadline_renderer, 4, case.module + ".views." + name + ".deadline_renderer")
                add(views, name, capability, 'view')
            for name, capability in definition.producers.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                        or not isinstance(capability, MetricProducer)
                        or not isinstance(capability.phase, ProducerPhase)
                        or not callable(capability.describe) or not callable(capability.execute)):
                    raise ValueError('invalid registered producer capability: ' + name)
                validate_call(capability.describe, 3, case.module + '.producers.' + name + '.describe')
                count = 1 if capability.phase is ProducerPhase.GATE else 2
                validate_call(capability.execute, count, case.module + '.producers.' + name + '.execute')
                add(producers, name, capability, 'producer')
            for name, capability in definition.sources.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                        or not isinstance(capability, EvidenceSource) or not callable(capability.factory)
                        or not capability.required_fields <= capability.fields):
                    raise ValueError('invalid registered source capability: ' + name)
                validate_call(capability.factory, 2, case.module + '.sources.' + name + '.factory')
                add(sources, name, capability, 'source')
        return cls(*(MappingProxyType(v) for v in (entries, views, producers, sources)))


def discover(root=None, *, package='cases'):
    """Only immediate, owned package directories containing program.py are cases."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent / "cases"
    entries = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / 'program.py').is_file():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', path.name):
                raise ValueError('invalid case directory: ' + path.name)
            module_name = package + '.' + path.name + '.program'
            try:
                module = import_module(module_name)
            except Exception as exc:
                raise ValueError(f'{module_name}: case import failed: {exc}') from exc
            entries.append(RegisteredCase(path.name, module_name, Path(module.__file__),
                                          getattr(module, 'CASE', None)))
    return CaseRegistry.build(entries)


# Discovery is lazy to avoid importing cases while their contracts are loading.
# A snapshot is immutable; a new process/explicit refresh sees new case folders.
_snapshot = None


def registry():
    global _snapshot
    if _snapshot is None:
        _snapshot = discover()
    return _snapshot


def view_capabilities():
    return registry().views
