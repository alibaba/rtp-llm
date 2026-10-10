"""Discover trusted case directories and validate their explicit capabilities."""

from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Mapping


class ProducerPhase(str, Enum):
    GATE = 'gate'
    FINALIZE = 'finalize'


@dataclass(frozen=True)
class ReportView:
    validator: Callable
    renderer: Callable


@dataclass(frozen=True)
class MetricProducer:
    describe: Callable
    execute: Callable
    phase: ProducerPhase

    @property
    def module(self):
        return self.describe.__module__


@dataclass(frozen=True)
class EvidenceSource:
    factory: Callable
    fields: frozenset[str]
    label: str
    required_fields: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CaseDefinition:
    builders: Mapping[str, Callable]
    numeric_parameters: Mapping = field(default_factory=dict)
    actions: tuple = ()
    report_views: Mapping[str, ReportView] = field(default_factory=dict)
    producers: Mapping[str, MetricProducer] = field(default_factory=dict)
    sources: Mapping[str, EvidenceSource] = field(default_factory=dict)

    def __post_init__(self):
        for name in ('builders', 'numeric_parameters', 'report_views', 'producers', 'sources'):
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

    @classmethod
    def build(cls, cases):
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
            entries[case.name] = case
            for name, capability in definition.report_views.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*\.yaml', name) or name == 'default.yaml'
                        or not isinstance(capability, ReportView)
                        or not callable(capability.validator) or not callable(capability.renderer)):
                    raise ValueError('invalid registered view capability: ' + name)
                add(views, name, capability, 'view')
            for name, capability in definition.producers.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                        or not isinstance(capability, MetricProducer)
                        or not isinstance(capability.phase, ProducerPhase)
                        or not callable(capability.describe) or not callable(capability.execute)):
                    raise ValueError('invalid registered producer capability: ' + name)
                add(producers, name, capability, 'producer')
            for name, capability in definition.sources.items():
                if (not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                        or not isinstance(capability, EvidenceSource) or not callable(capability.factory)
                        or not capability.required_fields <= capability.fields):
                    raise ValueError('invalid registered source capability: ' + name)
                add(sources, name, capability, 'source')
        return cls(*(MappingProxyType(v) for v in (entries, views, producers, sources)))


def discover(root=None, *, package='cases'):
    """Only immediate, owned package directories containing program.py are cases."""
    root = Path(root) if root is not None else Path(__file__).parent
    entries = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / 'program.py').is_file():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', path.name):
                raise ValueError('invalid case directory: ' + path.name)
            module_name = package + '.' + path.name + '.program'
            module = import_module(module_name)
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


def produce_gate_metrics(program, directory):
    case = registry().cases[program]
    for producer in case.definition.producers.values():
        if producer.phase is ProducerPhase.GATE:
            producer.execute(directory)
