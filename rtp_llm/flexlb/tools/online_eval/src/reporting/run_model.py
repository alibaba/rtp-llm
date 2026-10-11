"""Explicit run presentation data; dictionaries are decoded at this boundary."""
from dataclasses import dataclass
import copy
from typing import Optional


def _object(value, path):
    if not isinstance(value, dict):
        raise ValueError(path + ' must be an object')
    return copy.deepcopy(value)


@dataclass(frozen=True)
class CheckNode:
    identity: str
    status: str
    actual: object
    expected: object
    children: tuple['CheckNode', ...] = ()

    @classmethod
    def read(cls, data, prefix):
        data = _object(data, 'check')
        identity = prefix + '/' + data['id']
        evidence = _object(data.get('evidence', {}), 'check.evidence')
        children = evidence.get('checks', [])
        if not isinstance(children, list):
            raise ValueError('check children must be an array')
        return cls(identity, data['status'], data.get('actual'), data.get('expected'),
                   tuple(cls.read(child, identity) for child in children))

    def rows(self):
        yield [self.identity, self.status, self.actual, self.expected]
        for child in self.children:
            yield from child.rows()


@dataclass(frozen=True)
class RunValidity:
    runtime_validity: Optional[str]
    telemetry_completeness: object
    missing_telemetry: object
    telemetry_integrity_errors: object
    telemetry_diagnostics: object
    telemetry_warnings: object

    @classmethod
    def read(cls, workload):
        return cls(*(workload.get(name) for name in cls.__dataclass_fields__))

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class RunPresentation:
    identity: str
    status: Optional[str]
    validity: RunValidity
    checks: tuple[CheckNode, ...]
    configuration: object
    configuration_sha256: Optional[str]
    runtime_configuration: object
    implementation: dict
    traffic_manifests: object
    runtime_provenance: object
    clock_anchor: object
    report_timeline: Optional[dict]
    request_sources: object
    phases: tuple[dict, ...]
    events: tuple[dict, ...]

    @classmethod
    def read(cls, analysis):
        analysis = _object(analysis, 'run analysis')
        identity = analysis['id']
        if type(identity) is not str or not identity:
            raise ValueError('run identity must be a nonempty string')
        workload = _object(analysis.get('workload', {}), 'run.workload')
        checks = analysis.get('checks', [])
        if not isinstance(checks, list):
            raise ValueError('run checks must be an array')
        return cls(identity, analysis.get('status'), RunValidity.read(workload),
            tuple(CheckNode.read(check, check['stage']) for check in checks),
            analysis.get('configuration'), analysis.get('configuration_sha256'),
            workload.get('runtime_configuration'), _object(analysis.get('implementation', {}), 'run.implementation'),
            analysis.get('traffic_manifests'), analysis.get('runtime_provenance'),
            analysis.get('clock_anchor'), analysis.get('report_timeline'), analysis.get('request_sources'),
            tuple(analysis.get('phases', [])), tuple(analysis.get('events', [])))
