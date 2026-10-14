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
        if type(data.get('id')) is not str or not data['id']:
            raise ValueError('check identity must be a nonempty string')
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
        return cls(runtime_validity=workload.get('runtime_validity'),
                   telemetry_completeness=workload.get('telemetry_completeness'),
                   missing_telemetry=workload.get('missing_telemetry'),
                   telemetry_integrity_errors=workload.get('telemetry_integrity_errors'),
                   telemetry_diagnostics=workload.get('telemetry_diagnostics'),
                   telemetry_warnings=workload.get('telemetry_warnings'))

    def to_dict(self):
        return dict(runtime_validity=self.runtime_validity, telemetry_completeness=self.telemetry_completeness,
                    missing_telemetry=self.missing_telemetry, telemetry_integrity_errors=self.telemetry_integrity_errors,
                    telemetry_diagnostics=self.telemetry_diagnostics, telemetry_warnings=self.telemetry_warnings)


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
        # Copy only presentation fields; archived time series are read elsewhere.
        if not isinstance(analysis, dict):
            raise ValueError('run analysis must be an object')
        fields = set(cls.__dataclass_fields__) | {'id', 'workload'}
        analysis = copy.deepcopy({key: value for key, value in analysis.items() if key in fields})
        identity = analysis.get('id')
        if type(identity) is not str or not identity:
            raise ValueError('run identity must be a nonempty string')
        workload = _object(analysis.get('workload', {}), 'run.workload')
        checks = analysis.get('checks', [])
        if not isinstance(checks, list):
            raise ValueError('run checks must be an array')
        nodes = []
        for check in checks:
            if not isinstance(check, dict) or type(check.get('stage')) is not str or not check['stage']:
                raise ValueError('run check requires a nonempty stage')
            nodes.append(CheckNode.read(check, check['stage']))
        return cls(identity, analysis.get('status'), RunValidity.read(workload),
            tuple(nodes),
            analysis.get('configuration'), analysis.get('configuration_sha256'),
            workload.get('runtime_configuration'), _object(analysis.get('implementation', {}), 'run.implementation'),
            analysis.get('traffic_manifests'), analysis.get('runtime_provenance'),
            analysis.get('clock_anchor'), analysis.get('report_timeline'), analysis.get('request_sources'),
            tuple(analysis.get('phases', [])), tuple(analysis.get('events', [])))
