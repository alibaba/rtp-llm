"""Decode data-only YAML into explicit declarations before case compilation."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

from cases.config_data import mapping, data_only
from cases.variants import VariantAxis
from scenario.loader import ScenarioError
from scenario.suites import normalize_metadata, normalize_execution
from scenario.validation import names
from flexlb_cfg import PROFILES


@dataclass(frozen=True)
class VariantDeclaration:
    identity: str
    program: Optional[str]
    environment: dict
    parameters: dict
    parameter_schema: dict
    declaration: dict = field(repr=False)

    @classmethod
    def read(cls, value, source):
        mapping(value, {'id', 'program', 'profiles', 'environment', 'execution', 'parameters',
                        'metadata', 'parameter_schema', 'reports', 'reporting'}, source)
        groups = {}
        for name in ('environment', 'parameters', 'parameter_schema'):
            group = value.get(name, {})
            if not isinstance(group, dict):
                raise ScenarioError(source + '.' + name + ': expected mapping')
            groups[name] = copy.deepcopy(group)
        return cls(value.get('id'), value.get('program'), **groups, declaration=copy.deepcopy(value))


@dataclass(frozen=True)
class MonitoringPolicy:
    capture_metrics: bool
    sample_interval_s: float
    max_sample_gap_s: float
    collector_shutdown_s: float
    query_plan: Optional[str]

    @classmethod
    def read(cls, value):
        return cls(value['capture_metrics'], value['sample_interval_s'], value['max_sample_gap_s'],
                   value['collector_shutdown_s'], value.get('query_plan'))

    def to_dict(self):
        result = {name: getattr(self, name) for name in self.__dataclass_fields__ if name != 'query_plan'}
        if self.query_plan is not None:
            result['query_plan'] = self.query_plan
        return result


@dataclass(frozen=True)
class ExecutionPolicy:
    collection: str
    monitoring: MonitoringPolicy
    timeout_s: Optional[float]
    stage_timeout_s: Optional[float]
    cleanup_timeout_s: Optional[float]
    finalize_timeout_s: Optional[float]
    report_timeout_s: Optional[float]

    @classmethod
    def read(cls, value, kind):
        normalized = normalize_execution(value, kind=kind)
        return cls(normalized['collection'], MonitoringPolicy.read(normalized['monitoring']),
                   *(normalized.get(name) for name in ('timeout_s', 'stage_timeout_s', 'cleanup_timeout_s',
                                                      'finalize_timeout_s', 'report_timeout_s')))

    def to_dict(self):
        return {'collection': self.collection, 'monitoring': self.monitoring.to_dict(),
                **{name: getattr(self, name) for name in self.__dataclass_fields__
                   if name not in {'collection', 'monitoring'} and getattr(self, name) is not None}}


@dataclass(frozen=True)
class CaseDeclaration:
    case: str
    identity: str
    program: str
    profiles: tuple[str, ...]
    environment: dict
    parameters: dict
    parameter_schema: dict
    metadata: dict
    execution: ExecutionPolicy
    variants: tuple[VariantDeclaration, ...]
    variant_axis: Optional[VariantAxis]
    reports: Optional[tuple[str, ...]]
    reporting: Optional[dict]
    declaration: dict = field(repr=False)

    @classmethod
    def read(cls, value, source):
        data_only(value, source)
        mapping(value, {'case_schema_version', 'case', 'program', 'variant_axis', 'id', 'profiles',
                        'environment', 'execution', 'parameters', 'variants', 'metadata',
                        'parameter_schema', 'reports', 'reporting'}, source)
        if type(value.get('case_schema_version')) is not int or value['case_schema_version'] != 2:
            raise ScenarioError(f'{source}: only data-only case_schema_version 2 is accepted; move orchestration into Python')
        metadata = normalize_metadata(value.get('metadata'))
        execution = ExecutionPolicy.read(value.get('execution'), metadata['kind'])
        profiles = value.get('profiles')
        if not isinstance(profiles, list) or not profiles:
            raise ScenarioError(f'{source}: profiles must be a nonempty string list in YAML')
        names(profiles, source + '.profiles', PROFILES)
        groups = {}
        for name in ('environment', 'parameters', 'parameter_schema'):
            group = value.get(name, {})
            if not isinstance(group, dict):
                raise ScenarioError(source + '.' + name + ': expected mapping')
            groups[name] = copy.deepcopy(group)
        reports = value.get('reports')
        if reports is not None and not isinstance(reports, list):
            raise ScenarioError(source + '.reports: expected list')
        return cls(value.get('case'), value.get('id', value.get('case')), value.get('program'),
                   tuple(profiles), **groups, metadata=metadata, execution=execution,
                   variant_axis=VariantAxis.from_config(value, source),
                   variants=tuple(VariantDeclaration.read(row, source + '.variants') for row in value.get('variants', [])),
                   reports=tuple(reports) if reports is not None else None, reporting=copy.deepcopy(value.get('reporting')),
                   declaration=copy.deepcopy(value))
