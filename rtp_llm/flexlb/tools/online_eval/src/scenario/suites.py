"""Explicit suite membership and typed execution policies for CI jobs."""
import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scenario.loader import ScenarioError, load_document
from scenario.execution_policy import KINDS, normalize_metadata, normalize_execution
from scenario.validation import mapping, names, identifier, number
from flexlb_cfg import PROFILES

CATALOG = Path(__file__).resolve().parents[2] / "config/suites.yaml"

@dataclass(frozen=True)
class CiJob:
    suite: str
    profiles: tuple[str, ...]
    parallel: int

@dataclass(frozen=True)
class SuiteCatalog:
    default_suite: str
    default_profile: str
    suites: dict[str, tuple[str, ...]]
    ci_jobs: dict[str, CiJob]

    @classmethod
    def read(cls, path=CATALOG):
        data = load_document(path)
        fields = {'suite_schema_version', 'default_suite', 'default_profile', 'suites', 'ci_jobs'}
        mapping(data, 'suite catalog', fields, fields)
        if type(data['suite_schema_version']) is not int or data['suite_schema_version'] != 3:
            raise ScenarioError('suite catalog requires suite_schema_version 3')
        if not isinstance(data['suites'], dict):
            raise ScenarioError('suites must be a mapping')
        if not data['suites']:
            raise ScenarioError('suites must be nonempty')
        suites = {}
        for name, members in data['suites'].items():
            identifier(name, 'suites.name')
            if name == 'all':
                raise ScenarioError('reserved suite name: ' + name)
            if (not isinstance(members, list)
                    or any(type(member) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+::[A-Za-z0-9_-]+', member) for member in members)
                    or len(set(members)) != len(members)):
                raise ScenarioError('suite requires unique case::variant identities: ' + name)
            suites[name] = tuple(members)
        if type(data['default_suite']) is not str or data['default_suite'] not in suites:
            raise ScenarioError('default_suite must name a declared suite')
        if not suites[data['default_suite']]:
            raise ScenarioError('default_suite must name a nonempty suite')
        if type(data['default_profile']) is not str or data['default_profile'] not in PROFILES:
            raise ScenarioError('default_profile must name a supported profile')
        if not isinstance(data['ci_jobs'], dict):
            raise ScenarioError('ci_jobs must be a mapping')
        jobs = {}
        for name, value in data['ci_jobs'].items():
            identifier(name, 'ci_jobs.name')
            mapping(value, 'ci_jobs.' + name, {'suite', 'profiles', 'parallel'}, {'suite', 'profiles', 'parallel'})
            if type(value['suite']) is not str or value['suite'] not in suites:
                raise ScenarioError('CI job must name a declared suite: ' + name)
            if not suites[value['suite']]:
                raise ScenarioError('CI job must name a nonempty suite: ' + name)
            names(value['profiles'], 'ci_jobs.' + name + '.profiles', PROFILES)
            if not value['profiles']:
                raise ScenarioError('CI job profiles must be nonempty')
            number(value['parallel'], 'ci_jobs.' + name + '.parallel', minimum=1, integer=True)
            jobs[name] = CiJob(value['suite'], tuple(value['profiles']), value['parallel'])
        return cls(data['default_suite'], data['default_profile'], suites, jobs)

@dataclass(frozen=True)
class Selection:
    members: Optional[tuple[str, ...]] = None
    kind: Optional[str] = None

    @classmethod
    def read(cls, suite, kind, catalog):
        if suite is not None and kind is not None:
            raise ScenarioError('suite and kind selection are mutually exclusive')
        if kind is not None and kind not in KINDS:
            raise ScenarioError('unknown kind: ' + str(kind))
        if suite is not None and suite not in catalog.suites:
            raise ScenarioError('unknown suite: ' + str(suite))
        if suite is not None and not catalog.suites[suite]:
            raise ScenarioError('suite has no members and cannot run: ' + suite)
        return cls(catalog.suites[suite] if suite is not None else None, kind)

    def matches(self, key, kind):
        return (self.members is None or key in self.members) and (self.kind is None or self.kind == kind)


def suite_names(catalog=CATALOG):
    return tuple(SuiteCatalog.read(catalog).suites)


def default_suite(catalog=CATALOG):
    return SuiteCatalog.read(catalog).default_suite


def preselect_documents(documents, suite=None, catalog=CATALOG, *, kind=None):
    selection = Selection.read(suite, kind, SuiteCatalog.read(catalog))
    selected = []
    for path, document in documents:
        variants = [v for v in document['variants']
                    if selection.matches(document['id'] + '::' + v['id'], normalize_metadata(v['metadata'])['kind'])]
        if variants:
            doc = copy.deepcopy(document)
            doc['variants'] = copy.deepcopy(variants)
            selected.append((path, doc))
    return selected


def classify(plans, suite=None, catalog=CATALOG, *, kind=None):
    selection = Selection.read(suite, kind, SuiteCatalog.read(catalog))
    selected = []
    for plan in plans:
        metadata = normalize_metadata(plan['metadata'])
        execution = normalize_execution(plan['execution'], kind=metadata['kind'])
        if selection.matches(plan['scenario_id'] + '::' + plan['variant_id'], metadata['kind']):
            selected.append(dict(plan, metadata=metadata, execution=execution, test_kind=metadata['kind'],
                                 collection_profile=execution['collection'],
                                 workload_runtime=execution['monitoring'] if metadata['kind'] == 'workload' else {}))
    if selection.members is not None:
        actual = {p['scenario_id'] + '::' + p['variant_id'] for p in selected}
        missing = set(selection.members) - actual
        if missing:
            raise ScenarioError('suite is unsupported by the selected source/profile; missing: ' + ', '.join(sorted(missing)))
    return selected
