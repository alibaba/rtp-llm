"""One compiler-backed selection and worker assignment for local and hosted CI."""
import json
from dataclasses import dataclass
from pathlib import Path

from scenario import load_scenarios, compile_scenarios
from scenario.catalog import handlers
from scenario.loader import ScenarioError
from scenario.suites import SuiteCatalog, CATALOG, preselect_documents, classify


@dataclass(frozen=True)
class CiInstance:
    identity: str
    case: str
    profile: str


@dataclass(frozen=True)
class CiPlan:
    job: str
    suite: str
    profiles: tuple[str, ...]
    parallel: int
    instances: tuple[CiInstance, ...]
    groups: dict[str, tuple[str, ...]]
    grade: str
    experiment_params: str = "{}"

    def identities(self, worker=None):
        if worker is not None:
            if worker not in self.groups:
                raise ScenarioError('unknown CI worker: ' + worker)
            return self.groups[worker]
        return tuple(instance.identity for instance in self.instances)

    def to_dict(self, worker=None):
        selected = self.identities(worker)
        selected_profiles = [profile for profile in self.profiles if any(identity.endswith("::" + profile) for identity in selected)]
        return {'ci_partition_schema_version': 2, 'job': self.job, 'suite': self.suite,
                'experiment_params': json.loads(self.experiment_params), 'profiles': list(self.profiles), 'parallel': self.parallel, 'grade': self.grade,
                'instances': list(self.identities()), 'worker': worker,
                'selected_instances': list(selected), 'selected_profiles': selected_profiles,
                'groups': {worker: list(ids) for worker, ids in self.groups.items()}}


class CiPlanner:
    def __init__(self, catalog=CATALOG, source=None):
        self.catalog_path = catalog
        self.catalog = SuiteCatalog.read(catalog)
        self.source = source or Path(catalog).parent / 'scenarios'

    def build(self, job_name, *, suite=None, profiles=None, instances=None, grade='normal', experiment_params=None):
        if job_name not in self.catalog.ci_jobs:
            raise ScenarioError('unknown CI job: ' + job_name)
        job = self.catalog.ci_jobs[job_name]
        suite = job.suite if suite is None else suite
        exact = tuple(value.strip() for value in instances.split(',')) if instances else None
        if exact is not None:
            if len(set(exact)) != len(exact) or any(len(value.split('::')) != 3 or not all(value.split('::')) for value in exact):
                raise ScenarioError('instances requires unique complete case::variant::profile IDs')
            requested_profiles = {identity.rsplit('::', 1)[-1] for identity in exact}
            if requested_profiles - set(job.profiles):
                raise ScenarioError('instance profile is outside the CI job')
            selected_profiles = tuple(profile for profile in job.profiles if profile in requested_profiles)
        else:
            selected_profiles = job.profiles if profiles is None else tuple(profiles)
            if not selected_profiles or len(set(selected_profiles)) != len(selected_profiles) or set(selected_profiles) - set(job.profiles):
                raise ScenarioError('profiles requires unique profiles supported by the CI job')
        from scenario.experiment_params import ExperimentParams
        overrides = ExperimentParams.read(experiment_params)
        documents = preselect_documents(load_scenarios(self.source, experiment_params=overrides.to_json()), None if exact is not None else suite, catalog=self.catalog_path)
        selected = []
        registry = handlers()
        for profile in selected_profiles:
            plans = classify(compile_scenarios(documents, profile, registry, grade=grade), None if exact is not None else suite, catalog=self.catalog_path)
            selected.extend(CiInstance(p['id'], p['scenario_id'], p['profile']) for p in plans
                            if exact is None or p['id'] in exact)
        actual = {instance.identity for instance in selected}
        if exact is not None and actual != set(exact):
            raise ScenarioError('unknown exact instance IDs: ' + ', '.join(sorted(set(exact) - actual)))
        if not selected or len(actual) != len(selected):
            raise ScenarioError('empty or duplicate CI selection')
        overrides.require_cases(instance.case for instance in selected)
        groups = {worker: [] for worker in job.workers}
        for instance in selected:
            owners = [worker for worker, policy in job.workers.items() if policy.matches(instance.case, instance.profile)]
            if len(owners) != 1:
                raise ScenarioError(f'CI instance must have exactly one worker: {instance.identity}; matches={owners}')
            groups[owners[0]].append(instance.identity)
        return CiPlan(job_name, suite, selected_profiles, job.parallel, tuple(selected),
                      {worker: tuple(ids) for worker, ids in groups.items()}, grade, overrides.to_json())
