#!/usr/bin/env python3
"""Compile manual CI selection and partition it into three isolated workers."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from scenario import load_scenarios, compile_scenarios
from scenario.catalog import handlers
from scenario.loader import ScenarioError
from scenario.suites import SuiteCatalog, preselect_documents, classify
from scripts.commands.list_cases import select

PROFILES = ('single-nonbatch', 'batch-window')
GROUPS = ('workload_cache_single', 'workload_cache_batch', 'workload_other')


def partition(plans, catalog):
    """Keep cache controls with their case; put remaining selected cases together."""
    cache_cases = {member.split('::')[0]
                   for member in catalog.suites[catalog.ci_jobs[GROUPS[0]].suite]}
    result = {group: [] for group in GROUPS}
    seen = set()
    for plan in plans:
        if plan['id'] in seen or plan['profile'] not in PROFILES:
            raise ScenarioError('duplicate instance or unsupported CI profile: ' + plan['id'])
        seen.add(plan['id'])
        group = GROUPS[2]
        if plan['scenario_id'] in cache_cases:
            group = GROUPS[PROFILES.index(plan['profile'])]
        result[group].append(plan['id'])
    return result


def build_selection(suite, profiles, instances=None):
    catalog = SuiteCatalog.read()
    exact = [value.strip() for value in instances.split(',')] if instances else None
    if exact is not None:
        if len(set(exact)) != len(exact) or any(
                len(value.split('::')) != 3 or value.rsplit('::', 1)[-1] not in PROFILES
                for value in exact):
            raise ScenarioError('instances requires unique complete case::variant::profile IDs')
        profiles = [profile for profile in PROFILES
                    if any(value.endswith('::' + profile) for value in exact)]
    elif (not profiles or len(set(profiles)) != len(profiles)
          or any(profile not in PROFILES for profile in profiles)):
        raise ScenarioError('profiles requires supported, unique profile names')
    documents = preselect_documents(load_scenarios(ROOT / 'config/scenarios'),
                                   None if exact else suite)
    plans = []
    registry = handlers()
    for profile in profiles:
        compiled = classify(compile_scenarios(documents, profile, registry),
                            None if exact else suite)
        ids = ','.join(value for value in exact if value.endswith('::' + profile)) if exact else None
        plans.extend(select(compiled, ids))
    return {'ci_partition_schema_version': 1,
            'instances': [plan['id'] for plan in plans],
            'groups': partition(plans, catalog)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='workload')
    parser.add_argument('--profiles', default=','.join(PROFILES))
    parser.add_argument('--instances', default='')
    parser.add_argument('--json', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        payload = build_selection(args.suite, [p.strip() for p in args.profiles.split(',')], args.instances)
    except ScenarioError as exc:
        parser.error(str(exc))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
