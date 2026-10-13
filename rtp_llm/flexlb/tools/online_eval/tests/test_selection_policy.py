"""Selection defaults, kind boundaries and executable CI policy contracts."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scenario.selection_cli import add_selection_arguments, resolve_selection, selection_arguments
from scripts.commands import run_ci
from scenario.suites import SuiteCatalog, preselect_documents, classify
from scenario import load_scenarios, compile_scenarios
from scenario.catalog import handlers
from scenario.errors import ScenarioError
from scenario.ci_plan import CiPlan, CiInstance, CiPlanner

ROOT = Path(__file__).resolve().parents[1]


def parse(argv, **kwargs):
    parser = argparse.ArgumentParser()
    add_selection_arguments(parser, **kwargs)
    args = parser.parse_args(argv)
    resolve_selection(args, parser)
    return args


def test_listing_and_execution_share_selection_defaults():
    listed, executed = parse([]), parse([], allow_all_profiles=False)
    assert (listed.suite, listed.profile) == (executed.suite, executed.profile) == ('quick', 'batch-window')
    assert selection_arguments(executed) == ['--suite', 'quick']
    assert parse(['--all', '--profile', 'all']).profile is None


def test_quick_checks_basic_readiness_while_core_is_reserved():
    catalog = SuiteCatalog.read()
    assert catalog.suites['quick'] == ('request_completion::default',)
    assert catalog.ci_jobs['quick'].suite == 'quick'
    assert catalog.suites['core'] == ()
    with pytest.raises(ScenarioError, match='suite has no members.*core'):
        preselect_documents([], 'core')
    with pytest.raises(ScenarioError, match='suite has no members.*core'):
        classify([], 'core')


@pytest.mark.parametrize('target', ['default', 'job'])
def test_reserved_suite_cannot_be_default_or_ci_job(tmp_path, target):
    import yaml
    data = yaml.safe_load((ROOT / 'config/suites.yaml').read_text())
    if target == 'default':
        data['default_suite'] = 'core'
    else:
        data['ci_jobs']['quick']['suite'] = 'core'
    path = tmp_path / 'suites.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ScenarioError, match='nonempty suite'):
        SuiteCatalog.read(path)


@pytest.mark.parametrize('argv', [
    ['--suite', 'quick', '--kind', 'workload'], ['--all', '--suite', 'quick'],
    ['--all', '--kind', 'functional'], ['--suite', 'all'],
])
def test_selectors_are_exclusive_and_suite_is_not_a_filter(argv):
    with pytest.raises(SystemExit):
        parse(argv)


def test_workload_suite_excludes_controls_but_kind_keeps_them():
    docs = load_scenarios(ROOT / 'config/scenarios')
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)), patch('adapters.java_mock_resources.VICTIM_OFFSETS', (2048, 2049, 2050)):
        plans = compile_scenarios(docs, 'single-nonbatch', handlers())
    suite = classify(plans, 'workload')
    assert len(suite) == 3
    assert all(p['variant_id'] == 'default' for p in suite)
    assert {p['variant_id'] for p in classify(plans, kind='workload')} == {'default', 'affinity_disabled', 'non_rolling'}
    with pytest.raises(ScenarioError, match='mutually exclusive'):
        preselect_documents(docs, 'quick', kind='functional')


@pytest.mark.parametrize('mutation', ['suite', 'profiles', 'parallel'])
def test_ci_policy_rejects_bad_reference_profile_and_lane_count(tmp_path, mutation):
    import yaml
    data = yaml.safe_load((ROOT / 'config/suites.yaml').read_text())
    data['ci_jobs']['quick'][mutation] = {'suite': 'missing', 'profiles': ['unknown'], 'parallel': 0}[mutation]
    source = tmp_path / 'suites.yaml'
    source.write_text(yaml.safe_dump(data))
    with pytest.raises(ScenarioError):
        SuiteCatalog.read(source)


def test_ci_job_commands_use_frozen_policy_and_separate_profile_outputs(tmp_path):
    profiles = ('batch-window', 'single-nonbatch')
    ids = tuple(CiInstance('case::default::' + p, 'case', p) for p in profiles)
    plan = CiPlan('workload', 'workload', profiles, 1, ids, {'worker': tuple(x.identity for x in ids)}, 'normal')
    calls = list(run_ci.commands(plan, tmp_path, mock_stride=2100, dry_run=True))
    assert [profile for profile, _ in calls] == ['batch-window', 'single-nonbatch']
    for profile, command in calls:
        assert command[command.index('--parallel') + 1] == '1'
        assert '--all' in command
        assert command[command.index('--instances') + 1] == 'case::default::' + profile
        assert command[command.index('--out-dir') + 1] == str(tmp_path / profile)
        assert '--dry-run' in command


def test_ci_failure_stops_remaining_profiles_and_records_partial_run(tmp_path):
    from types import SimpleNamespace
    ids = tuple(CiInstance('case::default::' + p, 'case', p) for p in ('batch-window', 'single-nonbatch'))
    plan = CiPlan('quick', 'core', tuple(x.profile for x in ids), 1, ids, {'quick': tuple(x.identity for x in ids)}, 'normal')
    with patch('scripts.commands.run_ci.resolve_ci_selection', return_value=plan), \
         patch('scripts.commands.run_ci.subprocess.run', return_value=SimpleNamespace(returncode=1)) as run:
        assert run_ci.main(['--job', 'quick', '--out-dir', str(tmp_path)]) == 1
    assert run.call_count == 1
    result = json.loads((tmp_path / 'ci-results.json').read_text())
    assert not result['complete']
    assert result['results'][0]['returncode'] == 1


def test_ci_invalid_later_profile_cannot_start_first_profile(tmp_path):
    with patch('scenario.ci_plan.compile_scenarios', side_effect=[[], ScenarioError('bad profile')]), \
         patch('scenario.ci_plan.classify', return_value=[]), patch('scripts.commands.run_ci.subprocess.run') as run:
        with pytest.raises(SystemExit):
            run_ci.main(['--job', 'quick', '--out-dir', str(tmp_path)])
    run.assert_not_called()
    assert not (tmp_path / 'ci-results.json').exists()
