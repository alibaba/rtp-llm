"""Hosted and local CI share a complete, nonoverlapping execution plan."""
from pathlib import Path
from unittest.mock import patch
import json
import copy
import yaml
import pytest

from scenario.ci_plan import CiPlanner
from scenario.loader import ScenarioError
from scripts.commands import plan_ci, run_ci

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def capacity():
    with patch('scenario.compiler.VICTIM_OFFSETS', (2048, 2049, 2050)), patch('runtime.resource_plan.VICTIM_OFFSETS', (2048, 2049, 2050)):
        yield


def test_default_selection_and_assignment_have_one_source():
    plan = CiPlanner().build('workload')
    assert len(plan.instances) == 6
    assert plan.groups['workload_cache_single'] == ('cache_scale_in::default::single-nonbatch',)
    assert plan.groups['workload_cache_batch'] == ('cache_scale_in::default::batch-window',)
    assert len(plan.groups['workload_other']) == 4
    assigned = [identity for ids in plan.groups.values() for identity in ids]
    assert len(assigned) == len(set(assigned)) == 6
    assert set(assigned) == set(plan.identities())


def test_explicit_controls_override_defaults_and_stay_with_their_case():
    ids = ('cache_scale_in::affinity_disabled::batch-window', 'master_ha_failover::non_rolling::single-nonbatch')
    plan = CiPlanner().build('workload', instances=','.join(ids))
    assert plan.groups['workload_cache_single'] == ()
    assert plan.groups['workload_cache_batch'] == ids[:1]
    assert plan.groups['workload_other'] == ids[1:]


@pytest.mark.parametrize('instances', ['missing::default::batch-window', 'cache_scale_in::default::unknown',
    'cache_scale_in::default', 'cache_scale_in::default::batch-window,',
    'cache_scale_in::default::batch-window,cache_scale_in::default::batch-window'])
def test_bad_ids_cannot_partially_run(instances):
    with pytest.raises(ScenarioError):
        CiPlanner().build('workload', instances=instances)


@pytest.mark.parametrize('profiles', [[], ['unknown'], ['batch-window', 'batch-window']])
def test_profile_selection_cannot_silently_drop_instances(profiles):
    with pytest.raises(ScenarioError):
        CiPlanner().build('workload', profiles=profiles)


@pytest.mark.parametrize('failure', ['overlap', 'missing'])
def test_worker_assignment_must_cover_each_instance_exactly_once(tmp_path, failure):
    data = yaml.safe_load((ROOT / 'config/suites.yaml').read_text())
    workers = data['ci_jobs']['workload']['workers']
    if failure == 'overlap':
        workers['duplicate'] = copy.deepcopy(workers['workload_cache_single'])
    else:
        del workers['workload_cache_single']
    catalog = tmp_path / 'suites.yaml'
    catalog.write_text(yaml.safe_dump(data))
    with pytest.raises(ScenarioError, match='exactly one worker'):
        CiPlanner(catalog, ROOT / 'config/scenarios').build('workload')


def test_preview_and_execution_use_identical_worker_ids(tmp_path, capsys):
    output = tmp_path / 'plan.json'
    assert plan_ci.main(['--job', 'workload', '--worker', 'workload_cache_batch', '--json', str(output)]) == 0
    preview = json.loads(output.read_text())
    plan = CiPlanner().build('workload')
    calls = list(run_ci.commands(plan, tmp_path / 'run', worker='workload_cache_batch'))
    assert len(calls) == 1
    profile, command = calls[0]
    assert profile == 'batch-window'
    assert command[command.index('--instances') + 1].split(',') == preview['selected_instances']
    assert '--all' in command
    assert preview['selected_profiles'] == ['batch-window']
    assert len(preview['instances']) == 6


def test_unknown_or_empty_worker_cannot_be_a_successful_run(tmp_path):
    with pytest.raises(SystemExit):
        run_ci.main(['--job', 'workload', '--worker', 'missing', '--out-dir', str(tmp_path)])
    with pytest.raises(SystemExit):
        run_ci.main(['--job', 'workload', '--worker', 'workload_cache_single', '--profiles', 'batch-window', '--out-dir', str(tmp_path)])
    assert not (tmp_path / 'ci-results.json').exists()


def test_experiment_parameters_reach_compilation_and_execution(tmp_path):
    raw = json.dumps({'master_performance': {'parameters.traffic.client.playback.qps': 1600.0}})
    plan = CiPlanner().build('workload', instances='master_performance::default::batch-window', experiment_params=raw)
    assert plan.to_dict()['experiment_params'] == json.loads(raw)
    command = list(run_ci.commands(plan, tmp_path))[0][1]
    assert json.loads(command[command.index('--experiment-params') + 1]) == json.loads(raw)
    from scenario import load_scenarios, compile_scenarios
    from scenario.catalog import handlers
    docs = load_scenarios(ROOT / 'config/scenarios', experiment_params=raw)
    baseline = compile_scenarios(load_scenarios(ROOT / 'config/scenarios/master_performance.yaml'), 'batch-window', handlers())[0]
    actual = next(p for p in compile_scenarios(docs, 'batch-window', handlers()) if p['id'] == baseline['id'])
    assert actual['implementation']['configuration_sha256'] != baseline['implementation']['configuration_sha256']
    assert actual['implementation']['configuration']['parameters']['traffic']['client']['playback']['qps'] == 1600.0
    from scenario.execution_plan import freeze_plan, load_plan
    frozen = tmp_path / 'execution-plan.json'
    sha = freeze_plan(frozen, [actual])
    assert load_plan(frozen, sha)[0]['implementation']['configuration'] == actual['implementation']['configuration']


@pytest.mark.parametrize('params', [
    '[]', '{"master_performance": {"parameters.missing": 1}}',
    '{"master_performance": {"parameters.traffic.client.playback.qps": "1200"}}',
    '{"master_performance": {"parameters.traffic.client.playback.qps": -1}}',
    '{"master_performance": {"parameters.traffic.client.playback.qps": 1200}}',
    '{"master_performance": {"program": "other"}}',
    '{"master_performance": {"parameters.traffic": {}}}',
    '{"unknown": {"execution.timeout_s": 1000}}',
    '{"master_performance": {"execution.timeout_s": NaN}}',
    '{"master_performance": {"execution.timeout_s": 10, "execution.timeout_s": 20}}',
])
def test_invalid_experiment_parameters_fail_before_execution(params):
    with pytest.raises(ScenarioError):
        CiPlanner().build('workload', experiment_params=params)


def test_experiment_parameter_target_must_be_selected():
    with pytest.raises(ScenarioError, match='unselected'):
        CiPlanner().build('workload', instances='cache_scale_in::default::batch-window',
                          experiment_params='{"master_performance":{"execution.timeout_s":1000}}')


def test_runtime_preparation_tracks_only_selected_helpers():
    plan = CiPlanner().build('workload')
    assert plan.to_dict('workload_other')['prepare_zookeeper'] is True
    assert plan.to_dict('workload_cache_batch')['prepare_zookeeper'] is False
    assert plan.to_dict('workload_cache_single')['prepare_zookeeper'] is False
