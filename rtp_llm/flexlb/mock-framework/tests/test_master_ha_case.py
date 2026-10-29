import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml

from mock_framework.config import Registry, compile_case, load_yaml
from mock_framework.execution import ExecutionError, run_case
from mock_framework.execution.launch import local_launch
from mock_framework.execution.master import master_port
from mock_framework.input import read_inputs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cases/master_ha_failover/case.yaml'


def test_ha_case_keeps_one_playback_and_full_rolling_order_without_progress_or_sample_floor():
    raw = load_yaml(SOURCE)
    case = compile_case(SOURCE, Registry.standard())
    steps = [step for phase in case.phases for step in phase.steps]
    assert [step.id for step in steps if step.interface in ('master/stop', 'master/restart')] == [
        'kill_a', 'restore_a', 'kill_b', 'restore_b']
    assert sum(step.interface == 'client/start' for step in steps) == 1
    assert sum(step.interface == 'client/stop' for step in steps) == 1
    assert not any(step.interface in ('client/state/get', 'master/idle') for step in steps)
    rates = [step for step in steps if step.interface == 'client/success_rate']
    assert [step.id for step in rates] == ['baseline', 'b_success', 'a_success', 'both_success']
    assert all(step.args == {'range': ['-20s', '0s']} and step.expect.comparisons == {'gte': .95} for step in rates)
    assert 'min_samples' not in SOURCE.read_text()
    assert case.environment['client']['deployment']['replicas'] == 2
    assert len(list(read_inputs(SOURCE.parent / 'requests.binpb.zst'))) == 960
    for phase in raw['phases']:
        for step in phase['steps']:
            if step['id'] in ('kill_a', 'kill_b', 'restore_a', 'restore_b'):
                assert step['wait_after'] == '30s'


def records(output):
    return [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_native_master_restores_address_and_refreshes_only_its_current_discovery(tmp_path):
    from test_master_lifecycle import step, assert_reaped
    document = load_yaml(ROOT / 'cases/basic/case.yaml')
    document['environment'].pop('client')
    document['phases'] = [dict(id='lifecycle', describe='Native kill, restoration, and discovery refresh', timeout='120s', steps=[
        step('a', 'master/leadership/get', expect={'count': 1}),
        step('killed', 'master/stop', args={'targets': {'ref': 'steps.a'}, 'mode': 'kill'}),
        step('restored', 'master/restart', args={'targets': {'ref': 'steps.a'}}, within='60s', wait_after='2s'),
        step('leader', 'master/leadership/get', expect={'eq': {'ref': 'steps.a'}}),
        step('added', 'engine/scale_out', args={'role': 'prefill', 'count': 1}, expect={'count': 1})])]
    source = tmp_path / 'case.yaml'; source.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(source, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    values = {item['step']: item['result'] for item in records(output) if item['event'] == 'step_finished'}
    assert values['a'] == values['restored']
    first = output / 'components/master-0'
    new = first / 'restart-1'
    original = json.loads((first / 'endpoints.json').read_text())
    restored = json.loads((new / 'endpoints.json').read_text())
    assert original['rpc'] == restored['rpc'] and original['http'] == restored['http']
    assert len(json.loads((first / 'discovery.json').read_text())['p-domain']) == 1
    assert len(json.loads((new / 'discovery.json').read_text())['p-domain']) == 2
    process = [item for item in records(output) if item['event'] == 'process_started' and item['node'].startswith('master-')]
    assert len(process) == 2 and process[0]['pid'] != process[1]['pid']
    assert_reaped(output)


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_shipped_ha_case_runs_with_independent_native_masters(tmp_path):
    try:
        master_port(2)
    except ExecutionError as error:
        if 'enable loopback aliases' in str(error):
            pytest.skip(str(error))
        raise
    registry = Registry.standard(); case = compile_case(SOURCE, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    assert all(gate['status'] == 'PASS' for gate in gates)
    assert {item['step'] for item in gates if item['interface'] == 'client/success_rate'} == {
        'baseline', 'b_success', 'a_success', 'both_success'}
    log = records(output)
    starts = [item for item in log if item['event'] == 'process_started' and item['node'].startswith('master-')]
    assert len(starts) == 4 and len({item['pid'] for item in starts}) == 4
    assert len([item for item in log if item['event'] == 'process_stopped' and item['code'] == -9]) == 2
    assert json.loads((output / 'topology.json').read_text()) == []
    from test_master_lifecycle import assert_reaped
    assert_reaped(output)


@pytest.mark.skipif(not os.environ.get('JAVA_HOME') or not os.environ.get('PROMETHEUS_BIN'),
                    reason='Java artifacts, JAVA_HOME and PROMETHEUS_BIN required')
def test_success_rate_through_native_clients_and_prometheus(tmp_path):
    document = load_yaml(ROOT / 'cases/basic/case.yaml')
    document['environment']['client']['deployment'] = {'replicas': 2}
    for phase in document['phases']:
        for step in phase['steps']:
            if step['interface'] == 'client/start':
                step['args']['requests_file'] = str(ROOT / 'cases/basic/requests.binpb.zst')
            if step['interface'] == 'client/qps':
                step.update(interface='client/success_rate', args={'range': ['-20s', '0s']}, expect={'eq': 1.0})
    source = tmp_path / 'case.yaml'; source.write_text(yaml.safe_dump(document))
    registry = Registry.standard(); case = compile_case(source, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    rate, = [gate for gate in gates if gate['interface'] == 'client/success_rate']
    assert rate['actual'] == 1.0
    from test_master_lifecycle import assert_reaped
    assert_reaped(output)
