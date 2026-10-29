import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest

from mock_framework import cli
from mock_framework.config import Registry, compile_case, load_yaml
from mock_framework.execution import ExecutionError, prepare_launch, run_case
from mock_framework.execution.launch import local_launch
from mock_framework.input import read_inputs

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'cases/basic/case.yaml'


@pytest.fixture
def basic():
    return compile_case(SOURCE, Registry.standard())


@pytest.fixture
def artifacts(tmp_path):
    framework = tmp_path / 'framework with spaces'
    for relative in ('java/master/target/classes/org/flexlb/MasterNode.class',
                     'java/master/target/classes/org/flexlb/mock/master/ZookeeperNode.class',
                     'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar',
                     'java/client/target/mock-client-1.0-SNAPSHOT-all.jar', 'dependency.jar'):
        path = framework / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (framework / 'java/master/target/dependencies.txt').write_text(str(framework / 'dependency.jar') + '\n')
    java, prometheus = tmp_path / 'java', tmp_path / 'prometheus'
    for path in (java, prometheus):
        path.write_text('#!/bin/sh\nexit 0\n')
        path.chmod(0o755)
    return framework, java, prometheus


def test_basic_case_uses_its_own_finite_input_and_existing_interfaces(basic):
    start = next(step for phase in basic.phases for step in phase.steps if step.interface == 'client/start')
    source = basic.directory / start.args['requests_file']
    requests = list(read_inputs(source))
    assert len(requests) == 20
    assert all(request.token_ids and request.max_new_tokens == 3 for request in requests)
    assert start.args['qps'] == 2 and start.wait_after == '6s'
    assert set(basic.environment) == {'master', 'prefill', 'decode', 'client'}


@pytest.mark.parametrize('mode', ['shared', 'isolated'])
def test_local_recipes_round_trip_without_shell_or_cwd_assumptions(basic, artifacts, mode):
    framework, java, prometheus = artifacts
    launch = local_launch(basic, java, prometheus, framework=framework, engine_mode=mode)
    assert launch == prepare_launch(launch, basic, framework)
    assert launch['engine_mode'] == mode
    assert set(launch['components']) == set(basic.environment)
    assert launch['components']['prefill'] == launch['components']['decode']
    assert launch['components']['master']['command'][-1] == 'org.flexlb.MasterNode'
    assert launch['zookeeper']['command'][-1] == 'org.flexlb.mock.master.ZookeeperNode'
    assert launch['components']['client']['command'][-1].endswith('mock-client-1.0-SNAPSHOT-all.jar')
    assert all(recipe['command'][0] == str(java) and recipe['cwd'] == str(framework)
               for recipe in launch['components'].values())


@pytest.mark.parametrize('damage', ['java', 'prometheus', 'engine', 'master', 'classpath'])
def test_local_recipes_reject_missing_or_unusable_inputs(basic, artifacts, damage):
    framework, java, prometheus = artifacts
    if damage in ('java', 'prometheus'):
        (java if damage == 'java' else prometheus).chmod(0o644)
    elif damage == 'engine':
        (framework / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar').unlink()
    elif damage == 'master':
        (framework / 'java/master/target/classes/org/flexlb/MasterNode.class').unlink()
    else:
        (framework / 'java/master/target/dependencies.txt').write_text('relative-or-missing.jar\n')
    with pytest.raises(ExecutionError):
        local_launch(basic, java, prometheus, framework=framework)


@pytest.mark.parametrize('configuration', ['unsupported-role', 'non-native-master'])
def test_local_recipes_reject_roles_or_master_configuration_the_binaries_cannot_run(basic, artifacts, configuration):
    framework, java, prometheus = artifacts
    environment = ({'vit': basic.environment['prefill']} if configuration == 'unsupported-role'
                   else {'master': {'config': {}}})
    with pytest.raises(ExecutionError):
        local_launch(replace(basic, environment=environment), java, prometheus, framework=framework)


def test_cli_generates_a_readable_launch_file_and_refuses_overwrite(basic, artifacts, tmp_path, monkeypatch):
    framework, java, prometheus = artifacts
    monkeypatch.setattr(cli, '__file__', str(framework / 'python/mock_framework/cli.py'))
    output = tmp_path / 'launch.yaml'
    monkeypatch.setattr(sys, 'argv', ['mock-case', 'launch', str(SOURCE), '--java', str(java),
                                    '--prometheus', str(prometheus), '--output', str(output)])
    cli.main()
    content = output.read_text()
    assert '&id' not in content
    launch = prepare_launch(load_yaml(output), basic, output.parent)
    assert launch['components']['master']['command'][0] == str(java)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and output.read_text() == content


@pytest.mark.skipif(not os.environ.get('PROMETHEUS_BIN') or not os.environ.get('JAVA_HOME'),
                    reason='JAVA_HOME and PROMETHEUS_BIN required for the shipped basic Case')
def test_shipped_basic_case_runs_through_real_components(tmp_path):
    registry = Registry.standard()
    case = compile_case(SOURCE, registry)
    launch = local_launch(case, Path(os.environ['JAVA_HOME']) / 'bin/java', os.environ['PROMETHEUS_BIN'], framework=ROOT)
    output = tmp_path / 'run'
    result = asyncio.run(run_case(case, registry, launch, output))
    assert result['status'] == 'PASS', result
    assert result['cleanup_errors'] == []
    gates = [json.loads(line) for line in (output / 'gates.jsonl').read_text().splitlines()]
    assert {gate['step'] for gate in gates} == {'leader', 'qps', 'requests'}
    assert all(gate['status'] == 'PASS' for gate in gates)
    events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
    started = {event['node']: event['pid'] for event in events if event['event'] == 'process_started'}
    assert set(started) == {event['node'] for event in events if event['event'] == 'process_stopped'}
    for pid in started.values():
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
