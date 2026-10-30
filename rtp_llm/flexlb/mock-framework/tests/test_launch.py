from dataclasses import replace
import sys

import pytest

from mock_framework import cli
from mock_framework.config import load_yaml
from mock_framework.execution import ExecutionError, prepare_launch
from mock_framework.execution.launch import local_launch


@pytest.fixture
def case(case_data, compile_data):
    from test_master import configuration
    case_data['environment']['master']['config'] = configuration()
    case_data['environment']['decode'] = case_data['environment']['prefill']
    case_data['environment']['client'] = {'config': {}}
    return compile_data(case_data)


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


@pytest.mark.parametrize('mode', ['shared', 'isolated'])
def test_local_recipes_round_trip_without_shell_or_cwd_assumptions(case, artifacts, mode):
    framework, java, prometheus = artifacts
    launch = local_launch(case, java, prometheus, framework=framework, engine_mode=mode)
    assert launch == prepare_launch(launch, case, framework)
    assert launch['engine_mode'] == mode
    assert set(launch['components']) == set(case.environment)
    assert launch['components']['prefill'] == launch['components']['decode']
    assert launch['components']['master']['command'][-1] == 'org.flexlb.MasterNode'
    assert launch['zookeeper']['command'][-1] == 'org.flexlb.mock.master.ZookeeperNode'
    assert launch['components']['client']['command'][-1].endswith('mock-client-1.0-SNAPSHOT-all.jar')
    assert all(recipe['command'][0] == str(java) and recipe['cwd'] == str(framework)
               for recipe in launch['components'].values())


@pytest.mark.parametrize('damage', ['java', 'prometheus', 'engine', 'master', 'classpath'])
def test_local_recipes_reject_missing_or_unusable_inputs(case, artifacts, damage):
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
        local_launch(case, java, prometheus, framework=framework)


@pytest.mark.parametrize('configuration', ['unsupported-role', 'non-native-master'])
def test_local_recipes_reject_roles_or_master_configuration_the_binaries_cannot_run(case, artifacts, configuration):
    framework, java, prometheus = artifacts
    environment = ({'vit': case.environment['prefill']} if configuration == 'unsupported-role'
                   else {'master': {'config': {}}})
    with pytest.raises(ExecutionError):
        local_launch(replace(case, environment=environment), java, prometheus, framework=framework)


def test_cli_generates_a_readable_launch_file_and_refuses_overwrite(case, artifacts, tmp_path, monkeypatch):
    framework, java, prometheus = artifacts
    monkeypatch.setattr(cli, '__file__', str(framework / 'python/mock_framework/cli.py'))
    output = tmp_path / 'launch.yaml'
    monkeypatch.setattr(sys, 'argv', ['mock-case', 'launch', str(case.directory / 'case.yaml'), '--java', str(java),
                                    '--prometheus', str(prometheus), '--output', str(output)])
    cli.main()
    content = output.read_text()
    assert '&id' not in content
    launch = prepare_launch(load_yaml(output), case, output.parent)
    assert launch['components']['master']['command'][0] == str(java)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and output.read_text() == content
