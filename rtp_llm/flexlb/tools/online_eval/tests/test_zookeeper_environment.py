from __future__ import annotations
import json
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
from scenario.environment_config import environment
from scenario.backend import make_env_spec
from scenario.loader import ScenarioError
from adapters.leadership import LeadershipSnapshot
from adapters.zookeeper_config import ZooKeeperConfig
from scenario.actions.master import _leadership


def test_elected_layout_has_distinct_addresses_and_shared_ports():
    plan = environment({'master_layout': 'dual_zookeeper'}, 'fixture', 'single-nonbatch')
    spec = make_env_spec(plan, 'single-nonbatch', {'master_base': 28000})
    assert [(m.bind_ip, m.http_port, m.advertised_ip) for m in spec.masters] == [
        ('127.0.0.1', 28000, '127.0.0.1'), ('127.0.0.2', 28000, '127.0.0.2')]
    assert spec.zk_consistency == ZooKeeperConfig()


def test_external_zookeeper_is_explicit_and_frozen():
    plan = environment({'master_layout': 'dual_zookeeper', 'zookeeper': {
        'mode': 'external', 'address': 'zk.example:2181/test', 'session_timeout_ms': 15000}}, 'fixture', 'batch-window')
    assert make_env_spec(plan, 'batch-window', {'master_base': 28000}).zk_consistency.address == 'zk.example:2181/test'


@pytest.mark.parametrize('config', [
    {'master_layout': 'single', 'zookeeper': {}},
    {'master_layout': 'dual_zookeeper', 'zookeeper': {'mode': 'external'}},
    {'master_layout': 'dual_zookeeper', 'zookeeper': {'session_timeout_ms': 0}},
    {'master_layout': 'dual_zookeeper', 'zookeeper': {'mode': 'helper', 'address': 'localhost:2181'}},
])
def test_invalid_dependency_cannot_start(config):
    with pytest.raises(ScenarioError):
        environment(config, 'fixture', 'batch-window')


def state(ip, leader, address):
    return {'success': True, 'lbStatus': json.dumps({'consistency_enabled': True,
        'master': leader, 'local_host': ip, 'master_host': address})}


def test_leadership_rejects_split_brain(tmp_path):
    ctx = NS(artifact_dir=tmp_path, resource_count=0, clock=lambda: 0,
             env=NS(master_specs={'A': NS(http_port=28000), 'B': NS(http_port=28000)}))
    deadline = NS(check=lambda: None)
    with patch('scenario.actions.master._master_json', side_effect=[
        state('127.0.0.1', True, '127.0.0.1:28000'), state('127.0.0.2', True, '127.0.0.2:28000')]):
        with pytest.raises(ValueError, match='multiple elected'):
            _leadership(ctx, {'targets': ['A', 'B']}, deadline)
    evidence = json.loads((tmp_path / 'master-leadership-0.json').read_text())
    assert evidence[-1]['leaders'] == ['A', 'B']


def test_leadership_requires_explicit_success():
    with pytest.raises(ValueError):
        LeadershipSnapshot.read({'success': False})


def test_no_elected_leader_yet_is_unknown_not_zero():
    data = state('127.0.0.1', False, None)
    snapshot = json.loads(data['lbStatus'])
    del snapshot['master_host']
    data['lbStatus'] = json.dumps(snapshot)
    assert LeadershipSnapshot.read(data).master_host is None
