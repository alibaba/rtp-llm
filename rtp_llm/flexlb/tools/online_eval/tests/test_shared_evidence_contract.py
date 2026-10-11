"""Shared provenance and producer control reject incomplete evidence."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from runtime.flow_control import read_status, stop_sending, validate_drain
from tests.ha_fixtures import client_resource
from tests.test_performance_gate import evidence
from workload.run_provenance import validate_gate_provenance


@pytest.mark.parametrize('group,field,value', [
    ('topology', 'prefill', True), ('topology', 'decode', -1),
    ('capacity', 'decode_cache_blocks', '100'), ('capacity', 'prefill_cache_blocks', None),
    ('trace', 'workload_sha256', 'bad'), ('master_artifact', 'jar_sha256', ''),
])
def test_invalid_provenance_structure_cannot_produce_a_gate_pass(group, field, value):
    from cases.master_performance.analysis import analyze
    data = evidence()
    data['provenance'][group][field] = value
    with pytest.raises(ValueError):
        validate_gate_provenance(data['provenance'])
    assert analyze(data)['verdict'] == 'INVALID'


@pytest.mark.parametrize('field', ['configuration_sha256', 'mock_jar_sha256', 'analyzer_sha256'])
def test_shared_artifact_identity_requires_every_hash(field):
    data = evidence()['provenance']
    del data[field]
    with pytest.raises(ValueError, match='SHA256'):
        validate_gate_provenance(data)


def test_control_commands_are_idempotent_and_drain_requires_identity_and_ack(tmp_path):
    identity = dict(run_id='run', group_id='group', phase_id='phase')
    command = stop_sending(tmp_path, identity, None)
    first = (tmp_path / 'stop.json').read_bytes()
    assert stop_sending(tmp_path, identity, command) == command
    assert (tmp_path / 'stop.json').read_bytes() == first
    state = dict(identity, state='DRAINED', submitted=2, terminal=2, applied_command_id=command)
    (tmp_path / 'status.json').write_text(json.dumps(state))
    validate_drain(read_status(tmp_path, identity), identity, terminal_count=2, command_id=command)
    for patch in (dict(terminal=True), dict(submitted=3), dict(applied_command_id='other'),
                  dict(phase_id='other')):
        with pytest.raises(ValueError):
            validate_drain(dict(state, **patch), identity, terminal_count=2, command_id=command)
    state['phase_id'] = 'other'
    (tmp_path / 'status.json').write_text(json.dumps(state))
    with pytest.raises(ValueError, match='identity mismatch'):
        read_status(tmp_path, identity)


def test_failed_start_preserves_primary_error_when_sampler_cleanup_also_fails():
    sampler = Mock()
    sampler.stop.side_effect = RuntimeError('sampler stop failed')
    java = Mock()
    java.run_async.side_effect = OSError('Java start failed')
    client = client_resource(state_sampler=sampler, _client=java, _overrides={},
                             out_dir='/unused', log_file='/unused/log', name='fixture')
    with pytest.raises(OSError, match='Java start failed') as error:
        client.start()
    assert str(error.value.__cause__) == 'sampler stop failed'
    sampler.stop.assert_called_once()


def test_ha_action_requires_source_and_capture_instead_of_hidden_short_trace():
    from cases.master_ha_failover.actions import _ha_validate
    with pytest.raises(ValueError, match='missing YAML parameter'):
        _ha_validate({}, SimpleNamespace(path='stage', environment=dict(master_layout='dual_standalone')))
