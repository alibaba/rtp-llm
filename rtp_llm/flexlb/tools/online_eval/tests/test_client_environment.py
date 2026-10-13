"""Authored client input and trusted runtime injection share strict validation."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from adapters.java_client import ClientOps
from adapters.java_flow import JavaFlowGroup
from adapters.load_client import (
    FRAMEWORK_CLIENT_ENV_VARS, bind_environment, client_environment, collection_environment,
)
from scenario.runtime import Deadline


def settings():
    return dict(DURATION_S='10', MAX_CONCURRENCY='8', REPLAY_UNIQUE_PREFIX='false',
                FETCH_OUTPUT_STREAM='true', playback=dict(mode='uniform', qps=5))


@pytest.mark.parametrize('field', sorted(FRAMEWORK_CLIENT_ENV_VARS))
def test_authored_framework_settings_are_rejected_even_if_they_match(field):
    with pytest.raises(ValueError, match='resolved by framework'):
        client_environment(dict(settings(), **{field: 'true'}))


@pytest.mark.parametrize('patch', [dict(TYPO='1'), dict(DURATION_S='0'),
    dict(MAX_CONCURRENCY='0'), dict(REPLAY_UNIQUE_PREFIX='true'),
    dict(FETCH_OUTPUT_STREAM='false'), dict(TIMEOUT_MS={}),
    dict(playback=dict(mode='true-ts', speed=float('nan'))),
    dict(playback=dict(mode='true-ts', qps=1))])
def test_invalid_input_is_rejected_before_launch(patch):
    with pytest.raises(ValueError):
        client_environment(dict(settings(), **patch))


def test_subprocess_boundary_also_validates_before_creating_artifacts(tmp_path):
    client = ClientOps(None)
    destination = tmp_path / 'client'
    with mock.patch.object(client, '_argv') as argv, mock.patch('adapters.java_client.ProcessOps.start') as launch:
        for patch in [dict(TYPO='1'), dict(FETCH_OUTPUT_STREAM='false')]:
            environment, _ = client_environment(settings())
            with pytest.raises(ValueError):
                client.run_async(dict(environment, **patch), destination, tmp_path / 'log')
        argv.assert_not_called()
        launch.assert_not_called()
    assert not destination.exists()


@pytest.mark.parametrize('profile,latency,events', [('request','true','true'),
    ('aggregate','true','false'), ('diagnostic','false','true')])
def test_collection_policy_is_frozen_and_matches_launch(tmp_path, profile, latency, events):
    client = mock.Mock()
    process = SimpleNamespace(alive=lambda: True)
    client.run_async.return_value = (process, None)
    flow = JavaFlowGroup(client, tmp_path / 'flow', run_id='r', group_id='g', phase_id='p',
                         poll_s=1, collection_profile=profile)
    trace = tmp_path / 'trace.jsonl'
    trace.write_text('{}\n')
    def acknowledge(*args):
        snapshot = dict(trace_sha256=flow.trace_manifest['sha256'], offline_plan_supported=True,
            planner_version='PLAYBACK_V1', token_stride=1, mode='uniform', qps=5,
            speed=1, max_laps=1, identity='structural-relabel', ramp_up_seconds=0, num_shards=1, shard_index=0,
            duration_seconds=10, limit=0)
        (flow.directory / 'playback.json').write_text(json.dumps(snapshot))
        (flow.control / 'status.json').write_text(json.dumps(dict(flow.identity, state='SENDING')))
        return process, None
    client.run_async.side_effect = acknowledge
    import time
    flow.start(trace, settings(), Deadline(time.monotonic()+1), target='127.0.0.1:1234')
    frozen = json.loads((flow.directory / 'flow-input.json').read_text())['environment']
    assert frozen == client.run_async.call_args.args[0]
    assert frozen['SKIP_SERVER_LATENCY'] == latency
    assert frozen['LIVE_CLIENT_EVENTS'] == events
    assert frozen['COLLECTION_PROFILE'] == profile
    assert frozen['GRPC_TARGET'] == '127.0.0.1:1234'
    recorded = json.loads((flow.directory / 'flow-input.json').read_text())['trace']['java_playback']
    assert recorded == json.loads((flow.directory / 'playback.json').read_text())


def test_runtime_collision_is_an_error():
    environment, _ = client_environment(settings())
    with pytest.raises(ValueError, match='conflicts'):
        bind_environment(environment, {'DURATION_S': environment['DURATION_S']})
    assert collection_environment('request', live_events=False)['LIVE_CLIENT_EVENTS'] == 'false'


@pytest.mark.parametrize('field', ['MAX_INPUT_LEN', 'MAX_OUTPUT_LEN', 'FORCE_PRIORITY'])
def test_content_overrides_are_rejected_before_start(field):
    with pytest.raises(ValueError, match='content'):
        client_environment(dict(settings(), **{field: '1'}))


@pytest.mark.parametrize("field", ["UNKNOWN_SETTING", "RATE_CURVE", "PLAYBACK_SEED"])
def test_unknown_client_settings_use_standard_validation(field):
    with pytest.raises(ValueError, match="unknown Java client configuration names"):
        client_environment(dict(settings(), **{field: "1"}))
