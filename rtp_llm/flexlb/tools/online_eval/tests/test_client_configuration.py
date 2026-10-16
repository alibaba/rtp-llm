"""Defaults, strict typed admission, and experiment-to-sender projection."""
from dataclasses import asdict

import pytest

from adapters.client_configuration import ClientConfiguration, ClientLimits, ClientTransport
from adapters.load_client import client_environment
from traffic.playback.configuration import PlaybackConfiguration, PlaybackConfigurationCodec
from traffic.playback.defaults import DEFAULTS


def test_python_default_projection_matches_every_shared_java_default():
    configuration = ClientConfiguration(PlaybackConfiguration(), ClientLimits(), ClientTransport())
    assert configuration.environment() == DEFAULTS
    assert PlaybackConfigurationCodec.from_environment({key: '' for key in DEFAULTS}) == configuration.playback


def test_omitted_sections_resolve_without_leaking_environment(monkeypatch):
    for key in DEFAULTS:
        monkeypatch.setenv(key, 'poison')
    environment, playback = client_environment({'playback': {'duration_s': 30}})
    assert environment == dict(DEFAULTS, DURATION_S='30', REPLAY_SPEED='1.0',
        SEND_MODE_QPS='0.0', RAMP_UP_SECONDS='0.0', PRIORITY='50')
    assert playback['mode'] == 'true-ts'
    assert playback['speed'] == 1


@pytest.mark.parametrize('value', [
    {}, {'playback': {'qps': 1}},
    {'playback': {'mode': 'uniform', 'duration_s': 5}},
    {'playback': {'duration_s': '5'}}, {'playback': {'duration_s': True}},
    {'playback': {'duration_s': 5}, 'limits': {'max_concurrency': 0}},
    {'playback': {'duration_s': 5}, 'limits': {'request_timeout_s': 200}},
    {'playback': {'duration_s': 5}, 'transport': {'channels': '8'}},
    {'playback': {'duration_s': 5}, 'transport': {'event_loop_threads': True}},
    {'playback': {'duration_s': 5}, 'limits': {'request_timeout_s': 2147483647}},
    {'playback': {'duration_s': 5}, 'limits': {'typo': 1}},
    {'playback': {'duration_s': 5}, 'transport': {'typo': 1}},
    {'playback': {'duration_s': 5}, 'limits': None},
    {'playback': {'duration_s': 5}, 'TIMEOUT_MS': '1000'},
    {'playback': {'duration_s': 5}, 'FETCH_OUTPUT_STREAM': 'true'},
])
def test_invalid_layers_fail_before_start(value):
    with pytest.raises(ValueError):
        ClientConfiguration.read(value)


def test_complete_projection_and_priority_are_source_owned():
    value = dict(playback=dict(mode='uniform', duration_s=1350, qps=240),
        limits=dict(max_concurrency=32768, request_timeout_s=600, drain_timeout_s=660),
        transport=dict(channels=16, event_loop_threads=16))
    environment, playback = client_environment(value, priority=70)
    assert environment['TIMEOUT_MS'] == '600000'
    assert environment['RESPONSE_TIMEOUT'] == '660'
    assert environment['N_CHANNELS'] == environment['EVENT_LOOP_THREADS'] == '16'
    assert environment['PRIORITY'] == '70'
    assert environment['FETCH_OUTPUT_STREAM'] == 'true'
    assert environment['REPLAY_UNIQUE_PREFIX'] == 'false'
    assert playback['duration_s'] == 1350
    assert asdict(ClientConfiguration.read(value).limits) == value['limits']
