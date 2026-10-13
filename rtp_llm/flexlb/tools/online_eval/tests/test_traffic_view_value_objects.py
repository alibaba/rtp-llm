"""Admission failures and seeded generation across data boundaries."""
import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from reporting.view_config import view
from reporting.view_model import SubtitleTemplate
from scenario.loader import ScenarioError
from traffic.playback.configuration import PlaybackConfigurationCodec, normalize
from traffic.synthetic.configuration import IndependentShape, SyntheticWorkload, resolve
from traffic.synthetic.generator import iter_requests


@pytest.mark.parametrize('raw', [
    {'SEND_MODE': 'uniform', 'SEND_MODE_QPS': 'nan'},
    {'REPLAY_SPEED': '-1'}, {'BURST_FACTOR': '.5'}, {'BURST_DUTY': '1'},
    {'DIURNAL_AMPLITUDE': 'nan'}, {'MAX_LAPS': '-1'}, {'MAX_LAPS': str(2**31)},
    {'PLAYBACK_SEED': str(2**63)}, {'LAP_IDENTITY': 'unknown'},
    {'LAP_RETAIN_PROBABILITY': '.5'}, {'LOOP': 'true'},
])
def test_raw_environment_has_the_same_scalar_admission(raw):
    with pytest.raises(ValueError):
        normalize(raw)


def test_playback_value_object_and_complete_projection():
    authored = dict(mode='uniform', qps=13, identity='partial', seed=5,
                    retain_schedule=dict(kind='sequence', values=[0, .5, 1]))
    config = PlaybackConfigurationCodec.decode(authored)
    env, projection = normalize({'playback': authored})
    decoded = PlaybackConfigurationCodec.from_environment(env)
    assert config == decoded
    assert projection == decoded.to_dict()
    assert config.intensity.retain_schedule.probability(2) == .5
    with pytest.raises(ValueError):
        normalize({'playback': None})


def test_synthetic_domain_is_immutable_and_not_a_mapping():
    parameters = dict(seed=42, count=60, families=3, prefix_blocks=2, zipf_alpha=1,
                      cold_fraction=.3, session_requests=3, session_growth_blocks=1, output_tokens=8)
    workload, _ = resolve(parameters)
    assert isinstance(workload, SyntheticWorkload)
    assert isinstance(workload.shape, IndependentShape)
    rows = list(iter_requests(workload, 'golden'))
    digest = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()
    assert digest == '03bf45068987a81b361fd892865d850016c415666303be0e37a9a5388048f084'
    assert rows == list(iter_requests(workload, 'golden'))


@pytest.mark.parametrize('subtitle', ['{verdict:{unknown}}', '{verdict:d}', '{verdict!r}', '{verdict.name}'])
def test_subtitle_templates_fail_when_loaded(subtitle):
    data = view('cache_scale_in.yaml').to_dict()
    data['report']['subtitle'] = subtitle
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError):
            view('cache_scale_in.yaml')


def test_subtitle_template_uses_only_frozen_result_fields():
    template = SubtitleTemplate('{{run}} {verdict} / {monitoring_status}')
    assert template.render(verdict='PASS', monitoring_status='WARN') == '{run} PASS / WARN'


@pytest.mark.parametrize('mutation', ['reversed_bounds', 'equal_bounds', 'duplicate_curve'])
def test_panel_relations_fail_at_admission(mutation):
    data = view('cache_scale_in.yaml').to_dict()
    panel = data['charts']['panels'][0]
    if mutation == 'duplicate_curve':
        panel['curve_ids'].append(panel['curve_ids'][0])
    else:
        next(iter(panel['axes'].values())).update(min=2, max=1 if mutation == 'reversed_bounds' else 2)
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError):
            view('cache_scale_in.yaml')


def test_deterministic_curve_roundtrip_uses_one_seed_default():
    config = PlaybackConfigurationCodec.decode(dict(mode='uniform', qps=10, rate_curve=[[0, 1], [5, 2]]))
    assert config == PlaybackConfigurationCodec.from_environment(config.environment())


def test_cache_subtitle_fields_are_producer_scoped():
    with pytest.raises(ValueError, match="unknown subtitle"):
        SubtitleTemplate("{expected_behavior} {observed_behavior}")
    template = SubtitleTemplate("{expected_behavior} {observed_behavior}", cache_behavior=True)
    assert template.render(verdict="PASS", monitoring_status="OK",
                           expected_behavior="stable", observed_behavior="stable") == "stable stable"
