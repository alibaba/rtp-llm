"""Public schema regressions: profile identity, declared axes and isolated inputs."""
import copy
import json
from pathlib import Path

import pytest

from cases.config import configure_program
from flexlb_cfg import ConfigOverride, PROFILES, render_env
from scenario import ScenarioError, compile_scenarios, load_scenarios
from scenario.compiler import environment
from scenario.loader import load_document

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_document(ROOT / 'config/scenarios/request_completion.yaml')


def compile_config(data):
    return compile_scenarios([('case.yaml', configure_program(data, 'case.yaml'))])


def test_no_variants_matches_internal_default_without_inheritance():
    data = config()
    original = copy.deepcopy(data)
    document = configure_program(data, 'case.yaml')
    explicit = compile_scenarios([('case.yaml', document)])
    default = copy.deepcopy(document)
    default['stages'] = default.pop('variants')[0]['stages']
    # Test annotations are not compiler defaults; compare executable content.
    implicit = compile_scenarios([('case.yaml', default)])
    for left, right in zip(explicit, implicit):
        for key in ('id', 'environment', 'stages'):
            assert left[key] == right[key]
        assert left['variant_id'] == 'default'
    assert data == original
    for variants in ([], None):
        with pytest.raises(ScenarioError):
            compile_config({**data, 'variants': variants})


def test_profile_caps_use_dispatcher_units_and_l3_is_profile_keyed():
    data = config()
    data['environment']['config_overrides'] = {'request_timeout_ms': 12345}
    data['environment']['profile_overrides'] = {
        'single-nonbatch': {'max_inflight_per_prefill_worker': 96},
        'batch-window': {'max_inflight_per_prefill_worker': 3, 'queue_timeout_ms': {'omit': True}},
    }
    plans = compile_config(data)
    caps = {'single-nonbatch': 96, 'single-batch': 2, 'batch-window': 3, 'window-nonbatch': 64}
    for plan in plans:
        doc = plan['environment']['resolved_config']
        assert doc['dispatcher']['maxInflightPerPrefillWorker'] == caps[plan['profile']]
        assert doc['requestLifecycle']['request']['timeoutMs'] == 12345
        assert ('queueTimeoutMs' in doc['scheduler']) == (plan['profile'] != 'batch-window')
    for profile in PROFILES:
        doc = json.loads(render_env(profile))
        assert doc['dispatcher']['maxInflightPerPrefillWorker'] == (2 if doc['dispatcher']['type'] == 'BATCH' else 64)


@pytest.mark.parametrize('patch', [
    {'config_overrides': {'decision': 'single'}},
    {'profile_overrides': {'batch-window': {'dispatcher': 'non_batch'}}},
    {'profile_overrides': {'single-nonbatch': {'decision': 'fixed_window'}}},
    {'profile_overrides': {'typo': {}}},
    {'profile_overrides': {'single-nonbatch': {'max_requests': 'bad'}}},
    {'profile_overrides': {'single-nonbatch': {'invented': 1}}},
])
def test_invalid_or_retyped_unselected_profiles_fail_during_load(patch):
    data = config()
    data['profiles'] = ['batch-window']
    data['environment'].update(patch)
    with pytest.raises(ScenarioError):
        configure_program(data, 'case.yaml')


def test_second_data_point_is_isolated_and_identity_and_keys_cannot_collide(tmp_path):
    data = config()
    data['variant_axis'] = {'kind': 'data', 'fields': ['parameters.count']}
    data['variants'] = [
        {'id': 'one', 'parameters': {'count': 1}},
        {'id': 'two', 'parameters': {'count': 2}},
    ]
    original = copy.deepcopy(data)
    plans = compile_config(data)
    assert len({p['id'] for p in plans}) == 8
    for plan in plans:
        assert plan['stages'][1]['params']['count'] == (1 if plan['variant_id'] == 'one' else 2)
    assert data == original
    data['variants'][1]['id'] = 'one'
    with pytest.raises(ScenarioError, match='duplicate'):
        compile_config(data)
    path = tmp_path / 'collision.yaml'
    path.write_text('parameters:\n  flow: {count: 1}\n  flow: {count: 2}\n')
    with pytest.raises(ScenarioError, match='duplicate key'):
        load_document(path)


@pytest.mark.parametrize('patch', [
    {'program': 'immediate'}, {'profiles': ['single-nonbatch']},
    {'environment': {'config_overrides': {'queue_timeout_ms': 10}}},
    {'parameters': {'output_len': 10}},
])
def test_variant_cannot_change_undeclared_dimensions(patch):
    data = config()
    data['variant_axis'] = {'kind': 'data', 'fields': ['parameters.count']}
    data['variants'] = [{'id': 'sample', **patch}]
    with pytest.raises(ScenarioError, match='outside declared dimension'):
        compile_config(data)


def test_flow_identity_cannot_disagree_with_program_or_switch():
    data = load_document(ROOT / 'config/scenarios/master_ha_failover.yaml')
    document = configure_program(data, 'ha.yaml')
    assert [v['id'] for v in document['variants']] == ['rolling', 'non_rolling']
    for variant in document['variants']:
        ids = [s['id'] for s in variant['stages']]
        assert ('outage_failures' in ids) == (variant['id'] == 'non_rolling')
    wrong = copy.deepcopy(data)
    wrong['variants'][0]['program'] = 'non_rolling'
    with pytest.raises(ScenarioError, match='flow identity'):
        configure_program(wrong, 'ha.yaml')
    for mode in ('rolling', 'non_rolling'):
        wrong = copy.deepcopy(data)
        wrong['parameters']['dual_master_cycle']['restart_mode'] = mode
        with pytest.raises(ScenarioError, match='owned by flow identity'):
            configure_program(wrong, 'ha.yaml')


@pytest.mark.parametrize('profile', PROFILES)
def test_renderer_cannot_retype_functional_profile(profile):
    doc = json.loads(render_env(profile))
    for axis, replacement in (
        ('dispatcher', 'non_batch' if doc['dispatcher']['type'] == 'BATCH' else 'batch'),
        ('decision', 'single' if doc['scheduler']['decision']['type'] == 'FIXED_WINDOW' else 'fixed_window'),
    ):
        with pytest.raises(ValueError, match='profile identity'):
            render_env(profile, ConfigOverride(**{axis: replacement}))


def test_null_keyed_value_preserves_common_l3_and_l2_owns_linked_defaults():
    from flexlb_profile_data import FUNCTIONAL_DEFAULTS, FUNCTIONAL_PROFILE_KWARGS
    for values in FUNCTIONAL_PROFILE_KWARGS.values():
        assert not FUNCTIONAL_DEFAULTS.keys() & values.keys()
        assert {'decision', 'dispatcher', 'max_requests', 'max_collection_wait_ms',
                'max_predicted_execution_ms', 'max_inflight_per_prefill_worker'} <= values.keys()
    result = environment({'config_overrides': {'request_timeout_ms': 12345},
                          'profile_overrides': {'single-nonbatch': {'request_timeout_ms': None}}},
                         'environment', 'single-nonbatch')
    assert result['resolved_config']['requestLifecycle']['request']['timeoutMs'] == 12345


def test_unused_variant_parameter_and_overlapping_paths_fail():
    data = config()
    data['variant_axis'] = {'kind': 'data', 'fields': ['parameters.coutn']}
    data['variants'] = [{'id': 'two', 'parameters': {'coutn': 2}}]
    with pytest.raises(ScenarioError, match='unused variant parameter'):
        compile_config(data)
    data['variant_axis']['fields'] = ['parameters.completion', 'parameters.completion.expected']
    with pytest.raises(ScenarioError, match='overlapping'):
        compile_config(data)
