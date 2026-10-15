"""Behavioral counterexamples for additions crossing author, plan and runtime boundaries."""

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adapters.environment import EnvManager, FlexEnv
from adapters.environment_config import EnvSpec, MasterSpec
from case_api.builder import CaseBuilder
from case_api.criteria import CriteriaContract
from case_api.input_contract import InputContract
from case_api.numeric_parameters import NONNEGATIVE
from case_api.check_inputs import CheckSet, CheckSpec
from case_api.traffic_inputs import java_flow_inputs
from adapters.java_flow_spec import JavaFlowSpec
from scenario.environment_config import environment
from scenario.environment_declaration import EnvironmentDeclaration, startup_options, reconfigurable_fields
from scenario.backend_spec import make_env_spec


def test_heap_options_cross_authored_frozen_and_startup_boundaries():
    plan = environment({'master_jvm_heap': '3g', 'mock_heap': '4g'}, 'env', 'single-nonbatch')
    from scenario.plan import EnvironmentPlan
    plan = EnvironmentPlan.read(plan).to_dict()
    spec = make_env_spec(plan, 'single-nonbatch', {'master_base': 18080})
    assert (spec.master_jvm_heap, spec.mock_heap) == ('3g', '4g')


@pytest.mark.parametrize('name', ['master_jvm_heap', 'mock_heap'])
@pytest.mark.parametrize('bad', ['0g', '4', 'abc', True, 32])
def test_heap_values_cannot_be_silently_dropped(name, bad):
    with pytest.raises(ValueError):
        environment({name: bad}, 'env', 'single-nonbatch')


def test_new_environment_option_needs_only_its_declaration():
    # event_loop_threads already has an adapter implementation. Exposing it to
    # cases adds ONE declaration; parsing, startup, reconfigure and fingerprint
    # acquire it without any new name lists.
    plan = environment({'event_loop_threads': 12}, 'env', 'single-nonbatch')
    spec = make_env_spec(plan, 'single-nonbatch', {'master_base': 18080})
    assert spec.event_loop_threads == 12
    assert 'event_loop_threads' in reconfigurable_fields()
    from scenario.actions.environment import _validate
    context = SimpleNamespace(environment={}, path='replacement', profiles=['single-nonbatch'])
    replacement = _validate({'config_overrides': {}, 'event_loop_threads': 12}, context)
    assert replacement['environments']['single-nonbatch']['event_loop_threads'] == 12
    assert spec.fingerprint() != replace(spec, event_loop_threads=13).fingerprint()
    for invalid in (0, -1, 1.5, True, None):
        with pytest.raises(ValueError):
            environment({'event_loop_threads': invalid}, 'env', 'single-nonbatch')


def test_all_declared_startup_options_are_present_on_backend_model():
    options = {item.name for item in fields(EnvironmentDeclaration) if item.metadata.get('startup')}
    assert options <= {item.name for item in fields(EnvSpec)}
    values = {name: index for index, name in enumerate(sorted(options))}
    assert startup_options({**values, 'run_dir': '/untrusted', 'raw_config': 'bad'}) == values


@pytest.mark.parametrize('item', fields(EnvSpec), ids=lambda item: item.name)
def test_every_startup_field_affects_reuse_identity(item):
    original = EnvSpec(perf={})
    value = getattr(original, item.name)
    if item.name == 'masters':
        changed = [MasterSpec('peer', 18080)]
    elif isinstance(value, bool):
        changed = not value
    elif isinstance(value, (int, float)):
        changed = value + 1
    elif isinstance(value, str):
        changed = value + '-changed'
    elif isinstance(value, list):
        changed = [*value, '--changed']
    elif isinstance(value, dict):
        changed = {**value, 'changed': True}
    elif value is None:
        from flexlb_cfg import ConfigOverride
        from adapters.zookeeper_config import ZooKeeperConfig
        changed = {'run_dir': Path('/other'), 'config_overrides': ConfigOverride(),
                   'zk_consistency': ZooKeeperConfig(), 'master_pv_log': True}.get(item.name, '3g')
    else:
        raise AssertionError('new field needs a behavioral example: ' + item.name)
    assert original.fingerprint() != replace(original, **{item.name: changed}).fingerprint()


def test_nested_master_log_and_options_affect_identity():
    spec = EnvSpec(perf={}, masters=[MasterSpec('A', 18080)])
    changed = deepcopy(spec)
    changed.masters[0].log_dir_name = 'private'
    assert spec.fingerprint() != changed.fingerprint()
    # Equivalent effective management ports retain the same identity.
    changed = deepcopy(spec)
    changed.masters[0].management_port = 18081
    assert spec.fingerprint() == changed.fingerprint()


def test_reuse_compares_captured_startup_intent_even_if_caller_mutates_spec(tmp_path):
    spec = EnvSpec(perf={})
    manager = EnvManager(tmp_path)
    manager.current = FlexEnv(spec, tmp_path, 55151)
    old = manager.current
    spec.mock_auto_fetch = True
    manager.teardown = Mock()
    manager._build = Mock(return_value=SimpleNamespace())
    manager.ensure(spec)
    manager.teardown.assert_called_once()
    manager._build.assert_called_once_with(spec, deadline=None)
    assert old.spec.mock_auto_fetch is False


def test_new_check_reaches_projection_without_another_field_or_numeric_list():
    checks = CheckSet('trial', {'latency': CheckSpec('max_latency', 'latency', 'le', 'ms', ('window',), number=NONNEGATIVE)})
    inputs = InputContract.from_fields({'checks': checks.input_fields})
    criteria = CriteriaContract({'checks': checks.criteria_fields})
    builder = CaseBuilder({}, {'checks': {'latency': dict(metric='trial/latency', unit='ms',
        windows=['window'], op='le', expected=12)}}, number_rules=inputs.numeric_parameters)
    builder.validate_numbers()
    data = builder.bind(inputs)
    projected = criteria.project(checks=checks.compile(builder, data.checks))
    assert projected == {'max_latency': 12}
    assert criteria.read(projected) == projected
    with pytest.raises(ValueError):
        criteria.read({'max_latency': -1})
    with pytest.raises(ValueError):
        criteria.read({'max_latency': 12, 'typo': 1})


def test_capability_scalar_addition_derives_case_numeric_admission(monkeypatch):
    from dataclasses import make_dataclass, field
    from input_contract import NumberRule
    # Model a real extra capability field by extending the owner type only.
    extended = make_dataclass('ExtendedFlow', [('probe_budget', int,
        field(default=2, metadata={'number': NumberRule(True, 1, 10)}))], bases=(JavaFlowSpec,), frozen=True)
    monkeypatch.setattr('case_api.traffic_inputs.JavaFlowSpec', extended)
    declarations = java_flow_inputs()
    assert declarations['probe_budget'] == NumberRule(True, 1, 10)
    inputs = InputContract.from_fields({'traffic': declarations})
    assert inputs.numeric_parameters['traffic.probe_budget'] == NumberRule(True, 1, 10)
    from document_io import load_document
    root = Path(__file__).resolve().parents[1]
    traffic = load_document(root / 'config/scenarios/master_performance.yaml')['parameters']['traffic']
    traffic = dict(traffic, probe_budget=3)
    traffic.pop('kind')
    assert extended.read(traffic).probe_budget == 3
    traffic['probe_budget'] = 11
    with pytest.raises(ValueError):
        extended.read(traffic)


def test_new_environment_field_cannot_be_accepted_without_a_decoder(monkeypatch):
    from scenario.environment_declaration import environment_fields
    monkeypatch.setattr('scenario.environment_config.environment_fields',
                        lambda: environment_fields() | {'forgotten_option'})
    with pytest.raises(ValueError, match='no decoder'):
        environment({'forgotten_option': 12}, 'env', 'single-nonbatch')
