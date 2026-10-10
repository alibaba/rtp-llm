"""Universal numeric domains and authored budgets have separate authorities."""

import copy
import math
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from cases.config import configure_program, program_module
from cases.numeric_parameters import (
    COUNT, FRACTION, JAVA_LENGTH, NONNEGATIVE, OFFSET, POSITIVE_COUNT, PRIORITY,
    NumberRule, narrow_parameters, parameter_rules,
)
from scenario.loader import ScenarioError, load_document

ROOT = Path(__file__).resolve().parents[1]
CASES = ('request_completion', 'master_performance', 'cache_scale_in', 'master_ha_failover')


def config(name):
    return load_document(ROOT / 'config/scenarios' / (name + '.yaml'))


@pytest.mark.parametrize('rule,good,bad', [
    (COUNT, [0, 10], [-1, .5]),
    (POSITIVE_COUNT, [1, 10], [0, -1, 1.5]),
    (NONNEGATIVE, [0, .5], [-1]),
    (FRACTION, [0, .5, 1], [-.1, 1.1]),
    (OFFSET, [-5.5, 0, 7.25], []),
    (PRIORITY, [1, 100], [0, 101, 1.5]),
    (JAVA_LENGTH, [1, (1 << 31) - 1], [0, 1 << 31, 1.5]),
])
def test_numeric_types_keep_intrinsic_and_protocol_constraints(rule, good, bad):
    for value in good:
        assert rule.validate(value, 'field') == value
    for value in [True, None, '1', math.nan, math.inf, *bad]:
        with pytest.raises(ValueError, match='field'):
            rule.validate(value, 'field')


def test_ratio_unit_is_not_a_fraction_type():
    assert NONNEGATIVE.validate(2.5, 'skew') == 2.5
    with pytest.raises(ValueError):
        FRACTION.validate(2.5, 'success_share')


@pytest.mark.parametrize('name', CASES)
def test_authored_schemas_only_contain_the_case_budget(name):
    data = config(name)
    assert data.get('parameter_schema', {}) == (
        {'traffic.count': {'maximum': 10000}} if name == 'request_completion' else {})
    document = configure_program(data, name)
    rules = document.implementation['numeric_parameters']['default']
    assert rules
    if 'procedure.setup_timeout_s' in rules:
        assert rules['procedure.setup_timeout_s'] == asdict(NONNEGATIVE)
    if name == 'request_completion':
        assert rules['traffic.input_len'] == asdict(JAVA_LENGTH)
        assert rules['traffic.count']['maximum'] == 10000


def test_shared_field_cannot_drift_between_programs():
    module = program_module('cache_scale_in', 'fixture')
    declarations = dict(module.NUMERIC_PARAMETERS)
    declarations['traffic.source.parameters.priority'] = COUNT
    with patch.object(module, 'NUMERIC_PARAMETERS', declarations):
        with pytest.raises(ScenarioError, match='conflicting shared numeric field'):
            configure_program(config('cache_scale_in'), 'case.yaml')


@pytest.mark.parametrize('spec', [dict(minimum=0), dict(maximum=1 << 31),
                                 dict(integer=False), dict(maximum=math.inf),
                                 dict(minimum=True), dict(minimum=10, maximum=5)])
def test_yaml_cannot_weaken_protocol_type_or_range(spec):
    data = config('request_completion')
    data['parameter_schema']['traffic.input_len'] = spec
    with pytest.raises(ScenarioError):
        configure_program(data, 'case.yaml')


def test_case_and_variant_constraints_intersect_and_are_frozen():
    data = config('request_completion')
    data['variant_axis'] = dict(kind='data', fields=['parameters.traffic.count'])
    data['variants'] = [dict(id='small', parameters=dict(traffic=dict(count=2)),
                             parameter_schema={'traffic.count': {'maximum': 2}})]
    document = configure_program(data, 'case.yaml')
    assert document.implementation['numeric_parameters']['default']['traffic.count']['maximum'] == 10000
    assert document.implementation['numeric_parameters']['small']['traffic.count']['maximum'] == 2
    data['variants'][0]['parameters']['traffic']['count'] = 3
    with pytest.raises(ScenarioError, match='traffic.count'):
        configure_program(data, 'case.yaml')
    data['variants'][0]['parameter_schema']['traffic.count']['maximum'] = 10001
    with pytest.raises(ScenarioError, match='cannot weaken'):
        configure_program(data, 'case.yaml')


def test_declared_fields_are_required_without_yaml_schema():
    data = config('master_performance')
    del data['parameters']['observation']['windows']['measurement']['from']['offset_s']
    with pytest.raises(ScenarioError, match='missing YAML parameter'):
        configure_program(data, 'case.yaml')


@pytest.mark.parametrize('name', CASES)
def test_every_declared_field_rejects_bool_before_program_build(name):
    from cases.config import CaseBuilder
    data = config(name)
    module = program_module(name, 'fixture')
    rules = narrow_parameters(parameter_rules(module.NUMERIC_PARAMETERS), data.get('parameter_schema', {}))
    for path in rules:
        parameters = copy.deepcopy(data['parameters'])
        target = parameters
        parts = path.split('.')
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = True
        with pytest.raises(ScenarioError, match=path):
            CaseBuilder({}, parameters, number_rules=rules).validate_numbers()


def test_master_state_archive_rejects_negative_ledger_counts():
    from cases.master_ha_failover.metrics import _state_series
    row = dict(epoch_s=10, master='A', http_up=1, scheduler_inflight=-1,
               prefill_inflight_requests=0, decode_master_queued=0, decode_confirmed_running=0)
    with pytest.raises(ValueError, match='invalid HA Master state'):
        _state_series([row], 0)


def test_variant_constraint_cannot_escape_its_declared_axis():
    data = config('request_completion')
    data['variant_axis'] = dict(kind='data', fields=['parameters.traffic.count'])
    data['variants'] = [dict(id='small', parameter_schema={'traffic.input_len': {'maximum': 100}})]
    with pytest.raises(ScenarioError, match='outside declared dimension'):
        configure_program(data, 'case.yaml')


def test_nonrepresentable_numbers_are_rejected_without_overflow():
    from input_contract import finite_number
    value = 10 ** 400
    assert not finite_number(value)
    with pytest.raises(ValueError, match='field'):
        COUNT.validate(value, 'field')
