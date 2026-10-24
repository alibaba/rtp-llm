import pytest
from scenario.builder import CaseBuilder
from scenario.inputs import InputSchema
from scenario.numeric_parameters import NONNEGATIVE
from scenario.registry import CaseDefinition
from scenario.check_inputs import CheckSet, CheckSpec


def test_optional_number_is_explicit_and_does_not_consume_inputs():
    contract = InputSchema({'custom': {'name': None, 'threshold': NONNEGATIVE}}, optional={'custom.threshold'})
    definition = CaseDefinition({'default': lambda case: None}, input_schema=contract)
    case = CaseBuilder({}, {'custom': {'name': 'x'}}, number_rules=definition.input_schema.numeric_parameters,
                       optional_numbers=contract.optional_numbers)
    case.validate_numbers()
    assert case.read_parameters == set()
    assert contract.parse(case.parameters)['custom'] == {'name': 'x'}
    case.parameters['custom']['threshold'] = -1
    with pytest.raises(ValueError):
        case.validate_numbers()


def test_unknown_inputs_and_environment_typo_are_rejected():
    case = CaseBuilder({'n_prefill': 2}, {'custom': {'known': 1, 'typo': 2}}, number_rules={})
    with pytest.raises(ValueError):
        InputSchema({'custom': {'known': None}}).parse(case.parameters)
    assert case.env.n_prefill == 2
    with pytest.raises(AttributeError):
        _ = case.env.n_prefil


def test_new_threshold_declared_once_derives_numeric_and_metric_contracts():
    checks = CheckSet('trial', {'latency': CheckSpec('maximum', 'latency', 'le', 'ms', ('measurement',))})
    case = CaseBuilder({}, {'checks': {'latency': dict(metric='trial/latency', unit='ms',
        windows=['measurement'], op='le', expected=10)}}, number_rules=checks.numeric_parameters)
    case.validate_numbers()
    assert checks.compile(case, case.value('checks')) == {'maximum': 10}
    assert case.metric_dependencies['trial/latency']['unit'] == 'ms'
    assert checks.fields == {'maximum'}


def test_single_field_contract_decodes_case_owned_frozen_data():
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Parameters:
        threshold: float
    contract = InputSchema({'custom': {'threshold': NONNEGATIVE}})
    def parse_config(raw):
        sections = contract.parse(raw)
        return Parameters(sections['custom']['threshold'])
    case = CaseBuilder({}, {'custom': {'threshold': 3}}, number_rules=contract.numeric_parameters)
    case.validate_numbers()
    assert parse_config(case.parameters) == Parameters(3)


def test_generic_binding_and_explicit_threshold_dimension():
    from scenario.metric_inputs import bind_metrics
    from scenario.check_inputs import metric_criterion
    case = CaseBuilder({}, {}, number_rules={})
    bind_metrics(case, {'fields': {'latency': {'metric': 'hardware/latency', 'labels': {}}}},
                 {'latency': 'ms'}, labels=('machine',), mode='scrape')
    assert case.metric_dependencies['hardware/latency']['labels'] == ('machine',)
    expected, overrides = metric_criterion(case, dict(metric='trial/latency', unit='ms',
        windows=['measurement'], op='le', expected=10, expected_by_profile={'gpu_a': 20}),
        metric='trial/latency', unit='ms', windows=('measurement',), op='le', path='check',
        profiles=True, dimensions={'gpu_a'})
    assert (expected, overrides) == (10, {'gpu_a': 20})


def test_case_owned_timestamp_field_is_explicitly_epoch_seconds():
    from scenario.windows import TimestampOutput, ObservationWindow
    window = ObservationWindow.read({'from': {'stage': 'ready', 'field': 'ready_at'}},
        'window', timestamp_stages={'ready': TimestampOutput('ready', 'ready_at')})
    assert window.boundaries['from'] == {'$ref': 'stages.ready.output.ready_at'}
    with pytest.raises(ValueError):
        TimestampOutput('ready', 'ready_at', 'monotonic_s')


def test_nested_field_addition_derives_names_and_numeric_admission():
    declarations = {'procedure': {'timeout_s': NONNEGATIVE}}
    # One author declaration also adds the numeric rule; no second numeric table.
    declarations['procedure']['probe'] = {'min_samples': NONNEGATIVE}
    contract = InputSchema(declarations)
    assert set(contract.sections['procedure']) == {'timeout_s', 'probe'}
    assert set(contract.numeric_parameters) == {'procedure.timeout_s', 'procedure.probe.min_samples'}
    case = CaseBuilder({}, {'procedure': {'timeout_s': 2, 'probe': {'min_samples': -1}}},
                       number_rules=contract.numeric_parameters)
    with pytest.raises(ValueError, match='min_samples'):
        case.validate_numbers()
    case.parameters['procedure']['probe']['min_samples'] = 3
    case.validate_numbers()
    assert contract.parse(case.parameters)['procedure']['probe'] == {'min_samples': 3}


def test_optional_nested_number_remains_unconsumed_until_binding():
    contract = InputSchema({'checks': {'extra': {'expected': NONNEGATIVE}}},
                                        optional={'checks.extra'})
    assert contract.optional_numbers == {'checks.extra.expected'}
    case = CaseBuilder({}, {'checks': {}}, number_rules=contract.numeric_parameters,
                       optional_numbers=contract.optional_numbers)
    case.validate_numbers()
    assert not case.read_parameters
    assert contract.parse(case.parameters)['checks'] == {}


def test_parsed_configs_are_detached_and_yaml_narrowing_is_applied():
    from input_contract import NumberRule
    schema = InputSchema({'custom': {'threshold': NONNEGATIVE}})
    source = {'custom': {'threshold': 3}}
    first, second = schema.parse(source), schema.parse(source)
    first['custom']['threshold'] = 99
    assert second == source == {'custom': {'threshold': 3}}
    with pytest.raises(ValueError):
        schema.parse(source, {'custom.threshold': NumberRule(integer=False, minimum=4)})
    with pytest.raises(TypeError):
        schema.sections['custom']['threshold'] = None
