import pytest
from case_api.builder import CaseBuilder
from case_api.input_contract import InputContract, InputGroup, InputField
from case_api.numeric_parameters import NONNEGATIVE
from case_api.registry import CaseDefinition
from case_api.check_inputs import CheckSet, CheckSpec


def test_optional_number_is_explicit_and_does_not_consume_inputs():
    contract = InputContract({'custom': InputGroup({'name'}, {'threshold'})},
                             (InputField('custom.threshold', NONNEGATIVE, required=False),))
    definition = CaseDefinition({'default': lambda case: None}, input_contract=contract)
    case = CaseBuilder({}, {'custom': {'name': 'x'}}, number_rules=definition.numeric_parameters,
                       optional_numbers=contract.optional_numbers)
    case.validate_numbers()
    assert case.read_parameters == set()
    assert case.bind(contract).custom == {'name': 'x'}
    case.parameters['custom']['threshold'] = -1
    with pytest.raises(ValueError):
        case.validate_numbers()


def test_unknown_inputs_and_environment_typo_are_rejected():
    case = CaseBuilder({'n_prefill': 2}, {'custom': {'known': 1, 'typo': 2}}, number_rules={})
    with pytest.raises(ValueError):
        case.bind(InputContract({'custom': InputGroup({'known'})}))
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
    contract = InputContract.from_fields({'custom': {'threshold': NONNEGATIVE}},
        decode=lambda sections: Parameters(sections['custom']['threshold']))
    case = CaseBuilder({}, {'custom': {'threshold': 3}}, number_rules=contract.numeric_parameters)
    case.validate_numbers()
    assert case.bind(contract) == Parameters(3)


def test_generic_binding_and_explicit_threshold_dimension():
    from case_api.metric_inputs import bind_metrics
    from case_api.check_inputs import metric_criterion
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
    from case_api.windows import TimestampOutput, ObservationWindow
    window = ObservationWindow.read({'from': {'stage': 'ready', 'field': 'ready_at'}},
        'window', timestamp_stages={'ready': TimestampOutput('ready', 'ready_at')})
    assert window.boundaries['from'] == {'$ref': 'stages.ready.output.ready_at'}
    with pytest.raises(ValueError):
        TimestampOutput('ready', 'ready_at', 'monotonic_s')
