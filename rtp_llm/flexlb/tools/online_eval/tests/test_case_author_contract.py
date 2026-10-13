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
