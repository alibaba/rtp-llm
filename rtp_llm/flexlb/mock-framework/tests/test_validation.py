from decimal import Decimal

import pytest

from mock_framework.config import ConfigError
from mock_framework.config.validation import duration, number, text, validate


@pytest.mark.parametrize('value, expected', [('20ms', '0.020'), ('1ns', '0.000000001'), ('1.5s', '1.5')])
def test_duration_parsing_converts_to_seconds_but_schema_validation_keeps_text(value, expected):
    assert duration(value) == Decimal(expected)
    assert validate(value, {'type': 'duration'}, 'timeout') == value


def test_duration_callers_keep_their_own_sign_and_zero_rules():
    assert duration('0s') == 0
    assert duration('-20ms', negative=True) == Decimal('-0.020')
    with pytest.raises(ConfigError):
        duration('-20ms')
    with pytest.raises(ConfigError):
        duration('0s', positive=True)
    with pytest.raises(ConfigError):
        validate('0s', {'type': 'duration'}, 'timeout')
    assert validate(['-20ms', '0s'], {'type': 'range'}, 'range') == ['-20ms', '0s']


@pytest.mark.parametrize('value', [True, '1', None, float('nan'), float('inf'), -float('inf'), 10 ** 1000],
                         ids=['bool', 'string', 'null', 'nan', 'inf', 'negative-inf', 'huge-int'])
def test_direct_and_schema_numeric_checks_reject_invalid_values(value):
    with pytest.raises(ConfigError, match='Expected finite number'):
        number(value)
    with pytest.raises(ConfigError, match='rate: expected'):
        validate(value, {'type': 'number'}, 'rate')


@pytest.mark.parametrize('value', [0, 3, 1.5])
def test_numeric_checks_preserve_value_and_type(value):
    assert number(value) is value
    assert validate(value, {'type': 'number'}, 'rate') is value


@pytest.mark.parametrize('value', ['', '  '])
def test_schema_strings_allow_empty_text_but_required_text_does_not(value):
    assert validate(value, {'type': 'string'}, 'label') == value
    with pytest.raises(ConfigError, match='label: expected nonempty string'):
        text(value, 'label')
