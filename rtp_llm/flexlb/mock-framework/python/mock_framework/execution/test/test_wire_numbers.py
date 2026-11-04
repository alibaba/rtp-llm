import json
from pathlib import Path

import pytest

from mock_framework.config.validation import validate
from mock_framework.config import ConfigError


@pytest.mark.parametrize('sample', json.loads((Path(__file__).parent / 'fixtures/wire_numbers.json').read_text()))
def test_numeric_constraints_match_java(sample):
    if sample['valid']:
        assert validate(sample['value'], sample['schema'], 'sample') == sample['value']
    else:
        with pytest.raises(ConfigError):
            validate(sample['value'], sample['schema'], 'sample')
