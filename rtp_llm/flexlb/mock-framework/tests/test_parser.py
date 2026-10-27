import pytest
import yaml

from mock_framework.config import ConfigError, load_yaml


@pytest.mark.parametrize('content', [
    'a: 1\na: 2',
    'a:\n  b: 1\n  b: 2',
    'a: {x: 1}\nb:\n  <<: {x: 2}',
    'a: &a [*a]',
    'true: 1',
    '!!python/object:os.system {}',
])
def test_rejects_duplicates_merges_cycles_and_unsafe_tags(tmp_path, content):
    path = tmp_path / 'case.yaml'
    path.write_text(content)
    with pytest.raises(ConfigError):
        load_yaml(path)


def test_keeps_native_config_literals_and_does_not_mutate_pyyaml(tmp_path):
    path = tmp_path / 'case.yaml'
    path.write_text('flag: true\nword: on\ntime: 2026-10-27\nconfig:\n  FLEXLB_CONFIG: |\n    {"schemaVersion":3}\n')
    result = load_yaml(path)
    assert result['flag'] is True
    assert result['word'] == 'on'
    assert result['time'] == '2026-10-27'
    assert result['config']['FLEXLB_CONFIG'] == '{"schemaVersion":3}\n'
    assert yaml.safe_load('value: on')['value'] is True
