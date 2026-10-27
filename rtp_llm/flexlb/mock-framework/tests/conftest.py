from pathlib import Path

import pytest

from mock_framework.config import Registry, load_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry():
    return Registry.load(ROOT / 'build/interfaces.json')


@pytest.fixture
def case_data():
    return load_yaml(ROOT / 'tests/fixtures/compile_case.yaml')


@pytest.fixture
def compile_data(tmp_path, registry):
    import yaml
    from mock_framework.config import compile_case
    def compile_(data, *, catalog=None):
        path = tmp_path / 'case.yaml'
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
        return compile_case(path, catalog or registry, shared_root=tmp_path / 'shared')
    return compile_
