"""Exercise the real configured checker against new cases and indirect imports."""

import configparser
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_import_graph_includes_every_source_package():
    config = configparser.ConfigParser()
    config.read(ROOT / '.importlinter')
    packages = {path.name for path in (ROOT / 'src').iterdir()
                if path.is_dir() and (path / '__init__.py').exists()}
    assert packages | {'scripts'} <= set(config['importlinter']['root_packages'].split())


def test_ruff_rejects_undefined_names_with_project_configuration(tmp_path):
    command = shutil.which('ruff')
    if command is None:
        pytest.skip('install requirements-quality.txt to exercise Ruff admission')
    source = tmp_path / 'broken.py'
    source.write_text('def execute():\n    return missing_backend\n')
    result = subprocess.run([command, 'check', '--config', str(ROOT / 'ruff.toml'), str(source)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'F821' in result.stdout


@pytest.fixture
def architecture(tmp_path):
    command = shutil.which('lint-imports')
    if command is None:
        pytest.skip('install requirements-quality.txt to exercise import boundary violations')
    config = configparser.ConfigParser()
    config.read(ROOT / '.importlinter')
    for name in config['importlinter']['root_packages'].split():
        package = tmp_path / name
        package.mkdir()
        (package / '__init__.py').write_text('')
    case = tmp_path / 'cases' / 'future_case'
    case.mkdir()
    (case / '__init__.py').write_text('')
    for name in ('analysis', 'inputs', 'report', 'panels'):
        (case / (name + '.py')).write_text('')
    for name in ('compiler', 'program_compiler', 'stage_compiler'):
        (tmp_path / 'scenario' / (name + '.py')).write_text('')
    (tmp_path / '.importlinter').write_text((ROOT / '.importlinter').read_text())

    def run():
        return subprocess.run([command, '--config', '.importlinter', '--no-cache'],
            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(tmp_path)),
            capture_output=True, text=True)
    return tmp_path, run


def test_pure_new_case_passes_all_four_contracts(architecture):
    _, run = architecture
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert '4 kept, 0 broken' in result.stdout


@pytest.mark.parametrize('source,target,contract', [
    ('cases/future_case/analysis.py', 'reporting', '分析不依赖报表'),
    ('cases/future_case/analysis.py', 'cases.future_case.report', '分析不依赖报表'),
    ('cases/future_case/inputs.py', 'scenario.compiler', '输入契约不依赖编译器'),
    ('case_api/__init__.py', 'cases.future_case.inputs', '公共 case 接口不依赖业务实现'),
    ('case_api/__init__.py', 'workload', '公共 case 接口不依赖业务实现'),
    ('scenario/__init__.py', 'scripts', '场景库不依赖命令脚本'),
])
def test_new_case_cannot_break_a_boundary(architecture, source, target, contract):
    root, run = architecture
    (root / source).write_text('import ' + target + '\n')
    result = run()
    assert result.returncode != 0
    assert contract in result.stdout and 'BROKEN' in result.stdout


def test_indirect_compiler_dependency_is_rejected(architecture):
    root, run = architecture
    (root / 'cases/future_case/inputs.py').write_text('import adapters.helper\n')
    (root / 'adapters/helper.py').write_text('import scenario.compiler\n')
    result = run()
    assert result.returncode != 0
    assert 'adapters.helper' in result.stdout and 'scenario.compiler' in result.stdout
