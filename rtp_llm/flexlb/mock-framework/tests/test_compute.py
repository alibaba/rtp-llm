import json
import os
from pathlib import Path
import subprocess

import pytest

from mock_framework.config import ConfigError
from mock_framework.execution.records import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('compute', [
    {'expression': '1', 'unexpected': 1},
    {'expression': '1', 'noise': {'scale': -1}},
    {'expression': '1', 'noise': {'scale': '2'}},
    {'expression': '1', 'noise': {'seed': 1.5}},
    {'expression': '1', 'noise': {'seed': 2 ** 63}},
    {'prefill': {'expression': '1'}},
    {'expression': '1', 'noise': None},
])
def test_compiler_uses_java_compute_configuration_schema(case_data, compile_data, compute):
    case_data['environment']['prefill']['config']['compute'] = compute
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_top_level_compute_is_rejected(case_data, compile_data):
    case_data['environment']['compute'] = {'prefill': {'expression': '1'}}
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_python_compile_to_java_formula_noise_and_timer(case_data, compile_data, tmp_path):
    case_data['environment']['prefill']['config'].update({
        'compute.expression': 'param(baseMs, 1) + 0.5*totalComputeTokens', 'compute.noise.scale': 2,
    })
    compiled = compile_data(case_data)
    config = tmp_path / 'component.json'
    output = tmp_path / 'result.json'
    write_json(config, {'component': compiled.environment['prefill'], 'environment': compiled.environment})
    classpath = (ROOT / 'java/shared/compute/target/compute-classpath.txt').read_text().strip()
    java = Path(os.environ.get('JAVA_HOME', '/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home')) / 'bin/java'
    subprocess.run([str(java), '-cp', classpath, 'org.flexlb.mock.compute.ComputeProbe', str(config), str(output)],
                   capture_output=True, text=True, check=True, timeout=10)
    result = json.loads(output.read_text())
    assert result['state'] == 'COMPLETED'
    assert result['base_ns'] == 5_000_000 and result['planned_ns'] == 10_000_000
    assert result['elapsed_ns'] >= 10_000_000
    assert result['effective']['noise'] == {'scale': 2.0, 'stddev_ms': 0.0, 'stddev_ratio': 0.0,
                                            'max_deviation_ms': 0.0, 'seed': 0}
    assert result['effective']['expression'].startswith('param(baseMs')


@pytest.mark.parametrize('key,value', [
    ('decode', {'draft_tokens': -1}), ('decode', {'acceptance_rate': 1.01}),
    ('decode', {'seed': 1.5}), ('generation', {'eos_probability': -0.1}),
    ('generation', {'eos_probability': 1.1}), ('generation', {'unknown': 1}),
])
def test_group_model_types_and_probability_bounds_use_java_catalog(case_data, compile_data, key, value):
    case_data['environment']['prefill']['config'][key] = value
    with pytest.raises(ConfigError):
        compile_data(case_data)
