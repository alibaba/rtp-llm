import json
import os
from pathlib import Path
import subprocess

import pytest

from mock_framework.config import ConfigError
from mock_framework.execution.records import write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('compute', [
    {'expression': '1'},
    {'forward': {'expression': '1', 'unexpected': 1}},
    {'forward': {'expression': '1', 'noise': {'stddev_ms': -1}}},
    {'forward': {'expression': '1', 'noise': {'scale': 2}}},
    {'forward': {'expression': '1', 'scale': 2}},
    {'forward': {'expression': '1', 'noise': {'seed': 1.5}}},
    {'forward': {'expression': '1', 'noise': {'seed': 2 ** 63}}},
    {'forward': {'expression': '1', 'noise': None}},
    {'forward': None},
])
def test_java_loader_rejects_invalid_compute_configuration(case_data, compile_data, compute, tmp_path):
    case_data['environment']['prefill']['config']['compute'] = compute
    compiled = compile_data(case_data)
    config = tmp_path / 'component.json'
    write_json(config, {'component': compiled.environment['prefill'], 'environment': compiled.environment})
    classpath = (ROOT / 'java/shared/compute/target/compute-classpath.txt').read_text().strip()
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    result = subprocess.run([str(java), '-cp', classpath, 'org.flexlb.mock.compute.ComputeProbe', str(config), str(tmp_path / 'result')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'MOCK_COMPONENTS_DIR' not in result.stderr
    assert not (tmp_path / 'result').exists()


def test_top_level_compute_is_rejected(case_data, compile_data):
    case_data['environment']['compute'] = {'prefill': {'expression': '1'}}
    with pytest.raises(ConfigError):
        compile_data(case_data)


def test_python_compile_to_java_formula_noise_and_timer(case_data, compile_data, tmp_path):
    case_data['environment']['prefill']['config'].update({
        'compute.forward.expression': '2 * (param(baseMs, 1) + 0.5*totalComputeTokens)',
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
    assert result['base_ns'] == 10_000_000 and result['planned_ns'] == 10_000_000
    assert result['elapsed_ns'] >= 10_000_000
    assert result['effective']['forward']['noise'] == {'stddev_ms': 0.0, 'stddev_ratio': 0.0,
                                            'max_deviation_ms': 0.0, 'seed': 0}
    assert result['effective']['forward']['expression'].startswith('2 * (param(baseMs')


@pytest.mark.parametrize('key,value', [
    ('decode', {'draft_tokens': -1}), ('decode', {'acceptance_rate': 1.01}),
    ('decode', {'seed': 1.5}), ('eos', {'probability': -0.1}),
    ('eos', {'probability': 1.1}), ('eos', {'unknown': 1}),
])
def test_java_loader_rejects_invalid_generation_models(case_data, compile_data, key, value, tmp_path):
    role = 'decode' if key == 'decode' else 'prefill'
    case_data['environment'][role] = {'engine': 'rtp', 'config': {
        'seq_size_per_block': 16, 'test_block_num': 9, 'max_seq_len': 4096, 'compute': {'forward': {'expression': '1'}, key: value}}}
    compiled = compile_data(case_data)
    config = tmp_path / 'component.json'
    write_json(config, {'id': role + '-0', 'component_role': role, 'component': compiled.environment[role]})
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    env = dict(os.environ, MOCK_CONFIG_FILE=str(config), MOCK_OUTPUT_DIR=str(tmp_path), MOCK_READY_FILE=str(tmp_path / 'ready.json'))
    result = subprocess.run([str(java), '-cp', str(ROOT / 'java/engine/rtp/target/mock-rtp-1.0-SNAPSHOT-all.jar'), 'org.flexlb.mock.engine.rtp.RtpNode'],
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'MOCK_COMPONENTS_DIR' not in result.stderr
    assert not (tmp_path / 'ready.json').exists()
