import json
import os
from pathlib import Path
import subprocess

import pytest

from mock_framework.config import ConfigError
from mock_framework.execution.records import write_json

ROOT = Path(__file__).resolve().parents[1]


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
    java = Path(os.environ['JAVA_HOME']) / 'bin/java'
    subprocess.run([str(java), '-cp', classpath, 'org.flexlb.mock.compute.ComputeProbe', str(config), str(output)],
                   capture_output=True, text=True, check=True, timeout=10)
    result = json.loads(output.read_text())
    assert result['state'] == 'COMPLETED'
    assert result['base_ns'] == 10_000_000 and result['planned_ns'] == 10_000_000
    assert result['elapsed_ns'] >= 10_000_000
    assert result['effective']['forward']['noise'] == {'stddev_ms': 0.0, 'stddev_ratio': 0.0,
                                            'max_deviation_ms': 0.0, 'seed': 0}
    assert result['effective']['forward']['expression'].startswith('2 * (param(baseMs')
