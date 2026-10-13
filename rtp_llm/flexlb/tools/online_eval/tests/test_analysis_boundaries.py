"""Pure analysis stays importable without acquisition or workflow orchestration."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_analysis_imports_do_not_load_monitoring_or_diagnostics():
    subprocess.run([sys.executable, '-c', """
import sys
import analysis.checks
import analysis.gates
import analysis.request_metrics
import analysis.traffic_fidelity
assert not any(name.split('.')[0] in {'monitoring', 'diagnostics', 'scenario', 'adapters'}
               for name in sys.modules)
assert 'traffic.synthetic.generator' not in sys.modules
assert 'traffic.catalog' not in sys.modules
"""], check=True)


def test_core_analysis_does_not_hide_lazy_workflow_imports():
    for name in ('checks', 'gates', 'request_metrics', 'statistics', 'time_buckets',
                 'traffic_fidelity', 'workload_profile'):
        tree = ast.parse((ROOT / 'src/analysis' / (name + '.py')).read_text())
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
            assert not any(m.split('.')[0] in {'monitoring', 'diagnostics', 'scenario', 'adapters'}
                           or m.startswith(('traffic.synthetic', 'traffic.catalog')) for m in modules)


@pytest.mark.parametrize('module,obsolete', [
    ('analysis.checks', 'check_metric'), ('analysis.traffic_fidelity', 'run'),
    ('analysis.traffic_fidelity', 'load_capture'), ('analysis.workload_profile', 'main'),
    ('monitoring.contracts', 'implementation_identity'), ('artifacts.archive', 'file_digest')])
def test_moved_responsibilities_have_no_legacy_exports(module, obsolete):
    import importlib
    assert not hasattr(importlib.import_module(module), obsolete)


def test_lru_command_uses_the_library_and_rejects_invalid_capacity(tmp_path):
    from analysis.workload_profile import profile
    from traffic.token_plan import read_token_plan
    trace = tmp_path / 'trace.jsonl'
    trace.write_text(''.join(json.dumps(dict(ts=i, il=4, cache_key_block_size=2,
        input_ids=[1, 2, 3, 4], family='same')) + '\n' for i in range(3)))
    output = tmp_path / 'profile.json'
    command = [sys.executable, str(ROOT / 'scripts/commands/profile_workload.py'),
               str(trace), '--output', str(output)]
    subprocess.run(command + ['--capacities', '2', '4'], check=True, cwd=tmp_path)
    assert json.loads(output.read_text()) == profile(read_token_plan(trace), (2, 4))
    previous = output.read_bytes()
    result = subprocess.run(command + ['--capacities', '0'], cwd=tmp_path, capture_output=True)
    assert result.returncode == 2
    assert output.read_bytes() == previous
