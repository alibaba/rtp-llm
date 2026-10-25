"""Mechanism imports stay isolated from protocols, adjudication and evidence."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CALLERS = {
    "adapters", "analysis", "artifacts", "case_api", "cases", "input_contract",
    "monitoring", "reporting", "scenario", "traffic", "workload",
}


def test_runtime_has_no_static_or_lazy_imports_of_callers():
    violations = []
    for path in (ROOT / "src/runtime").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [])
            for name in imports:
                if name.split(".")[0] in CALLERS:
                    violations.append((path.name, node.lineno, name))
    assert not violations


def test_runtime_does_not_embed_master_protocol_routes():
    violations = []
    for path in (ROOT / "src/runtime").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "/rtp_llm/" in node.value:
                    violations.append((path.name, node.lineno, node.value))
    assert not violations


def test_runtime_mechanisms_load_without_client_or_case_configuration():
    script = """
import builtins
import importlib
from pathlib import Path
import runtime
blocked = {'adapters', 'analysis', 'artifacts', 'case_api', 'cases', 'input_contract',
           'monitoring', 'reporting', 'scenario', 'traffic', 'workload', 'grpc'}
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in blocked:
        raise AssertionError('runtime imported a caller: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
for path in Path(runtime.__file__).parent.glob('*.py'):
    importlib.import_module('runtime.' + path.stem)
from runtime.resource_plan import port_numbers, validate_disjoint_intervals
assert port_numbers([('service', 11000, 11001)]) == (11000, 11001)
validate_disjoint_intervals([('service', 11000, 11001), ('other', 11002, 11003)])
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               FLEXLB_FT_WORKER_PORT_CAPACITY="invalid-backend-configuration")
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, check=True)


@pytest.mark.parametrize("name", ["adapters.requests", "adapters.stream", "adapters.resource_evidence",
                                 "adapters.java_mock_resources", "scenario.outcome", "artifacts.startup"])
def test_moved_components_load_in_a_fresh_interpreter(name):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    subprocess.run([sys.executable, "-c", "import importlib; importlib.import_module(" + repr(name) + ")"],
                   cwd=ROOT, env=env, check=True)
