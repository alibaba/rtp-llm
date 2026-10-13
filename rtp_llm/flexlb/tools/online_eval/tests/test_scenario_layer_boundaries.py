"""Counterexamples for leaf rules, admission ownership and partial startup."""

import ast
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from adapters import port_windows
from adapters.environment import EnvManager
from adapters.environment_config import EnvSpec
from adapters.master_evidence import endpoint_loads, scheduler_inflight, zero_owner_loads
from runtime.deadline import Deadline
from scenario.errors import ScenarioError
from scenario.references import OutputReference, reference

ROOT = Path(__file__).resolve().parents[1]


def test_primitive_import_and_parsing_do_not_load_compiler_or_programs(tmp_path):
    document = tmp_path / "plain.json"
    document.write_text('{"value": 1}')
    script = '''
import builtins
original = builtins.__import__
blocked = {'scenario.compiler', 'scenario.loader', 'scenario.program_compiler', 'case_api', 'adapters'}
def guard(name, *args, **kwargs):
    if name in blocked or name.split('.')[0] in blocked:
        raise AssertionError('primitive imported orchestration: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from scenario.errors import ScenarioError
from scenario.document_io import load_document
from scenario.references import reference, OutputReference
assert load_document(DOCUMENT) == {'value': 1}
assert reference({'$ref': 'stages.ready.output.count'}, 'r', {'ready': {'count': 'integer'}}, 'integer') == 'integer'
assert OutputReference.read({'$ref': 'stages.ready.output.count'}).stage_id == 'ready'
'''.replace('DOCUMENT', repr(str(document)))
    subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                   env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True)


def test_contracts_never_import_stage_compilation():
    tree = ast.parse((ROOT / "src/scenario/contracts.py").read_text())
    assert not any(isinstance(n, ast.ImportFrom) and n.module == 'scenario.stage_compiler'
                   for n in ast.walk(tree))


@pytest.mark.parametrize('value', [None, {'$ref': False}, {'$ref': 'stages.future.output'},
                                   {'$ref': 'stages.future.output.count', 'other': 1}])
def test_reference_grammar_is_shared_and_strict(value):
    with pytest.raises(ScenarioError):
        OutputReference.read(value)
    with pytest.raises(ScenarioError):
        reference(value, 'stage', {})


def test_reference_forward_and_type_checks_survive_shared_decoder():
    value = {'$ref': 'stages.future.output.count'}
    with pytest.raises(ScenarioError, match='forward reference'):
        reference(value, 'stage', {'earlier': {'count': 'integer'}})
    with pytest.raises(ScenarioError, match='expected requests'):
        reference(value, 'stage', {'future': {'count': 'integer'}}, 'requests')


def test_library_admission_is_owned_and_does_not_mutate_environment(tmp_path):
    with patch.object(port_windows, 'PORT_WINDOW_LOCK_DIR', tmp_path), \
         patch.object(port_windows, 'port_in_use', side_effect=lambda port, host: port == 18080):
        before = dict(os.environ)
        with port_windows.admit_port_window(1) as admitted:
            assert (admitted.master_base, admitted.mock_base) == (18070, 54651)
            assert dict(os.environ) == before
            assert port_windows._try_window_lock(18070, 54651, 500, 1) is None
        with port_windows.admit_port_window(1, master=18070, mock=54651, explicit=True):
            pass


def test_probe_error_releases_partial_ownership(tmp_path):
    with patch.object(port_windows, 'PORT_WINDOW_LOCK_DIR', tmp_path), \
         patch.object(port_windows, 'port_in_use', side_effect=OSError('probe unavailable')):
        with pytest.raises(port_windows.PortAdmissionError, match='infrastructure'):
            port_windows.admit_port_window(1, explicit=True)
        preview = port_windows.admit_port_window(1, explicit=True, dry_run=True)
        assert not preview.admitted and preview.lane_port_status is None
    with patch.object(port_windows, 'PORT_WINDOW_LOCK_DIR', tmp_path):
        holders = port_windows._try_window_lock(18080, 55151, 500, 1)
        assert holders is not None
        port_windows._close_window_locks(holders)


def test_partial_mock_failure_retains_exact_owned_process_for_bounded_cleanup(tmp_path):
    created, remembered = [], []
    process = Mock()
    spec = EnvSpec(label='partial', n_prefill=1, n_decode=1, perf={})
    manager = EnvManager(tmp_path, verbose=False, on_environment=created.append,
                         on_processes=lambda env: remembered.append(env.mock), defer_cleanup=True)
    def start(env, **kwargs):
        env.mock = process
        raise RuntimeError('startup rejected')
    with patch.object(manager, '_pick_base_grpc_port', return_value=55151), \
         patch('adapters.environment._pick_master_http_port', return_value=(18080, 'fixture')), \
         patch.object(manager, '_start_mock', side_effect=start):
        with pytest.raises(RuntimeError, match='startup rejected'):
            manager.ensure(spec, deadline=Deadline(float('inf')))
    assert len(created) == 1 and created[0].mock is process
    assert process in remembered and manager.current is None
    process.terminate.assert_not_called()


def test_master_restart_records_replacement_even_when_readiness_fails(tmp_path):
    old, new = object(), object()
    env = NS(master=old)
    seen = []
    manager = EnvManager(tmp_path, on_processes=lambda env: seen.append(env.master))
    def restart(env, log_name=None, **kwargs):
        env.master = new
        raise RuntimeError('not ready')
    with patch.object(manager.master_lifecycle, 'start_master', side_effect=restart):
        with pytest.raises(RuntimeError, match='not ready'):
            manager.start_master(env)
    assert seen == [old, new]


def test_invalid_master_probe_does_not_change_mock_or_environment_spec(tmp_path):
    raw = EnvSpec(label='probe', n_prefill=1, n_decode=1, perf={}, raw_config='invalid')
    mock_spec = replace(raw, raw_config=None)
    manager = EnvManager(tmp_path, mock_spec=mock_spec)
    env = NS(spec=raw, run_dir=tmp_path, base_grpc_port=55151,
             perf_file=tmp_path / 'perf.json', endpoint_file=tmp_path / 'endpoints.json',
             discovery_file=tmp_path / 'discovery.json', mock_http_port=55150)
    # Stop before process spawn: config serialization must see the separate valid mock spec.
    with patch('adapters.environment.MOCK_JAR', NS(is_file=lambda: True)), \
         patch('adapters.environment.resolve_java21', return_value='java'), \
         patch('adapters.environment._write_master_config', side_effect=RuntimeError('stop')) as render:
        with pytest.raises(RuntimeError, match='stop'):
            manager._start_mock(env)
    assert render.call_args.args == (env, mock_spec)
    assert env.spec is raw and env.spec.raw_config == 'invalid'


@pytest.mark.parametrize('value', [False, -1, '0', None])
def test_master_evidence_never_turns_invalid_counts_into_zero(value):
    with pytest.raises(ValueError):
        scheduler_inflight({'scheduler_inflight': value})
    with pytest.raises(ValueError):
        endpoint_loads({'prefill_endpoints': [{'inflight_batches': value}],
                        'decode_endpoints': [{'total_load': 0}]})


def test_zero_owner_decoder_preserves_both_decode_counters():
    data = {'prefill_endpoints': [{'inflight_batches': []}],
            'decode_endpoints': [{'inflight_requests': 1, 'total_load': 0}]}
    assert endpoint_loads(data)['decode'] == [0]
    assert zero_owner_loads(data)['decode'] == [1]


def test_victim_start_renders_its_own_environment_spec(tmp_path):
    spec = EnvSpec(label='victim', n_prefill=1, n_decode=1, perf={})
    env = NS(spec=spec, run_dir=tmp_path, base_grpc_port=55151,
             perf_file=tmp_path / 'perf.json')
    manager = EnvManager(tmp_path, mock_spec=replace(spec, raw_config='mock-only'))
    with patch('adapters.environment.port_accepting', return_value=False), \
         patch('adapters.environment.resolve_java21', return_value='java'), \
         patch('adapters.environment._write_master_config', side_effect=RuntimeError('stop')) as render:
        with pytest.raises(RuntimeError, match='stop'):
            manager.start_victim(env, 'prefill')
    assert render.call_args.args == (env,)
