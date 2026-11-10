"""CPU subprocess tests for source locking and the driver identity barrier."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_4plus4_launch_candidate as launch


class LaunchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {'Q36_RUN_ROOT': str(self.root), 'Q36_CODE_DIR': str(self.root)}
        self.marker = self.root / 'driver-executed'
        (self.root / 'run_standalone_q36_candidate_v3.py').write_text(
            'from pathlib import Path\nPath(' + repr(str(self.marker)) + ').touch()\nraise SystemExit(7)\n')

    def test_actual_subprocess_exit_and_no_replay(self):
        psutil = types.SimpleNamespace(Process=lambda pid: types.SimpleNamespace(create_time=lambda: 42.0))
        with patch.dict('sys.modules', {'psutil': psutil}), patch.object(launch, 'preflight', return_value={'checked': True}), \
             patch('q36_4plus4_recipe.overrides', return_value=[]):
            self.assertEqual(launch.run(self.env), 7)
            self.assertTrue(self.marker.exists())
            result = json.loads((self.root / 'training-driver.result.json').read_text())
            self.assertEqual(result['actual_wait_returncode'], 7)
            self.assertEqual(result['create_time'], 42.0)
            with self.assertRaisesRegex(FileExistsError, 'replay'): launch.run(self.env)

    def test_identity_failure_never_executes_driver(self):
        def fail(pid): raise RuntimeError('cannot identify child')
        with patch.dict('sys.modules', {'psutil': types.SimpleNamespace(Process=fail)}), \
             patch.object(launch, 'preflight', return_value={'checked': True}), \
             patch('q36_4plus4_recipe.overrides', return_value=[]):
            with self.assertRaisesRegex(RuntimeError, 'identity recording failed'):
                launch.run(self.env)
        self.assertFalse(self.marker.exists())
        result = json.loads((self.root / 'training-driver.result.json').read_text())
        self.assertEqual(result['actual_wait_returncode'], 88)
        self.assertIn('cannot identify child', result['identity_error'])

    def test_source_lock_rejects_wrong_content_or_resolution(self):
        source = self.root / 'qualified_source.py'
        source.write_text('x = 1\n')
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        env = {'SOURCE_ROOT': str(self.root)}
        spec = types.SimpleNamespace(origin=str(source))
        with patch.object(launch.importlib.machinery.PathFinder, 'find_spec', return_value=spec):
            launch.verify_sources(env, {'SOURCE_ROOT': {source.name: sha}})
            with self.assertRaisesRegex(RuntimeError, 'fixed source mismatch'):
                launch.verify_sources(env, {'SOURCE_ROOT': {source.name: '0' * 64}})
        with patch.object(launch.importlib.machinery.PathFinder, 'find_spec', return_value=types.SimpleNamespace(origin='/tmp/other.py')):
            with self.assertRaisesRegex(RuntimeError, 'different source'):
                launch.verify_sources(env, {'SOURCE_ROOT': {source.name: sha}})

    def test_missing_or_wrong_peer_receipt_rejected_before_imports(self):
        acceptance = self.root / 'acceptance'
        acceptance.mkdir()
        env = dict(launch.REQUIRED_FLAGS, Q36_RUN_ROOT=str(self.root),
                   RLLM_RTP_ACCEPTANCE_DIR=str(acceptance), Q36_RAY_CLAIM_SHA256='a' * 64,
                   UCX_TLS='rc,self,sm,cuda_copy,cuda_ipc')
        with self.assertRaisesRegex(RuntimeError, 'both nodes'):
            launch.preflight(env)
        (self.root / 'ray-claim-ready.json').write_text(json.dumps({'run_id': 'wrong'}))
        with self.assertRaisesRegex(RuntimeError, 'receipt differs'):
            launch.preflight(env)

    def test_missing_single_owner_cleanup_flags_fail_preflight(self):
        flags = ('RTP_TP_SIZE', 'RLLM_RTP_GPUS_PER_NODE', 'RLLM_RTP_EXPECTED_REPLICAS')
        for name in flags:
            for value in (None, '99'):
                env = dict(launch.REQUIRED_FLAGS)
                if value is None:
                    env.pop(name)
                else:
                    env[name] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(RuntimeError, name):
                    launch.preflight(env)

    def test_actual_cleanup_profile_qualifies_one_owner(self):
        import importlib.util
        root = Path(os.environ['Q36_RTP_ADAPTER_SOURCE_DIR'])
        spec = importlib.util.spec_from_file_location('test_real_rtp_cleanup', root / 'rtp_cleanup.py')
        cleanup = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'psutil': types.ModuleType('psutil')}):
            spec.loader.exec_module(cleanup)
        with patch.dict(os.environ, launch.REQUIRED_FLAGS, clear=True):
            self.assertEqual(cleanup._qualified_owner_count(), 1)
            for name in ('RTP_TP_SIZE', 'RLLM_RTP_GPUS_PER_NODE', 'RLLM_RTP_EXPECTED_REPLICAS'):
                with patch.dict(os.environ, {name: ''}):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'exact explicit'):
                        cleanup._qualified_owner_count()

    def test_frozen_recipe_tampering_or_cached_wrong_module_is_rejected(self):
        original = Path(__file__).parent / 'q36_4plus4_recipe.py'
        source = self.root / original.name
        source.write_bytes(original.read_bytes())
        expected = launch.SOURCES['Q36_CODE_DIR'][source.name]
        spec = types.SimpleNamespace(origin=str(source))
        sources = {'Q36_CODE_DIR': {source.name: expected}}
        with patch.object(launch.importlib.machinery.PathFinder, 'find_spec', return_value=spec), \
             patch.dict('sys.modules', {'q36_4plus4_recipe': None}):
            launch.verify_sources(self.env, sources)
            source.write_bytes(source.read_bytes() + b'\n# modified\n')
            with self.assertRaisesRegex(RuntimeError, 'fixed source mismatch'):
                launch.verify_sources(self.env, sources)
        source.write_bytes(original.read_bytes())
        with patch.object(launch.importlib.machinery.PathFinder, 'find_spec', return_value=spec), \
             patch.dict('sys.modules', {'q36_4plus4_recipe': types.SimpleNamespace(__file__='/tmp/wrong.py')}):
            with self.assertRaisesRegex(RuntimeError, 'already imported a different source'):
                launch.verify_sources(self.env, sources)

    def test_detached_launcher_lock_blocks_tampering_before_python(self):
        directory = Path(__file__).parent
        source = self.root / 'q36_4plus4_launch_candidate.py'
        source.write_bytes((directory / source.name).read_bytes())
        fake_python = self.root / 'fake-python'
        fake_python.write_text('#!/bin/sh\nprintf invoked >> ' + str(self.marker) + '\n')
        fake_python.chmod(0o700)
        env = dict(os.environ, Q36_CODE_DIR=str(self.root), Q36_LOCKED_PYTHON=str(fake_python))
        command = ['bash', str(directory / 'q36_4plus4_locked_launch.sh'), 'preflight']
        self.assertEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)
        self.assertEqual(self.marker.read_text(), 'invoked')
        source.write_bytes(source.read_bytes() + b'\n# modified\n')
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn('SHA mismatch', result.stderr)
        self.assertEqual(self.marker.read_text(), 'invoked')


if __name__ == '__main__': unittest.main()
