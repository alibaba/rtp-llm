"""Scope and missing-identity tests; these never signal a process or start Ray."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PROFILE = {'Q36_DISAGG_CLEANUP_SINGLE_NODE_TP4': '1', 'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT': '1',
           'Q36_DISAGG_TRAIN_GPUS': '4', 'Q36_DISAGG_ROLLOUT_GPUS': '4', 'RTP_TP_SIZE': '4',
           'RTP_EP_SIZE': '1', 'RLLM_RTP_GPUS_PER_NODE': '4', 'RLLM_RTP_EXPECTED_REPLICAS': '1'}


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.deploy = self.root / 'deploy'
        self.deploy.mkdir()
        self.audit = self.root / 'audit'
        self.audit.mkdir()
        self.env = dict(PROFILE, RLLM_RTP_DEPLOY_ROOT=str(self.deploy), RLLM_RTP_ACCEPTANCE_DIR=str(self.audit))
        env_patch = patch.dict(os.environ, self.env, clear=True)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.ray = types.ModuleType('ray')
        self.ray.is_initialized = lambda: True
        self.ray.get_runtime_context = lambda: types.SimpleNamespace(namespace='qualified')
        self.ray.get = lambda value, timeout=None: value
        fake_psutil = types.ModuleType('psutil')
        modules = patch.dict(sys.modules, {'ray': self.ray, 'psutil': fake_psutil})
        modules.start()
        self.addCleanup(modules.stop)
        spec = importlib.util.spec_from_file_location('q36_cleanup_candidate', HERE / 'rtp_cleanup.py')
        self.cleanup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cleanup)

    def identity(self, count):
        return {'replica_rank': 0, 'root': str(self.deploy), 'audit_dir': str(self.audit),
                'ray_actor': {'name': 'rtp_server_0_0', 'namespace': 'qualified'},
                'node_owner_count': count, 'node_owners': [{'node_index': n} for n in range(count)]}

    def supply(self, identity):
        actor = types.SimpleNamespace(prepare=types.SimpleNamespace(remote=lambda: identity))
        self.ray.get_actor = lambda name, namespace: actor

    def test_one_owner_discovered_only_in_exact_profile(self):
        self.supply(self.identity(1))
        self.assertEqual(self.cleanup._discover_required_clusters(self.audit, 1), [])
        self.assertEqual(json.loads((self.audit/'rtp_cluster_0.json').read_text())['node_owner_count'], 1)

    def test_legacy_default_still_requires_four_owners(self):
        os.environ.pop('Q36_DISAGG_CLEANUP_SINGLE_NODE_TP4')
        self.supply(self.identity(1))
        self.assertTrue(self.cleanup._discover_required_clusters(self.audit, 1))
        self.assertFalse((self.audit/'rtp_cluster_0.json').exists())
        self.supply(self.identity(4))
        self.assertEqual(self.cleanup._discover_required_clusters(self.audit, 1), [])

    def test_missing_identity_never_becomes_successful_cleanup(self):
        self.ray.get_actor = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('not found'))
        with self.assertRaisesRegex(RuntimeError, 'cleanup incomplete'):
            self.cleanup.cleanup_owned_rtp_processes(self.audit)
        receipt=json.loads((self.audit/'driver-cleanup.json').read_text())
        self.assertFalse(receipt['passed'])
        self.assertEqual(receipt['replicas'], [])

    def test_explicit_profile_rejects_missing_expected_or_wrong_geometry(self):
        del os.environ['RLLM_RTP_EXPECTED_REPLICAS']
        with self.assertRaisesRegex(ValueError, 'exact explicit'):
            self.cleanup.cleanup_owned_rtp_processes(self.audit)
        os.environ['RLLM_RTP_EXPECTED_REPLICAS'] = '1'
        os.environ['RTP_EP_SIZE'] = '4'
        with self.assertRaisesRegex(ValueError, 'exact explicit'):
            self.cleanup.cleanup_owned_rtp_processes(self.audit)

    def test_preexisting_wrong_owner_record_is_rejected(self):
        (self.audit/'rtp_cluster_0.json').write_text(json.dumps(self.identity(4)))
        self.supply(self.identity(4))
        with patch.object(self.cleanup, '_cleanup_multinode', side_effect=AssertionError('must not signal')):
            with self.assertRaisesRegex(RuntimeError, 'exactly one native owner'):
                self.cleanup.cleanup_owned_rtp_processes(self.audit)


if __name__ == '__main__': unittest.main()
