"""CPU-only claim writer tests; Ray operations are simulated explicitly."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import q36_4plus4_claim_guard as guard
import q36_4plus4_make_claim as maker


class ClaimWriterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.run = base / 'q36-disagg-real-0123456789'
        self.run.mkdir()
        self.supervisors = base / 'supervisors'
        self.supervisors.mkdir()
        self.env = {'Q36_RUN_ROOT': str(self.run), 'Q36_RAY_SUPERVISOR_ROOT': str(self.supervisors),
                    'Q36_RAY_ADDRESS': '10.44.49.51:31100', 'Q36_RAY_SESSION_NAME': 'session_123_4321',
                    'Q36_DISAGG_ALLOWED_NODE_IPS': '10.44.49.51,10.44.33.31'}
        self.nodes = [{'Alive': True, 'NodeManagerAddress': ip, 'NodeID': node, 'Resources': {'GPU': 4}}
                      for ip, node in (('10.44.49.51', 'node1'), ('10.44.33.31', 'node2'))]
        def observed(role, pid, nums):
            return {'gpu_uuids': ['GPU-' + str(uuid.UUID(int=n)) for n in nums],
                    'supervisor': {'role': role, 'pid': pid, 'create_time': 1.0, 'monitor_pid': pid+1,
                      'monitor_create_time': 2.0,
                      'status_file': str(self.supervisors / self.run.name / ('ray-' + role + '.status.json'))}}
        self.observed = [observed('head', 4321, range(1, 5)), observed('worker', 5000, range(5, 9))]
        self.session = {'name': 'session_123_4321',
                        'session_dir': '/tmp/q36r-0123456789/session_123_4321',
                        'gcs_address': '10.44.49.51:31100', 'cluster_id': 'ClusterID(1234)'}

    def ray(self):
        ctx = types.SimpleNamespace(get_session_name=lambda: self.session['name'])
        class Task:
            def __init__(self, fn): self.fn = fn
            def options(self, **kwargs): return self
            def remote(self, *args): return (self.fn.__name__, args)
        def get(refs, timeout):
            if isinstance(refs, list): return self.observed
            name, args = refs
            assert name == 'install_peer_claim'
            return {'node_ip': args[1], 'claim_path': str(Path(args[0]) / 'ray-claim.json'),
                    'claim_sha256': args[3]}
        return types.SimpleNamespace(nodes=lambda: self.nodes, get_runtime_context=lambda: ctx,
            _private=types.SimpleNamespace(worker=types.SimpleNamespace(global_worker=types.SimpleNamespace(
                node=types.SimpleNamespace(cluster_id=self.session['cluster_id'],
                    address_info={'session_dir': self.session['session_dir'],
                                  'gcs_address': self.session['gcs_address']})))),
            remote=lambda **kwargs: lambda fn: Task(fn), get=get)

    def run_writer(self, ray=None):
        strategy = types.SimpleNamespace(NodeAffinitySchedulingStrategy=lambda node_id, soft: (node_id, soft))
        with patch.dict('sys.modules', {'ray': types.ModuleType('ray'), 'ray.util': types.ModuleType('ray.util'),
                                        'ray.util.scheduling_strategies': strategy}), \
             patch.object(guard, 'verify_supervisors', return_value=[{'verified': True}]):
            return maker.make_claim(ray or self.ray(), guard, self.env)

    def test_writes_guard_checked_claim_once(self):
        result = self.run_writer()
        self.assertEqual(Path(result['claim_path']).name, 'ray-claim.json')
        ready = json.loads((self.run / 'ray-claim-ready.json').read_text())
        self.assertEqual(ready['claim_sha256'], result['claim_sha256'])
        self.assertEqual(ready['rollout_readback']['node_ip'], '10.44.33.31')
        loaded = guard.load_claim(dict(self.env, Q36_RAY_CLAIM_JSON=result['claim_path'],
                                       Q36_RAY_CLAIM_SHA256=result['claim_sha256']))
        self.assertEqual(loaded['training']['gpu_uuids'], self.observed[0]['gpu_uuids'])
        with self.assertRaises(FileExistsError): self.run_writer()

    def test_duplicate_uuid_or_wrong_session_leaves_no_claim(self):
        self.observed[1]['gpu_uuids'][0] = self.observed[0]['gpu_uuids'][0]
        with self.assertRaisesRegex(ValueError, 'overlap'): self.run_writer()
        self.assertFalse((self.run / 'ray-claim.json').exists())
        self.assertFalse(list(self.run.glob('.ray-claim-*')))
        self.observed[1]['gpu_uuids'][0] = 'GPU-' + str(uuid.UUID(int=5))
        self.env['Q36_RAY_SESSION_NAME'] = 'session_not_owned'
        with self.assertRaisesRegex(ValueError, 'session identity'): self.run_writer()
        self.assertFalse((self.run / 'ray-claim.json').exists())

    def test_peer_copy_is_exclusive_and_verified(self):
        import hashlib
        data = b'{"test":true}\n'
        sha = hashlib.sha256(data).hexdigest()
        fake_ray = types.SimpleNamespace(util=types.SimpleNamespace(get_node_ip_address=lambda: '10.44.33.31'))
        with patch.dict('sys.modules', {'ray': fake_ray}):
            receipt = maker.install_peer_claim(str(self.run), '10.44.33.31', data, sha)
            self.assertEqual(receipt['claim_sha256'], sha)
            self.assertEqual((self.run / 'ray-claim.json').read_bytes(), data)
            with self.assertRaisesRegex(FileExistsError, 'already exists'):
                maker.install_peer_claim(str(self.run), '10.44.33.31', data, sha)

    def test_failed_peer_copy_never_publishes_ready(self):
        ray = self.ray()
        original = ray.get
        def get(refs, timeout):
            if not isinstance(refs, list): raise RuntimeError('peer copy failed')
            return original(refs, timeout)
        ray.get = get
        with self.assertRaisesRegex(RuntimeError, 'peer copy failed'):
            self.run_writer(ray)
        self.assertTrue((self.run / 'ray-claim.json').exists())
        self.assertFalse((self.run / 'ray-claim-ready.json').exists())
        with self.assertRaisesRegex(FileExistsError, 'already exists'):
            self.run_writer()

    def test_rejects_wrong_resource_count_and_extra_nodes(self):
        self.nodes[0]['Resources']['GPU'] = 1
        with self.assertRaisesRegex(RuntimeError, 'four GPUs'): self.run_writer()
        self.nodes[0]['Resources']['GPU'] = 4
        self.nodes.append({'Alive': True, 'NodeManagerAddress': '10.44.48.230'})
        with self.assertRaisesRegex(RuntimeError, 'unexpected nodes'): self.run_writer()

    def test_failed_supervisor_verification_leaves_no_claim(self):
        strategy = types.SimpleNamespace(NodeAffinitySchedulingStrategy=lambda node_id, soft: (node_id, soft))
        with patch.dict('sys.modules', {'ray': types.ModuleType('ray'), 'ray.util': types.ModuleType('ray.util'),
                                        'ray.util.scheduling_strategies': strategy}), \
             patch.object(guard, 'verify_supervisors', side_effect=RuntimeError('changed identity')):
            with self.assertRaisesRegex(RuntimeError, 'changed identity'):
                maker.make_claim(self.ray(), guard, self.env)
        self.assertFalse((self.run / 'ray-claim.json').exists())
        self.assertFalse(list(self.run.glob('.ray-claim-*')))


if __name__ == '__main__': unittest.main()
