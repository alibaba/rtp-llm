"""CPU adversarial tests for a task claim. No Ray/GPU processes are started."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import q36_4plus4_claim_guard as guard
import q36_4plus4_claim_loader as loader


class ClaimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / 'q36-disagg-real-0123456789'
        self.run.mkdir()
        self.task = self.root / 'supervisors'
        self.task.mkdir()
        self.path = self.run / 'RAY-CLAIM.json'
        self.env = {'Q36_RUN_ROOT': str(self.run), 'Q36_RAY_CLAIM_JSON': str(self.path),
                    'Q36_RAY_SESSION_NAME': 'session_123_4321', 'Q36_RAY_ADDRESS': '10.44.49.51:31100',
                    'Q36_RAY_SUPERVISOR_ROOT': str(self.task),
                    'Q36_DISAGG_ALLOWED_NODE_IPS': '10.44.49.51,10.44.33.31'}
        def role(ip, node, name, pid, nums):
            return {'ip': ip, 'node_id': node, 'gpu_uuids': [str(uuid.UUID(int=n)) for n in nums],
                    'supervisor': {'role': name, 'pid': pid, 'create_time': 1.0,
                                   'monitor_pid': pid + 1, 'monitor_create_time': 2.0,
                                   'status_file': str(self.task / self.run.name / ('ray-' + name + '.status.json'))}}
        self.claim = {'schema': 'q36-owned-ray-4plus4-v1', 'run_id': self.run.name,
                      'session': {'name': 'session_123_4321',
                                  'session_dir': '/tmp/q36r-0123456789/session_123_4321',
                                  'gcs_address': '10.44.49.51:31100', 'cluster_id': 'ClusterID(1234)'},
                      'training': role('10.44.49.51', 'node1', 'head', 4321, range(1, 5)),
                      'rollout': role('10.44.33.31', 'node2', 'worker', 5000, range(5, 9))}

    def write_claim(self):
        self.path.write_text(json.dumps(self.claim))
        self.env['Q36_RAY_CLAIM_SHA256'] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def fake_ray(self, name=None, cluster=None, nodes=None):
        session = self.claim['session']
        ctx = types.SimpleNamespace(get_session_name=lambda: name or session['name'])
        worker = types.SimpleNamespace(node=types.SimpleNamespace(cluster_id=cluster or session['cluster_id'],
            address_info={'session_dir': session['session_dir'], 'gcs_address': session['gcs_address']}))
        if nodes is None:
            nodes = [{'Alive': True, 'NodeManagerAddress': self.claim[k]['ip'],
                      'NodeID': self.claim[k]['node_id'], 'Resources': {'GPU': 4}}
                     for k in ('training', 'rollout')]
        return types.SimpleNamespace(get_runtime_context=lambda: ctx, nodes=lambda: nodes,
                                     _private=types.SimpleNamespace(worker=types.SimpleNamespace(global_worker=worker)))

    def test_real_session_and_exact_nodes_are_bound(self):
        self.write_claim()
        loaded = guard.load_claim(self.env)
        guard.verify_attached_session(self.fake_ray(), loaded)
        with self.assertRaisesRegex(RuntimeError, 'session'):
            guard.verify_attached_session(self.fake_ray(name='session_other'), loaded)
        with self.assertRaisesRegex(RuntimeError, 'session'):
            guard.verify_attached_session(self.fake_ray(cluster='ClusterID(other)'), loaded)
        bad = [{'Alive': True, 'NodeManagerAddress': self.claim['training']['ip'],
                'NodeID': 'other', 'Resources': {'GPU': 4}},
               {'Alive': True, 'NodeManagerAddress': self.claim['rollout']['ip'],
                'NodeID': 'node2', 'Resources': {'GPU': 4}}]
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            guard.verify_attached_session(self.fake_ray(nodes=bad), loaded)

    def test_claim_hash_role_and_gpu_duplication_are_rejected(self):
        self.write_claim()
        self.env['Q36_RAY_CLAIM_SHA256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest'):
            guard.load_claim(self.env)
        self.claim['rollout']['gpu_uuids'][0] = self.claim['training']['gpu_uuids'][0]
        self.write_claim()
        with self.assertRaisesRegex(ValueError, 'overlap'):
            guard.load_claim(self.env)
        self.claim['rollout']['gpu_uuids'][0] = str(uuid.UUID(int=5))
        self.claim['rollout']['ip'] = '10.44.48.230'
        self.write_claim()
        with self.assertRaisesRegex(ValueError, 'prepared training and rollout nodes'):
            guard.load_claim(self.env)

    def test_claim_guard_source_is_locked_before_use(self):
        source_dir = Path(guard.__file__).resolve().parent
        with patch.dict('os.environ', {'Q36_CODE_DIR': str(source_dir)}), \
             patch.object(loader, 'GUARD_SHA256', '0' * 64):
            with self.assertRaisesRegex(RuntimeError, 'source identity'):
                loader.load_guard()
        with patch.dict('os.environ', {'Q36_CODE_DIR': str(source_dir)}):
            self.assertIs(loader.load_guard(), guard)

    def test_pool_uuid_and_node_role_must_match_claim(self):
        self.write_claim()
        claim = guard.load_claim(self.env)
        ownership = {key: [{'node_ip': role['ip'], 'node_id': role['node_id'], 'gpu_uuid': item}
                           for item in role['gpu_uuids']] for key, role in
                     ((key, claim[key]) for key in ('training', 'rollout'))}
        guard.validate_pool_ownership(ownership, claim)
        bad = copy.deepcopy(ownership)
        bad['rollout'][0]['gpu_uuid'] = ownership['training'][0]['gpu_uuid']
        with self.assertRaisesRegex(RuntimeError, 'rollout'):
            guard.validate_pool_ownership(bad, claim)
        bad = copy.deepcopy(ownership)
        bad['training'][0]['node_id'] = 'other'
        with self.assertRaisesRegex(RuntimeError, 'training'):
            guard.validate_pool_ownership(bad, claim)

    def test_monitor_launcher_and_services_are_identity_bound(self):
        session = self.root / 'session_123_4321'
        session.mkdir()
        status_path = self.root / 'ray-head.status.json'
        status_path.write_text(json.dumps({'run_id': self.run.name, 'role': 'head', 'state': 'RUNNING',
            'ray_start_pid': 4321, 'ray_start_returncode': None, 'supervisor_pid': 4322,
            'supervisor_create_time': 2.0, 'live': [{'name': 'raylet', 'pid': 10, 'create_time': 3.0},
                                                  {'name': 'gcs_server', 'pid': 11, 'create_time': 4.0}]}))
        class Process:
            def __init__(self, pid, created, parent=0): self.pid, self.created, self.parent = pid, created, parent
            def create_time(self): return self.created
            def is_running(self): return True
            def status(self): return 'running'
            def ppid(self): return self.parent
            def children(self, recursive=False): return [processes[10], processes[11]]
        processes = {4321: Process(4321, 1.0, 4322), 4322: Process(4322, 2.0),
                     10: Process(10, 3.0, 4321), 11: Process(11, 4.0, 4321)}
        psutil = types.SimpleNamespace(Process=lambda pid: processes[pid], STATUS_ZOMBIE='zombie', STATUS_DEAD='dead')
        ray = types.SimpleNamespace(util=types.SimpleNamespace(get_node_ip_address=lambda: '10.44.49.51'))
        supervisor = dict(self.claim['training']['supervisor'], status_file=str(status_path))
        with patch.dict('sys.modules', {'psutil': psutil, 'ray': ray}):
            result = guard.inspect_supervisor(self.run.name, '10.44.49.51', session.name, str(session), supervisor)
            self.assertEqual(result['service_pids'], [10, 11])
            processes[4322].created = 5.0
            with self.assertRaisesRegex(RuntimeError, 'monitor identity'):
                guard.inspect_supervisor(self.run.name, '10.44.49.51', session.name, str(session), supervisor)
            processes[4322].created = 2.0
            processes[10].created = 7.0
            with self.assertRaisesRegex(RuntimeError, 'verified descendant'):
                guard.inspect_supervisor(self.run.name, '10.44.49.51', session.name, str(session), supervisor)


if __name__ == '__main__': unittest.main()
