"""CPU protocol tests only; these do not qualify native RPC or CUDA IPC."""
import asyncio
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
import weakref
from unittest.mock import patch


class TransactionTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        adapter_dir = Path(os.environ['Q36_RTP_ADAPTER_SOURCE_DIR']).resolve()
        sys.modules.pop('standalone_rtp_adapter_v3', None)
        self.addCleanup(lambda: sys.modules.pop('standalone_rtp_adapter_v3', None))
        path_patch = patch.object(sys, 'path', [str(adapter_dir)] + sys.path)
        path_patch.start()
        self.addCleanup(path_patch.stop)
        env_patch = patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT': '1', 'Q36_ADAPTER_DIR': str(adapter_dir)})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        events = self.events
        self.torch = types.ModuleType('torch')
        self.torch.bfloat16 = 'bf16'
        self.torch.float32 = 'fp32'
        self.torch.uint8 = 'uint8'
        self.torch.device = lambda kind, index: f'{kind}:{index}'
        self.torch.cuda = types.SimpleNamespace(current_device=lambda: 0)
        distributed = types.ModuleType('torch.distributed')
        distributed.is_initialized = lambda: True
        distributed.get_world_size = lambda: 4
        distributed.get_rank = lambda: 0
        self.torch.distributed = distributed
        ray = types.ModuleType('ray')
        async def ensure(*args): return {'native_boot': 'boot', 'native_incarnation': 'boot-1'}
        ray.get_actor = lambda _: types.SimpleNamespace(ensure_native_running=types.SimpleNamespace(remote=ensure))
        backend = types.ModuleType('rtp_verl_backend')
        class OriginalAdapter:
            def _required_sources(self): return {'normal', 'x.linear_attn.A_log'}
            def _local_ipc_binding(self, step): return {'bound': step, 'native_incarnation': 'boot-1'}
            def _broadcast(self, obj): return obj
            def _gather(self, obj): return [obj] * 4
        backend.RTPServerAdapter = OriginalAdapter
        backend.RTPReplica = type('RTPReplica', (), {})
        native = types.ModuleType('rtp_native_server')
        self.control = None
        async def new_control(replica): return self.control
        native._new_control = new_control
        native._source = lambda: None
        native._audit_root = lambda: Path('/unused')
        native._write_json = lambda path, data: events.append(('record', data))
        topology = types.ModuleType('rtp_topology')
        topology.preserve_fp32_source = lambda name: name.endswith('A_log')
        rollout = types.ModuleType('verl.workers.rollout.base')
        rollout.BaseRollout = type('BaseRollout', (), {})
        ipc = types.ModuleType('rtp_llm.utils.cuda_ipc')
        ipc.CudaIpcV2 = types.SimpleNamespace(export_tensor=lambda value: b'local-ipc')
        digest = types.ModuleType('q36_refit_bounded_digest_candidate')
        class Workspace:
            def __init__(self, device): events.append(('workspace', device))
            def hash(self, value, name, source_digest):
                source_digest.update(name.encode())
                return hashlib.sha256(b'1234').hexdigest(), 4
            def close(self): events.append('workspace-close')
        digest.CudaDigestWorkspace = Workspace
        modules = {'torch': self.torch, 'torch.distributed': distributed, 'ray': ray,
                   'rtp_verl_backend': backend, 'rtp_native_server': native,
                   'rtp_topology': topology, 'verl.workers.rollout.base': rollout,
                   'rtp_llm.utils.cuda_ipc': ipc, 'q36_refit_bounded_digest_candidate': digest}
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        path = Path(__file__).with_name('standalone_rtp_adapter_v4.py')
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / 'adapter' / 'standalone_rtp_adapter_v4.py'
        spec = importlib.util.spec_from_file_location('standalone_rtp_adapter', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.adapter = object.__new__(self.module.StandaloneRTPAdapter)
        self.adapter.replica_rank = 0
        self.adapter.tp_size = 4
        self.adapter._last_sync_version = None
        self.adapter._poisoned = False
        self.adapter._unacknowledged_ipc = []

    def tensor(self):
        events = self.events
        class Tensor:
            device = 'cuda:0'
            dtype = 'bf16'
            shape = (2,)
            def detach(self): return self
            def to(self, dtype):
                events.append(('dtype', dtype))
                result = Tensor()
                result.dtype = dtype
                return result
            def contiguous(self): return self
            def cpu(self): return self
            def view(self, dtype): return self
            def numpy(self): return b'1234'
        return Tensor()

    def test_strict_standalone_binds_rollout_before_worker_creation(self):
        events = self.events
        class Pool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        async def parent_init(replica, resource_pool=None):
            events.append(('parent-init', resource_pool))
        parent = self.module.base.StandaloneRTPReplica
        parent.init_standalone = parent_init
        self.addCleanup(delattr, parent, 'init_standalone')
        binding = types.SimpleNamespace(_bind_claimed_pool=lambda pool, role: events.append(('bind', role, pool)))
        base = types.SimpleNamespace(RayResourcePool=Pool)
        replica = object.__new__(self.module.StandaloneRTPReplica)
        replica.nnodes = 1
        replica.gpus_per_replica_node = 4
        replica.replica_rank = 0
        with patch.dict(os.environ, {'Q36_STRICT_4PLUS4_CLAIM':'1'}), \
             patch.dict(sys.modules, {'onpolicy_standalone_candidate':binding,
                                      'verl.single_controller.ray.base':base}):
            asyncio.run(replica.init_standalone())
        self.assertEqual(events[0][:2], ('bind', 'rollout'))
        self.assertEqual(events[1][0], 'parent-init')
        self.assertIs(events[0][2], events[1][1])
        self.assertEqual(events[1][1].kwargs['process_on_nodes'], [4])

    def make_control(self, fail=False):
        events = self.events
        class Control:
            count = 0
            async def status(self): return [types.SimpleNamespace(phase='paused', model_version='v0')]
            async def begin_update(self, **kwargs): return [types.SimpleNamespace(update_id='u1')]
            async def push_weight(self, name, metadata):
                events.append(('push', name))
                if fail: raise RuntimeError('native push failed')
                self.count += 1
                return [types.SimpleNamespace(tp_rank=i, update_id='u1', received_weights=self.count) for i in range(4)]
            async def commit_update(self):
                events.append('commit')
                return [types.SimpleNamespace(global_step=1, model_version='v1', received_weights=2,
                                              applied_weights=2, tp_rank=i, tp_size=4, phase='paused') for i in range(4)]
            async def close(self): events.append('close')
        self.control = Control()

    def iterator(self):
        events, make_tensor = self.events, self.tensor
        class Weights:
            names = ['normal', 'x.linear_attn.A_log']
            index = 0
            pending = None
            previous = None
            def __aiter__(self): return self
            async def __anext__(self):
                if self.pending is not None: raise RuntimeError('next before ACK')
                if self.previous is not None and self.previous() is not None:
                    raise RuntimeError('previous source is retained across next allocation')
                if self.index == len(self.names): raise StopAsyncIteration
                name = self.names[self.index]
                self.index += 1
                self.pending = name
                value = make_tensor()
                self.previous = weakref.ref(value)
                return name, value
            def confirm_native_ack(self, name, proof):
                if name != self.pending or not proof['all_rank_acked']: raise RuntimeError('bad proof')
                events.append(('ack', name))
                self.pending = None
        return Weights()

    def test_native_ack_precedes_next_and_commit(self):
        self.assertIs(self.module.StandaloneRTPAdapter,
                      sys.modules['standalone_rtp_adapter_v3'].StandaloneRTPAdapter)
        self.assertEqual(self.module.StandaloneRTPAdapter.__module__, 'standalone_rtp_adapter_v3')
        self.make_control()
        asyncio.run(self.adapter.update_weights(self.iterator(), global_steps=1))
        self.assertLess(self.events.index(('push', 'normal')), self.events.index(('ack', 'normal')))
        self.assertLess(self.events.index(('ack', 'normal')), self.events.index(('push', 'x.linear_attn.A_log')))
        self.assertIn(('dtype', 'fp32'), self.events)
        self.assertIn('commit', self.events)
        record = next(x[1] for x in self.events if isinstance(x, tuple) and x[0] == 'record')
        self.assertEqual(record['replica_rank'], 0)
        self.assertEqual(record['native_boot'], 'boot')
        self.assertEqual(record['hf_tensor_count'], record['native_received_weights'])
        self.assertEqual([r['tp_rank'] for r in record['rank_status']], list(range(4)))
        self.assertTrue(all(r['model_version'] == record['model_version'] for r in record['rank_status']))
        self.assertGreater(record['finished_unix'], 0)

    def test_failed_native_push_never_confirms(self):
        self.make_control(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'native push failed'):
            asyncio.run(self.adapter.update_weights(self.iterator(), global_steps=1))
        self.assertTrue(self.adapter._poisoned)
        self.assertFalse(any(isinstance(item, tuple) and item[0] == 'ack' for item in self.events))
        self.assertNotIn('commit', self.events)

    def test_controller_quiescence_and_real_probe_gate(self):
        events = self.events
        replica = object.__new__(self.module.StandaloneRTPReplica)
        replica.replica_rank, replica.world_size = 0, 4
        state = types.SimpleNamespace(generation_ready=True)
        async def pause():
            events.append('pause-controller')
            state.generation_ready = False
            return {'paused': True, 'statuses': [dict(tp_rank=i, tp_size=4, phase='paused', model_version='v1') for i in range(4)]}
        async def probe(step, version):
            events.append(('probe-controller', step, version))
            state.generation_ready = True
            return {'global_step': step, 'model_version': version}
        replica._server_handle = types.SimpleNamespace(
            pause_for_training=types.SimpleNamespace(remote=pause),
            probe_synced_policy=types.SimpleNamespace(remote=probe))
        async def native_control(operation): events.append(('native-control', operation))
        replica._control = native_control
        class ReadyControl:
            async def status(self):
                return [types.SimpleNamespace(phase='running', model_version='v1', global_step=1) for _ in range(4)]
            async def close(self): pass
        self.control = ReadyControl()
        async def scenario():
            await replica.abort_all_requests()
            self.assertFalse(state.generation_ready)
            await replica.resume_generation()
            self.assertTrue(state.generation_ready)
        asyncio.run(scenario())
        self.assertEqual(events, ['pause-controller', ('native-control', 'resume'), ('probe-controller', 1, 'v1')])

    def test_actual_manager_abort_signature_closes_admission(self):
        replica = object.__new__(self.module.StandaloneRTPReplica)
        replica.replica_rank, replica.world_size = 0, 4
        state = types.SimpleNamespace(generation_ready=True, count=0)
        async def pause():
            state.generation_ready = False
            state.count += 1
            return {'paused': True, 'statuses': [dict(tp_rank=i, tp_size=4, phase='paused', model_version='v1') for i in range(4)]}
        replica._server_handle = types.SimpleNamespace(pause_for_training=types.SimpleNamespace(remote=pause))
        class Manager:
            replicas = [replica]
            async def abort_replicas(self, reject_request=False):
                await asyncio.gather(*[r.abort_all_requests(reject_request=reject_request) for r in self.replicas])
        async def scenario():
            with patch.dict(os.environ, {'Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT': '0'}):
                with self.assertRaisesRegex(RuntimeError, 'no-concurrent-rollout'):
                    await Manager().abort_replicas()
            await Manager().abort_replicas()
            self.assertFalse(state.generation_ready)
            state.generation_ready = True
            await Manager().abort_replicas(reject_request=True)
            self.assertFalse(state.generation_ready)
            with self.assertRaisesRegex(TypeError, 'boolean'):
                await replica.abort_all_requests(reject_request='yes')
        asyncio.run(scenario())
        self.assertEqual(state.count, 2)

    def test_rollout_identity_is_checked_before_native_server_launch(self):
        replica = object.__new__(self.module.StandaloneRTPReplica)
        events = []
        async def inspect(_fn): return {'gpu_uuid': 'expected'}
        replica.workers = [types.SimpleNamespace(__ray_call__=types.SimpleNamespace(remote=inspect))]
        guard = types.SimpleNamespace(load_claim=lambda: {'run': 'expected'},
            validate_role_ownership=lambda rows, role, claim: events.append(('verified', role)))
        loader = types.ModuleType('q36_4plus4_claim_loader')
        loader.load_guard = lambda: guard
        onpolicy = types.ModuleType('onpolicy_standalone_candidate')
        onpolicy._worker_gpu_identity = lambda worker: worker
        async def native_start(_self): events.append('native-start')
        with patch.dict(os.environ, {'Q36_STRICT_4PLUS4_CLAIM': '1'}), \
             patch.dict(sys.modules, {'q36_4plus4_claim_loader': loader, 'onpolicy_standalone_candidate': onpolicy}), \
             patch.object(self.module.base.StandaloneRTPReplica, 'launch_servers', native_start, create=True):
            asyncio.run(replica.launch_servers())
        self.assertEqual(events, [('verified', 'rollout'), 'native-start'])

    def test_bad_rollout_identity_never_starts_native_server(self):
        replica = object.__new__(self.module.StandaloneRTPReplica)
        async def inspect(_fn): return {'gpu_uuid': 'wrong'}
        replica.workers = [types.SimpleNamespace(__ray_call__=types.SimpleNamespace(remote=inspect))]
        def reject(*args): raise RuntimeError('wrong rollout identity')
        guard = types.SimpleNamespace(load_claim=lambda: {}, validate_role_ownership=reject)
        loader = types.ModuleType('q36_4plus4_claim_loader')
        loader.load_guard = lambda: guard
        onpolicy = types.ModuleType('onpolicy_standalone_candidate')
        onpolicy._worker_gpu_identity = lambda worker: worker
        async def forbidden(_self): self.fail('native must not start')
        with patch.dict(os.environ, {'Q36_STRICT_4PLUS4_CLAIM': '1'}), \
             patch.dict(sys.modules, {'q36_4plus4_claim_loader': loader, 'onpolicy_standalone_candidate': onpolicy}), \
             patch.object(self.module.base.StandaloneRTPReplica, 'launch_servers', forbidden, create=True):
            with self.assertRaisesRegex(RuntimeError, 'wrong rollout identity'):
                asyncio.run(replica.launch_servers())

    def test_changed_v3_source_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder).resolve()
            (fake / 'standalone_rtp_adapter_v3.py').write_text('raise AssertionError("must not import")\n')
            with patch.dict(os.environ, {'Q36_ADAPTER_DIR': str(fake)}), patch.object(sys, 'path', [str(fake)] + sys.path):
                path = Path(__file__).with_name('standalone_rtp_adapter_v4.py')
                if not path.is_file():
                    path = Path(__file__).resolve().parents[2] / 'adapter' / 'standalone_rtp_adapter_v4.py'
                spec = importlib.util.spec_from_file_location('wrong_q36_v4', path)
                module = importlib.util.module_from_spec(spec)
                with self.assertRaisesRegex(RuntimeError, 'source identity mismatch'):
                    spec.loader.exec_module(module)

    def test_probe_failure_closes_admission_preserves_error(self):
        events = self.events
        replica = object.__new__(self.module.StandaloneRTPReplica)
        replica.replica_rank, replica.world_size = 0, 4
        state = types.SimpleNamespace(generation_ready=True)
        async def pause():
            events.append('pause-controller')
            state.generation_ready = False
            raise RuntimeError('pause failed after admission closed')
        async def probe(step, version): raise RuntimeError('probe failed')
        replica._server_handle = types.SimpleNamespace(
            pause_for_training=types.SimpleNamespace(remote=pause),
            probe_synced_policy=types.SimpleNamespace(remote=probe))
        async def native_control(operation): pass
        replica._control = native_control
        class ReadyControl:
            async def status(self):
                return [types.SimpleNamespace(phase='running', model_version='v1', global_step=1) for _ in range(4)]
            async def close(self): pass
        self.control = ReadyControl()
        with self.assertRaisesRegex(RuntimeError, 'probe failed'):
            asyncio.run(replica.resume_generation())
        self.assertFalse(state.generation_ready)
        self.assertEqual(events, ['pause-controller'])

    def test_actual_controller_interface_contract(self):
        root = Path(os.environ.get('Q36_RTP_ADAPTER_SOURCE_DIR', str(Path(__file__).resolve().parents[2] / 'adapter')))
        multi_source = (root / 'rtp_multinode_server.py').read_text()
        native_source = (root / 'rtp_native_server.py').read_text()
        def methods(source, classname):
            tree = ast.parse(source)
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
            return {node.name: node for node in cls.body if isinstance(node, ast.AsyncFunctionDef)}
        actual = methods(multi_source, 'MultiNodeRTPServer')
        self.assertEqual([x.arg for x in actual['pause_for_training'].args.args], ['self'])
        self.assertEqual([x.arg for x in actual['probe_synced_policy'].args.args], ['self', 'expected_step', 'expected_version'])
        probe_text = ast.get_source_segment(multi_source, actual['probe_synced_policy'])
        pause_text = ast.get_source_segment(multi_source, actual['pause_for_training'])
        self.assertIn('self._generation_ready=True', probe_text)
        self.assertIn('self.client.close_admission', pause_text)
        base = methods(native_source, 'RTPServer')['probe_synced_policy']
        base_text = ast.get_source_segment(native_source, base)
        self.assertIn('rtp_sync_rank', base_text)
        self.assertIn('full_source_sha256', base_text)


if __name__ == '__main__': unittest.main()
