import importlib.util
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


path = Path(__file__).with_name('onpolicy_standalone_candidate.py')
spec = importlib.util.spec_from_file_location('onpolicy_standalone_candidate', path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class OnPolicyStandaloneTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events

        class Backend:
            def _init_colocated_workers(self):
                raise AssertionError('old initializer used')

        class Role:
            Actor = 'actor'
            ActorRollout = 'actor_rollout'
            ActorRolloutRef = 'actor_rollout_ref'

        class PoolManager:
            def create_resource_pool(self): events.append('pools')
            def get_resource_pool(self, role):
                events.append(('pool', role))
                return 'training-pool'

        class Actor:
            workers = ['train-worker']
            def init_model(self): events.append('actor-init')

        def worker(**kwargs):
            events.append(('worker', kwargs['role']))
            return kwargs

        class Group:
            def __init__(self, **kwargs):
                events.append(('group-pool', kwargs['resource_pool']))
            def spawn(self, prefix_set):
                return {next(iter(prefix_set)): Actor()}

        class Server:
            @classmethod
            def create(cls, **kwargs):
                events.append(('server', kwargs['worker_group'], kwargs['rollout_resource_pool']))
                return cls()
            def get_client(self): return 'client'
            def get_replicas(self): return [NS(workers=['roll-worker'])]

        class Agent:
            @classmethod
            def create(cls, **kwargs):
                events.append(('agent', kwargs['llm_client']))
                return cls()

        class Checkpoint:
            def __init__(self, **kwargs):
                events.append(('checkpoint', len(kwargs['replicas'])))
            def sleep_replicas(self): events.append('sleep')
            async def update_weights(self, global_steps=None): events.append(('update', global_steps))
            async def abort_replicas(self): events.append('abort')

        self.module = NS(VerlBackend=Backend, Role=Role, RayClassWithInitArgs=worker,
                         create_colocated_worker_cls=lambda class_dict: class_dict,
                         build_wg_kwargs=lambda config, device: {}, LLMServerManager=Server,
                         AgentLoopManager=Agent, omega_conf_to_dataclass=lambda x: x,
                         CheckpointEngineManager=Checkpoint)
        self.backend = Backend()
        self.backend.config = NS(actor_rollout_ref=NS(rollout=NS(checkpoint_engine=NS(backend='nccl'))))
        self.backend.is_separated = False
        self.backend.use_reference_policy = False
        self.backend.ref_in_actor = False
        self.backend.resource_pool_manager = PoolManager()
        self.backend.role_worker_mapping = {Role.ActorRollout: object()}
        self.backend.ray_worker_group_cls = Group
        self.backend.device_name = 'cuda'
        self.original_audit = candidate.audit_gpu_pools
        candidate.audit_gpu_pools = lambda *args, **kwargs: {'training': [], 'rollout': []}
        self.addCleanup(setattr, candidate, 'audit_gpu_pools', self.original_audit)
        environment = patch.dict('os.environ', {'Q36_DISAGG_TRAIN_GPUS': '1',
                                                 'Q36_DISAGG_ROLLOUT_GPUS': '1',
                                                 'Q36_DISAGG_ALLOWED_NODE_IPS': '10.0.0.1'})
        environment.start()
        self.addCleanup(environment.stop)
        candidate.install(self.module)

    def test_actor_only_and_standalone_manager(self):
        self.backend._init_colocated_workers()
        self.assertIn(('worker', 'actor'), self.events)
        self.assertIn(('group-pool', 'training-pool'), self.events)
        self.assertIn(('server', None, None), self.events)
        self.assertEqual(self.events[-1], 'sleep')

    def test_guards_fail_before_allocating(self):
        self.backend.config.actor_rollout_ref.rollout.checkpoint_engine.backend = 'naive'
        with self.assertRaises(ValueError): self.backend._init_colocated_workers()
        self.assertEqual(self.events, [])

    def test_strict_training_identity_checked_before_model_or_native_start(self):
        loader = NS(load_guard=lambda: guard)
        guard = NS(load_claim=lambda: {'claim': 'frozen'},
                   validate_role_ownership=lambda rows, role, claim: self.events.append('trainer-identity'),
                   validate_pool_ownership=lambda ownership, claim: self.events.append('pool-identity'))
        with patch.dict('os.environ', {'Q36_STRICT_4PLUS4_CLAIM': '1'}), \
             patch.dict(sys.modules, {'q36_4plus4_claim_loader': loader}), \
             patch.object(candidate, '_collect_gpu_identities', lambda workers: [{'gpu_uuid': 'claimed'}]):
            self.backend._init_colocated_workers()
        self.assertLess(self.events.index('trainer-identity'), self.events.index('actor-init'))
        self.assertLess(self.events.index('actor-init'), self.events.index(('server', None, None)))
        self.assertIn('pool-identity', self.events)

    def test_bad_training_identity_never_starts_model(self):
        def reject(*args): raise RuntimeError('wrong GPU assignment')
        guard = NS(load_claim=lambda: {}, validate_role_ownership=reject)
        loader = NS(load_guard=lambda: guard)
        with patch.dict('os.environ', {'Q36_STRICT_4PLUS4_CLAIM': '1'}), \
             patch.dict(sys.modules, {'q36_4plus4_claim_loader': loader}), \
             patch.object(candidate, '_collect_gpu_identities', lambda workers: []):
            with self.assertRaisesRegex(RuntimeError, 'wrong GPU'):
                self.backend._init_colocated_workers()
        self.assertNotIn('actor-init', self.events)
        self.assertNotIn(('server', None, None), self.events)

    def test_install_is_idempotent(self):
        method = self.module.VerlBackend._init_colocated_workers
        self.assertIs(candidate.install(self.module), method)

    def test_overlap_fails_before_sync(self):
        def collect(workers):
            return [{'node_id': 'node1', 'node_ip': '10.0.0.1', 'gpu_id': '0',
                     'gpu_uuid': 'GPU-aaa'} for _ in workers]
        with self.assertRaisesRegex(RuntimeError, 'overlapping'):
            self.original_audit(['trainer'], [NS(workers=['rollout'])], collect=collect,
                                allowed_node_ips={'10.0.0.1'}, expected_train=1, expected_rollout=1)

    def test_unknown_node_fails(self):
        def collect(workers):
            return [{'node_id': workers[0], 'node_ip': '10.0.0.2', 'gpu_id': '0',
                     'gpu_uuid': 'GPU-' + workers[0]}]
        with self.assertRaisesRegex(RuntimeError, 'outside'):
            self.original_audit(['trainer'], [NS(workers=['rollout'])], collect=collect,
                                allowed_node_ips={'10.0.0.1'}, expected_train=1, expected_rollout=1)

    def test_version_failure_poison_prevents_retry(self):
        events = self.events
        class Manager:
            async def update_weights(self, global_steps=None):
                events.append(('sync', global_steps))
                if global_steps == 1: raise RuntimeError('injected failure')
            async def abort_replicas(self): events.append('abort')
        manager = Manager()
        candidate.guard_weight_versions(manager)
        async def scenario():
            await manager.update_weights(global_steps=0)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                await manager.update_weights(global_steps=1)
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                await manager.update_weights(global_steps=1)
        asyncio.run(scenario())
        self.assertEqual(events, [('sync', 0), ('sync', 1), 'abort'])


if __name__ == '__main__': unittest.main()
