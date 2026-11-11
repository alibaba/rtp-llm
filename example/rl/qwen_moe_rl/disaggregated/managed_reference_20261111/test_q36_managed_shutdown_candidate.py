import asyncio
import os
import types
import unittest
from unittest.mock import patch

import q36_managed_shutdown_candidate as candidate


class Held(Exception):pass


class ManagedShutdownTest(unittest.TestCase):
    def setup_backend(self):
        events=[]
        class Backend:
            def shutdown(self):events.append('original')
        candidate.install(Backend);candidate.install(Backend)
        manager=types.SimpleNamespace(_q36_version_state={'poisoned':False},
                                      _q36_transaction_gate=types.SimpleNamespace(poisoned=False),
                                      close=lambda:events.append('close'))
        backend=Backend();backend.checkpoint_manager=manager
        async def close_contexts(got):
            self.assertIs(got,manager);events.append('contexts')
        bootstrap=types.SimpleNamespace(shutdown_managed_contexts=close_contexts)
        return backend,events,bootstrap

    def test_success_closes_contexts_then_manager_then_original(self):
        backend,events,bootstrap=self.setup_backend()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}),patch.object(candidate,'_bootstrap',return_value=bootstrap), \
             patch.object(candidate,'_receipt',side_effect=lambda *args:events.append('receipt')):
            backend.shutdown()
        self.assertEqual(events,['contexts','close','receipt','original'])

    def test_original_exception_or_poison_never_closes_or_calls_original(self):
        for mode in ('exception','poison','missing'):
            backend,events,bootstrap=self.setup_backend()
            if mode=='poison':backend.checkpoint_manager._q36_transaction_gate.poisoned=True
            if mode=='missing':backend.checkpoint_manager=None
            def hold(*args):events.append(('hold',args[2]));raise Held()
            with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}), \
                 patch.object(candidate,'_bootstrap',return_value=bootstrap),patch.object(candidate,'hold',side_effect=hold):
                if mode=='exception':
                    try:raise ValueError('original')
                    except ValueError:
                        with self.assertRaises(Held):backend.shutdown()
                else:
                    with self.assertRaises(Held):backend.shutdown()
            self.assertEqual(events,[('hold','backend_shutdown')])

    def test_context_shutdown_failure_holds_and_skips_original(self):
        backend,events,_=self.setup_backend()
        async def fail(_):events.append('contexts');raise ValueError('failure')
        def hold(*args):events.append(('hold',args[2]));raise Held()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}), \
             patch.object(candidate,'_bootstrap',return_value=types.SimpleNamespace(shutdown_managed_contexts=fail)), \
             patch.object(candidate,'hold',side_effect=hold):
            with self.assertRaises(Held):backend.shutdown()
        self.assertEqual(events,['contexts',('hold','managed_context_shutdown')])

    def test_nonmanaged_preserves_original(self):
        backend,events,_=self.setup_backend()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'0'}):backend.shutdown()
        self.assertEqual(events,['original'])

    def test_unified_constructor_failure_retains_partially_built_self(self):
        seen=[]
        class UnifiedTrainer:
            def __init__(self):
                self.backend=types.SimpleNamespace(checkpoint_manager=object())
                raise ValueError('later initialization')
            def shutdown(self):pass
        module=types.SimpleNamespace(__name__=__name__,UnifiedTrainer=UnifiedTrainer)
        candidate.install_unified(module)
        def hold(obj,error,phase):
            seen.append((obj.backend.checkpoint_manager,type(error),phase));raise Held()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}),patch.object(candidate,'hold',side_effect=hold):
            with self.assertRaises(Held):UnifiedTrainer()
        self.assertEqual(seen[0][1:],(ValueError,'unified_initialization'))

    def test_unified_shutdown_precleanup_exception_holds_trainer(self):
        seen=[]
        class UnifiedTrainer:
            def __init__(self):self.backend=types.SimpleNamespace(checkpoint_manager=object())
            def shutdown(self):raise ValueError('gateway cleanup failed')
        module=types.SimpleNamespace(__name__=__name__,UnifiedTrainer=UnifiedTrainer)
        candidate.install_unified(module);trainer=UnifiedTrainer()
        def hold(obj,error,phase):seen.append((obj,type(error),phase));raise Held()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}),patch.object(candidate,'hold',side_effect=hold):
            with self.assertRaises(Held):trainer.shutdown()
        self.assertEqual(seen,[(trainer,ValueError,'unified_shutdown')])

    def test_taskrunner_outer_holds_actual_exception(self):
        seen=[]
        class VerlTaskRunner:
            def run(self):raise RuntimeError('trainer shutdown failed')
        module=types.SimpleNamespace(__name__=__name__,VerlTaskRunner=VerlTaskRunner)
        candidate.install_outer(module);runner=VerlTaskRunner()
        def hold(obj,error,phase):seen.append((obj,type(error),phase));raise Held()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}),patch.object(candidate,'hold',side_effect=hold):
            with self.assertRaises(Held):runner.run()
        self.assertEqual(seen,[(runner,RuntimeError,'task_runner_outer')])

    def test_ray_actor_wrapper_changes_serialized_class_not_wrapper(self):
        seen=[]
        class RealActor:
            def run(self):raise ValueError('actor failure')
        class ActorWrapper:pass
        wrapper=ActorWrapper()
        wrapper.__module__=__name__
        wrapper.__ray_metadata__=types.SimpleNamespace(
            modified_class=RealActor,last_export_cluster_and_job=None,
            method_meta=types.SimpleNamespace(methods={'run':RealActor.run}))
        module=types.SimpleNamespace(__name__=__name__,VerlTaskRunner=wrapper)
        candidate.install_outer(module);candidate.install_outer(module)
        self.assertIs(wrapper.__ray_metadata__.method_meta.methods['run'],RealActor.run)
        def hold(obj,error,phase):seen.append((obj,type(error),phase));raise Held()
        obj=RealActor()
        with patch.dict(os.environ,{'Q36_USE_MANAGED_NIXL':'1'}),patch.object(candidate,'hold',side_effect=hold):
            with self.assertRaises(Held):obj.run()
        self.assertEqual(seen,[(obj,ValueError,'task_runner_outer')])

    def test_exported_ray_actor_is_rejected_before_mutation(self):
        class Actor:
            def run(self):pass
        class Wrapper:pass
        wrapper=Wrapper();wrapper.__module__=__name__
        wrapper.__ray_metadata__=types.SimpleNamespace(
            modified_class=Actor,last_export_cluster_and_job=('cluster','job'),
            method_meta=types.SimpleNamespace(methods={'run':Actor.run}))
        original=Actor.run
        with self.assertRaises(RuntimeError):
            candidate.install_outer(types.SimpleNamespace(__name__=__name__,VerlTaskRunner=wrapper))
        self.assertIs(Actor.run,original)


if __name__=='__main__':unittest.main()
