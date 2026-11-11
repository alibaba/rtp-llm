"""Private opt-in wiring for fixed verl coupled-NIXL manager/worker APIs.

The same module must be imported in driver and each actual Ray worker before
engine construction. No changes are made until install() passes all checks.
This is a qualification candidate, not a production deployment.
"""
import asyncio
import hashlib
import inspect
import os
from pathlib import Path
import threading

import verl.checkpoint_engine.base as base
import verl_nixl_guarded as guard
import production_nixl_context_candidate as context
import production_manager_gate_candidate as gate_module
import full_hf_relay as relay

BASE_SHA='504a3dd30f6187fde200fd3bc444d8849975c0b8675cecc7797ffa0373a2b7a5'
GUARD_SHA='b14be8fa0b69bc8c33dff4ae0694beed5e34bc0849517f9a4ca6464e48191406'
CONTEXT_SHA='7d2572cc565b394cc857a9aa1b0a2a15b9c8d33c0a6ce5343aad49ae13697c53'
GATE_SHA='f3ea0447d8dc02f4583fa97b46c1e6e9b6aecf9cd9b7987aaa10596d80658e7c'
RELAY_SHA='0043b7fee1b6e80551174460cb1401aa2eb6c4216f346778c202984ab1fe55af'
_GATE_INIT_LOCK=threading.Lock()


def _manager_gate(manager):
    # More than one caller may enter through different driver event loops.
    # Publish exactly one gate before either reserves/starts shutdown.
    with _GATE_INIT_LOCK:
        gate=getattr(manager,'_q36_transaction_gate',None)
        if gate is None:
            gate=gate_module.ProductionManagerGate()
            manager._q36_transaction_gate=gate
        return gate


def _check(module,expected):
    path=Path(module.__file__)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('unqualified manager/worker source: '+str(path))


def _rollout_group(manager):
    workers=[]
    for replica in manager.replicas:workers.extend(replica.workers)
    return base.RayWorkerGroup(worker_handles=workers,ray_cls_with_init=base.RayClassWithInitArgs(cls=base._worker_cls))


def _participants(manager):
    def ids(workers):
        result=[]
        for worker in workers:
            value=worker._actor_id.hex()
            if not isinstance(value,str) or not value or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError('invalid Ray actor identity')
            result.append(value)
        return tuple(result)
    trainers=ids(manager.actor_wg.workers)
    if len(trainers)!=manager.actor_wg.world_size:raise RuntimeError('trainer actor count changed')
    rollouts=ids([worker for replica in manager.replicas for worker in replica.workers])
    return trainers,rollouts


async def _build_managed_process_group(manager,rollout,tx,gate):
    actor=manager.actor_wg
    await _verify_worker_incarnations(manager,rollout,gate,tx,'identity_prepare')
    tx['prepare_refs']=[]
    def prepare():
        tx['prepare_refs'].extend(actor.execute_checkpoint_engine(['prepare']*actor.world_size))
        tx['prepare_refs'].extend(rollout.execute_checkpoint_engine(['prepare']*rollout.world_size))
    gate.dispatch(tx,prepare)
    tx['prepare_gather']=asyncio.gather(*tx['prepare_refs'])
    metadata=await asyncio.shield(tx['prepare_gather'])
    if gate.poisoned:raise RuntimeError('manager poisoned during prepare')
    actor_kwargs,rollout_kwargs=manager.backend_cls.build_topology(actor.world_size,rollout.world_size,metadata)
    for values,size in ((actor_kwargs,actor.world_size),(rollout_kwargs,rollout.world_size)):
        if any(len(v)!=size for v in values.values()):raise RuntimeError('invalid NIXL topology size')
        values['method']=['init_process_group']*size
    tx['init_refs']=[]
    def initialize():
        tx['init_refs'].extend(actor.execute_checkpoint_engine(**actor_kwargs))
        tx['init_refs'].extend(rollout.execute_checkpoint_engine(**rollout_kwargs))
    gate.dispatch(tx,initialize)
    tx['init_gather']=asyncio.gather(*tx['init_refs'])
    await asyncio.shield(tx['init_gather'])
    if gate.poisoned:raise RuntimeError('manager poisoned during topology init')


async def _verify_worker_incarnations(manager,rollout,gate,holder,prefix,expected_shutdown=False):
    gate.bind_participants(_participants(manager))
    refs=[];holder[prefix+'_refs']=refs
    refs.extend(gate._command(manager.actor_wg,'managed_identity'))
    refs.extend(gate._command(rollout,'managed_identity'))
    holder[prefix+'_gather']=asyncio.gather(*refs)
    rows=await asyncio.shield(holder[prefix+'_gather'])
    gate.bind_worker_incarnations(rows,expected_shutdown=expected_shutdown)


async def _await_manager_action(method):
    # Pinned verl auto_await inspects its immediate caller: call from an async
    # frame, never from the synchronous dispatch lambda.
    return await method()


async def _quiesce_and_build(manager,rollout,tx,gate):
    tx['abort_task']=gate.dispatch(tx,lambda:asyncio.create_task(_await_manager_action(manager.abort_replicas)))
    await asyncio.shield(tx['abort_task'])
    tx['release_task']=gate.dispatch(tx,lambda:asyncio.create_task(_await_manager_action(manager.release_kv_cache_replicas)))
    await asyncio.shield(tx['release_task'])
    await _build_managed_process_group(manager,rollout,tx,gate)


async def _managed_submit(self,global_steps,rollout_quiesced=False):
    if self.backend!='rtp_nixl_guarded':
        return await self.__class__._q36_original_submit_update(self,global_steps,rollout_quiesced)
    if self.backend_cls is not context.ProductionManagedNixl or self.backend_cls.support_publish_weights:
        raise RuntimeError('managed coupled NIXL backend required')
    gate=_manager_gate(self)
    try:gate.bind_participants(_participants(self))
    except BaseException:
        gate.poison_without_active()
        raise
    rollout=_rollout_group(self)
    tx=gate.reserve(global_steps,self.actor_wg,rollout)
    try:
        tx['prepare_future']=asyncio.run_coroutine_threadsafe(
            _quiesce_and_build(self,rollout,tx,gate),self._background_loop)
        tx['prepare_waiter']=asyncio.wrap_future(tx['prepare_future'])
        await asyncio.wait_for(asyncio.shield(tx['prepare_waiter']),timeout=1800)
        def start_trainers():
            tx['trainer_refs']=self.actor_wg.update_weights(global_steps=global_steps,mode=self.backend)
        gate.dispatch(tx,start_trainers)
        async def resume():
            tx['resume_kv_task']=gate.dispatch(tx,lambda:asyncio.create_task(_await_manager_action(self.resume_kv_cache_replicas)))
            await asyncio.shield(tx['resume_kv_task'])
            tx['resume_generation_task']=gate.dispatch(tx,lambda:asyncio.create_task(_await_manager_action(self.resume_generation_replicas)))
            await asyncio.shield(tx['resume_generation_task'])
        def merge(results):return self._merge_sync_metrics(results[:self.actor_wg.world_size])
        async def verify():
            await _verify_worker_incarnations(self,rollout,gate,tx,'identity_'+tx['phase'])
        tx['background_future']=asyncio.run_coroutine_threadsafe(
            gate.commit_after_prepare(tx['trainer_refs'],resume,merge,verify),self._background_loop)
        return base.WeightUpdateTicket(version=global_steps,trainer_published=tx['trainer_refs'],
                                       rollout_committed=tx['background_future'])
    except BaseException as error:
        gate.fail(error)
        raise


def _managed_close(self):
    gate=getattr(self,'_q36_transaction_gate',None)
    if (gate is None and getattr(self,'backend',None)=='rtp_nixl_guarded' and
            getattr(self,'backend_cls',None) is context.ProductionManagedNixl):
        raise RuntimeError('managed worker contexts require explicit shutdown')
    if gate is not None and (gate.poisoned or gate.active is not None or not gate.shutdown_completed):
        raise RuntimeError('cannot close before managed contexts safely shut down')
    if gate is not None:
        try:gate.bind_participants(_participants(self))
        except BaseException:
            gate.poison_without_active()
            raise
    return self.__class__._q36_original_close(self)


async def shutdown_managed_contexts(manager):
    gate=_manager_gate(manager)
    try:
        gate.begin_shutdown(_participants(manager))
        rollout=_rollout_group(manager)
        gate.shutdown_state={}
        async def on_background():
            await _verify_worker_incarnations(manager,rollout,gate,gate.shutdown_state,'before_shutdown')
            if gate.poisoned:raise RuntimeError('manager poisoned before shutdown')
            def shutdown():
                gate.shutdown_refs.extend(gate._command(manager.actor_wg,'shutdown_managed_context'))
                gate.shutdown_refs.extend(gate._command(rollout,'shutdown_managed_context'))
            gate.dispatch_shutdown(shutdown)
            gate.shutdown_gather=asyncio.gather(*gate.shutdown_refs)
            await asyncio.shield(gate.shutdown_gather)
            await _verify_worker_incarnations(manager,rollout,gate,gate.shutdown_state,'after_shutdown',True)
            if gate.poisoned:raise RuntimeError('manager poisoned during shutdown')
            gate.shutdown_completed=True
        gate.shutdown_future=asyncio.run_coroutine_threadsafe(on_background(),manager._background_loop)
        gate.shutdown_waiter=asyncio.wrap_future(gate.shutdown_future)
        await asyncio.wait_for(asyncio.shield(gate.shutdown_waiter),timeout=1800)
    except BaseException:
        gate.poison_without_active()
        raise


def _managed_add_replicas(self,replicas):
    if self.backend=='rtp_nixl_guarded' and self.backend_cls is context.ProductionManagedNixl:
        raise RuntimeError('elastic replica changes are forbidden by managed profile')
    return self.__class__._q36_original_add_replicas(self,replicas)


def _managed_remove_replicas(self,replicas):
    if self.backend=='rtp_nixl_guarded' and self.backend_cls is context.ProductionManagedNixl:
        raise RuntimeError('elastic replica changes are forbidden by managed profile')
    return self.__class__._q36_original_remove_replicas(self,replicas)


def install():
    if (os.environ.get('Q36_USE_MANAGED_NIXL')!='1' or
        os.environ.get('Q36_ONPOLICY_NO_CONCURRENT_ROLLOUT')!='1'):
        raise RuntimeError('managed NIXL requires explicit on-policy profile')
    for module,sha in ((base,BASE_SHA),(guard,GUARD_SHA),(context,CONTEXT_SHA),(gate_module,GATE_SHA),(relay,RELAY_SHA)):
        _check(module,sha)
    if guard.tensor_digest is not relay.tensor_digest:
        raise RuntimeError('guard does not reference the qualified relay')
    cls=base.CheckpointEngineManager
    existing=base.CheckpointEngineRegistry.get('rtp_nixl_guarded')
    if existing not in (guard.RtpGuardedNixl,context.ProductionManagedNixl):
        raise RuntimeError('unexpected existing checkpoint engine registration')
    if hasattr(cls,'_q36_original_submit_update'):
        if (cls.submit_update is not _managed_submit or cls.close is not _managed_close or
                cls.add_replicas is not _managed_add_replicas or cls.remove_replicas is not _managed_remove_replicas or
                existing is not context.ProductionManagedNixl):
            raise RuntimeError('inconsistent prior manager installation')
        return
    if not inspect.iscoroutinefunction(cls.submit_update) or cls.submit_update.__module__!=base.__name__:
        raise RuntimeError('unexpected manager submit implementation')
    cls._q36_original_submit_update=cls.submit_update
    cls._q36_original_close=cls.close
    cls._q36_original_add_replicas=cls.add_replicas
    cls._q36_original_remove_replicas=cls.remove_replicas
    cls.submit_update=_managed_submit
    cls.close=_managed_close
    cls.add_replicas=_managed_add_replicas
    cls.remove_replicas=_managed_remove_replicas
    base.CheckpointEngineRegistry.register('rtp_nixl_guarded')(context.ProductionManagedNixl)
