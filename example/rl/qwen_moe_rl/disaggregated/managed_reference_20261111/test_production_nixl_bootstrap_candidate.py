"""CPU-only real pinned manager wiring with simulated worker groups."""
import asyncio
import threading
import types
import unittest
from unittest.mock import patch

import production_nixl_bootstrap_candidate as bootstrap
from verl.utils.ray_utils import auto_await


class Group:
    def __init__(self,name,events,fail=None):
        self.name=name;self.events=events;self.fail=fail;self.world_size=1;self.prepare_delay=0;self.mutate=None
        self.incarnation='a'*32 if name=='actor' else 'b'*32;self.shutdown_started=False
        self.identity_count=0;self.identity_block_at=None;self.identity_started=threading.Event();self.identity_release=threading.Event()
        self.workers=[types.SimpleNamespace(_actor_id=types.SimpleNamespace(hex=lambda:'aa' if name=='actor' else 'bb'))]
    def update_weights(self,**kwargs):
        async def work():
            self.events.append((self.name,'update'))
            if self.fail=='update':raise RuntimeError('injected update')
            if self.mutate is not None:self.mutate()
            return {'role':self.name}
        return [work()]
    def execute_checkpoint_engine(self,methods=None,*args,**kwargs):
        if methods is None:methods=kwargs['method']
        action=methods[0];self.events.append((self.name,action))
        async def work():
            if action=='prepare' and self.prepare_delay:
                await asyncio.sleep(self.prepare_delay);self.events.append((self.name,'prepare_finished'))
            if self.fail==action:raise RuntimeError('injected '+action)
            if action=='managed_identity':
                self.identity_count+=1
                if self.identity_count==self.identity_block_at:
                    self.identity_started.set()
                    while not self.identity_release.is_set():await asyncio.sleep(.005)
                return {'node_ip':self.name,'pid':1 if self.name=='actor' else 2,'create_time':1.0,
                        'engine_nonce':self.incarnation,'agent_name':self.name,'shutdown_started':self.shutdown_started}
            if action=='shutdown_managed_context':self.shutdown_started=True
            return self.name+'-metadata' if action=='prepare' else action
        return [work()]


class Manager:
    backend='rtp_nixl_guarded'
    backend_cls=bootstrap.context.ProductionManagedNixl
    def __init__(self,loop,fail=None):
        self._background_loop=loop;self.events=[];self.actor_wg=Group('actor',self.events,fail)
        self.rollout=Group('rollout',self.events);self.fail=fail
        self.replicas=[types.SimpleNamespace(workers=self.rollout.workers)]
    @auto_await
    async def abort_replicas(self):self.events.append(('admission','pause'))
    @auto_await
    async def release_kv_cache_replicas(self):self.events.append(('kv','release'))
    @auto_await
    async def resume_kv_cache_replicas(self):self.events.append(('kv','resume'))
    @auto_await
    async def resume_generation_replicas(self):self.events.append(('admission','resume'))
    def build_process_group(self,_rollout):
        self.events.append(('topology','build'))
        if self.fail=='prepare':raise RuntimeError('injected prepare')
    def _merge_sync_metrics(self,values):return {'merged':values}


class Tests(unittest.TestCase):
    def setUp(self):
        self.loop=asyncio.new_event_loop()
        self.thread=threading.Thread(target=self.loop.run_forever,daemon=True);self.thread.start()
    def tearDown(self):
        self.loop.call_soon_threadsafe(self.loop.stop);self.thread.join(timeout=5);self.loop.close()
    def test_three_generations_through_fixed_ticket_api(self):
        manager=Manager(self.loop)
        async def run():
            for step in range(3):
                ticket=await bootstrap._managed_submit(manager,step)
                result=await ticket.wait_committed()
                self.assertEqual(result,{'merged':[{'role':'actor'}]})
            self.assertEqual(manager._q36_transaction_gate.last_step,2)
            with self.assertRaisesRegex(RuntimeError,'safely shut down'):bootstrap._managed_close(manager)
            await bootstrap.shutdown_managed_contexts(manager)
            self.assertTrue(manager._q36_transaction_gate.shutdown_completed)
            with self.assertRaises(RuntimeError):await bootstrap._managed_submit(manager,3)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        self.assertEqual(sum(x==('actor','authorize_finalize') for x in manager.events),3)
        self.assertEqual(sum(x==('rollout','authorize_finalize') for x in manager.events),3)
        self.assertEqual(sum(x==('admission','resume') for x in manager.events),3)
    def test_prepare_failure_poison_blocks_future_submit(self):
        manager=Manager(self.loop,fail='prepare')
        manager.rollout.prepare_delay=.1
        async def run():
            with self.assertRaisesRegex(RuntimeError,'prepare'):await bootstrap._managed_submit(manager,0)
            self.assertTrue(manager._q36_transaction_gate.poisoned)
            with self.assertRaises(RuntimeError):await bootstrap._managed_submit(manager,1)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        asyncio.run_coroutine_threadsafe(asyncio.sleep(.2),self.loop).result(timeout=3)
        self.assertIn(('rollout','prepare_finished'),manager.events)
        self.assertEqual(len(manager._q36_transaction_gate.active['prepare_refs']),2)
        self.assertNotIn(('rollout','init_process_group'),manager.events)
        self.assertNotIn(('admission','resume'),manager.events)
    def test_update_failure_poison_blocks_finalize_and_resume(self):
        manager=Manager(self.loop,fail='update')
        async def run():
            ticket=await bootstrap._managed_submit(manager,0)
            with self.assertRaisesRegex(RuntimeError,'update'):await ticket.wait_committed()
            self.assertTrue(manager._q36_transaction_gate.poisoned)
            with self.assertRaises(RuntimeError):await bootstrap._managed_submit(manager,1)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        self.assertNotIn(('actor','finalize'),manager.events)
        self.assertNotIn(('admission','resume'),manager.events)
    def test_shutdown_failure_prevents_further_generations(self):
        manager=Manager(self.loop)
        async def run():
            ticket=await bootstrap._managed_submit(manager,0);await ticket.wait_committed()
            manager.rollout.fail='shutdown_managed_context'
            with self.assertRaisesRegex(RuntimeError,'shutdown_managed_context'):
                await bootstrap.shutdown_managed_contexts(manager)
            self.assertTrue(manager._q36_transaction_gate.poisoned)
            self.assertTrue(manager._q36_transaction_gate.shutdown_started)
            with self.assertRaises(RuntimeError):await bootstrap._managed_submit(manager,1)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
    def test_worker_membership_change_blocks_shutdown(self):
        manager=Manager(self.loop)
        async def run():
            ticket=await bootstrap._managed_submit(manager,0);await ticket.wait_committed()
            manager.replicas[0].workers=[types.SimpleNamespace(_actor_id=types.SimpleNamespace(hex=lambda:'cc'))]
            with self.assertRaisesRegex(RuntimeError,'identities changed'):
                await bootstrap.shutdown_managed_contexts(manager)
            self.assertTrue(manager._q36_transaction_gate.poisoned)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        self.assertNotIn(('actor','shutdown_managed_context'),manager.events)
    def test_worker_membership_change_before_authorize_poisons(self):
        manager=Manager(self.loop)
        manager.rollout.mutate=lambda:setattr(manager.replicas[0],'workers',
            [types.SimpleNamespace(_actor_id=types.SimpleNamespace(hex=lambda:'cc'))])
        async def run():
            ticket=await bootstrap._managed_submit(manager,0)
            with self.assertRaisesRegex(RuntimeError,'identities changed'):await ticket.wait_committed()
            self.assertTrue(manager._q36_transaction_gate.poisoned)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        self.assertNotIn(('actor','authorize_finalize'),manager.events)
    def test_same_actor_id_but_restarted_engine_poisons(self):
        manager=Manager(self.loop)
        manager.rollout.mutate=lambda:setattr(manager.rollout,'incarnation','c'*32)
        async def run():
            ticket=await bootstrap._managed_submit(manager,0)
            with self.assertRaisesRegex(RuntimeError,'incarnation changed'):await ticket.wait_committed()
            self.assertTrue(manager._q36_transaction_gate.poisoned)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):asyncio.run(run())
        self.assertNotIn(('actor','authorize_finalize'),manager.events)
    def test_no_transaction_contexts_require_explicit_shutdown(self):
        manager=Manager(self.loop)
        with self.assertRaisesRegex(RuntimeError,'explicit shutdown'):bootstrap._managed_close(manager)
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):
            asyncio.run(bootstrap.shutdown_managed_contexts(manager))
        self.assertTrue(manager._q36_transaction_gate.shutdown_completed)
    def test_delayed_identity_after_poison_dispatches_no_new_stage(self):
        for count,forbidden in ((1,('actor','prepare')),(2,('actor','authorize_finalize')),
                                (3,('actor','finalize')),(4,('admission','resume'))):
            manager=Manager(self.loop);manager.rollout.identity_block_at=count;errors=[]
            async def run():
                try:
                    ticket=await bootstrap._managed_submit(manager,0)
                    await ticket.wait_committed()
                except BaseException as error:errors.append(error)
            with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):
                thread=threading.Thread(target=lambda:asyncio.run(run()));thread.start()
                self.assertTrue(manager.rollout.identity_started.wait(5))
                manager._q36_transaction_gate.fail(RuntimeError('injected timeout'))
                manager.rollout.identity_release.set();thread.join(5)
            self.assertFalse(thread.is_alive());self.assertTrue(errors)
            self.assertTrue(manager._q36_transaction_gate.poisoned)
            self.assertNotIn(forbidden,manager.events)
    def test_first_concurrent_updates_share_one_gate(self):
        manager=Manager(self.loop);barrier=threading.Barrier(2);results=[];first_done=threading.Event()
        manager.rollout.identity_block_at=1
        def run():
            async def update():
                try:
                    ticket=await bootstrap._managed_submit(manager,0);await ticket.wait_committed()
                    results.append('success')
                except BaseException:results.append('rejected')
                finally:first_done.set()
            barrier.wait();asyncio.run(update())
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):
            threads=[threading.Thread(target=run) for _ in range(2)]
            for t in threads:t.start()
            self.assertTrue(manager.rollout.identity_started.wait(5))
            self.assertTrue(first_done.wait(5))
            manager.rollout.identity_release.set()
            for t in threads:t.join(5)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(sorted(results),['rejected','success'])
        self.assertEqual(manager.events.count(('actor','update')),1)

    def test_first_update_and_shutdown_share_one_gate(self):
        manager=Manager(self.loop);barrier=threading.Barrier(2);results=[];first_done=threading.Event()
        manager.rollout.identity_block_at=1
        def run(kind):
            async def command():
                try:
                    if kind=='update':
                        ticket=await bootstrap._managed_submit(manager,0);await ticket.wait_committed()
                    else:await bootstrap.shutdown_managed_contexts(manager)
                    results.append((kind,'success'))
                except BaseException:results.append((kind,'rejected'))
                finally:first_done.set()
            barrier.wait();asyncio.run(command())
        with patch.object(bootstrap,'_rollout_group',lambda m:m.rollout):
            threads=[threading.Thread(target=run,args=(kind,)) for kind in ('update','shutdown')]
            for t in threads:t.start()
            self.assertTrue(manager.rollout.identity_started.wait(5))
            self.assertTrue(first_done.wait(5))
            manager.rollout.identity_release.set()
            for t in threads:t.join(5)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(len(results),2)
        self.assertLessEqual(sum(status=='success' for _,status in results),1)
        self.assertFalse(('actor','prepare') in manager.events and ('actor','shutdown_managed_context') in manager.events)


if __name__=='__main__':unittest.main()
