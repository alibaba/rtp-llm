"""CPU manager control contracts; fake Ray refs/native ACK, no GPU."""
import asyncio
import threading
import unittest
from production_manager_gate_candidate import ProductionManagerGate


class Group:
    def __init__(self,role,events,*,failure=None):
        self.role=role;self.events=events;self.failure=failure;self.world_size=1;self.result=None
    def update_weights(self,global_steps):
        self.events.append(('receive',global_steps));return [self.result]
    def execute_checkpoint_engine(self,methods,*args):
        method=methods[0];self.events.append((self.role,method))
        if self.failure==method:raise RuntimeError('injected '+method)
        async def done():return method
        return [asyncio.create_task(done())]


class Tests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events=[];self.gate=ProductionManagerGate()
        self.actor=Group('actor',self.events);self.rollout=Group('rollout',self.events)
        self.trainer=asyncio.get_running_loop().create_future()
        self.rollout.result=asyncio.get_running_loop().create_future()
        self.gate.reserve(0,self.actor,self.rollout)
    async def resume(self):self.events.append(('resume',))

    async def test_authorize_only_after_all_success(self):
        task=asyncio.create_task(self.gate.commit_after_prepare([self.trainer],self.resume))
        await asyncio.sleep(0);self.trainer.set_result('trainer');await asyncio.sleep(0)
        self.assertNotIn(('actor','authorize_finalize'),self.events)
        self.rollout.result.set_result('native-acked');self.assertEqual(await task,['trainer','native-acked'])
        self.assertEqual(self.events,[('receive',0),('actor','authorize_finalize'),('rollout','authorize_finalize'),
                                      ('actor','finalize'),('rollout','finalize'),('resume',)])
        self.assertFalse(self.gate.poisoned);self.assertIsNone(self.gate.active)

    async def test_worker_failure_poison_retains_sibling_and_forbids_next(self):
        task=asyncio.create_task(self.gate.commit_after_prepare([self.trainer],self.resume));await asyncio.sleep(0)
        self.trainer.set_exception(RuntimeError('trainer failure'))
        with self.assertRaises(RuntimeError):await task
        self.assertTrue(self.gate.poisoned);self.assertFalse(self.rollout.result.done())
        self.assertIs(self.gate.active['rollout_refs'][0],self.rollout.result)
        self.assertFalse(any('finalize' in str(x) for x in self.events));self.assertNotIn(('resume',),self.events)
        with self.assertRaises(RuntimeError):self.gate.reserve(1,self.actor,self.rollout)
        self.rollout.result.set_result('late-result')

    async def test_manager_cancel_does_not_cancel_transfer_tasks(self):
        task=asyncio.create_task(self.gate.commit_after_prepare([self.trainer],self.resume));await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(self.gate.poisoned)
        self.assertFalse(self.trainer.cancelled());self.assertFalse(self.rollout.result.cancelled())
        self.trainer.set_result('done');self.rollout.result.set_result('done')
        await self.gate.active['gather']

    async def test_partial_authorize_dispatch_is_retained(self):
        self.rollout.failure='authorize_finalize'
        self.trainer.set_result('trainer');self.rollout.result.set_result('native-acked')
        with self.assertRaises(RuntimeError):await self.gate.commit_after_prepare([self.trainer],self.resume)
        self.assertTrue(self.gate.poisoned);self.assertEqual(len(self.gate.active['authorization_refs']),1)
        await self.gate.active['authorization_refs'][0]
        self.assertNotIn(('actor','finalize'),self.events);self.assertNotIn(('resume',),self.events)

    async def test_resume_failure_is_poisoned(self):
        self.trainer.set_result('trainer');self.rollout.result.set_result('native-acked')
        async def fail():raise RuntimeError('resume failed')
        with self.assertRaisesRegex(RuntimeError,'resume failed'):
            await self.gate.commit_after_prepare([self.trainer],fail)
        self.assertTrue(self.gate.poisoned);self.assertEqual(self.gate.active['phase'],'failed')
        with self.assertRaises(RuntimeError):self.gate.reserve(1,self.actor,self.rollout)


class ThreadTests(unittest.TestCase):
    def test_concurrent_reserve_has_one_owner(self):
        gate=ProductionManagerGate();barrier=threading.Barrier(12);accepted=[];rejected=[]
        def runner():
            barrier.wait()
            try:accepted.append(gate.reserve(0,object(),object()))
            except RuntimeError:rejected.append(True)
        threads=[threading.Thread(target=runner) for _ in range(12)]
        for t in threads:t.start()
        for t in threads:t.join(5)
        self.assertEqual(len(accepted),1);self.assertEqual(len(rejected),11)

    def test_shutdown_reservation_blocks_transaction(self):
        gate=ProductionManagerGate();gate.begin_shutdown((('aa',),('bb',)))
        with self.assertRaises(RuntimeError):gate.reserve(0,object(),object())


if __name__=='__main__':unittest.main()
