"""Threaded 4-receiver ring regression; mocks transport, not GPU/NIXL."""
import asyncio
import importlib.util
import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_candidate():
    torch=types.ModuleType('torch')
    torch.cuda=types.SimpleNamespace(synchronize=lambda:None)
    base=types.ModuleType('verl.checkpoint_engine.base')
    base.CheckpointEngineRegistry=types.SimpleNamespace(register=lambda _name:lambda cls:cls)
    upstream=types.ModuleType('verl.checkpoint_engine.nixl_checkpoint_engine')
    upstream.NIXLCheckpointEngine=type('NIXLCheckpointEngine',(),{})
    package=types.ModuleType('verl.checkpoint_engine');package.nixl_checkpoint_engine=upstream
    relay=types.ModuleType('full_hf_relay');relay.tensor_digest=lambda _value:None
    modules={'torch':torch,'verl':types.ModuleType('verl'),'verl.checkpoint_engine':package,
             'verl.checkpoint_engine.base':base,'verl.checkpoint_engine.nixl_checkpoint_engine':upstream,
             'full_hf_relay':relay}
    with patch.dict(sys.modules,modules):
        source=Path(__file__).with_name('verl_nixl_guarded.py')
        if not source.is_file():
            source=Path(__file__).resolve().parents[2]/'disaggregated'/'verl_nixl_guarded.py'
        spec=importlib.util.spec_from_file_location('private_nixl_ring',source)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RingTest(unittest.TestCase):
    def _run_ring(self, legacy):
        module=load_candidate()
        queues=[queue.Queue() for _ in range(4)]
        barrier=threading.Barrier(4, timeout=1.5)
        events=[];errors=[];lock=threading.Lock()

        class Read:
            def __init__(self,agent,previous,local,size):self.agent=agent;self.local=local
            async def read_metadata(self):
                item=await asyncio.to_thread(queues[self.agent].get,True,3)
                self.remote,self.ack,self.meta=item
                return self.meta
            def begin_read(self):self.local[:]=self.remote
            async def wait_for_complete(self):self.ack.set()

        class Readable:
            def __init__(self,agent,next_agent,local,meta):
                self.ack=threading.Event()
                queues[next_agent].put((local,self.ack,meta))
            async def wait_for_complete(self):
                if not await asyncio.to_thread(self.ack.wait,3):raise TimeoutError('downstream read')

        module.upstream.ReadOperation=Read;module.upstream.ReadableOperation=Readable
        meta0={'is_last':False,'bucket_meta':{'a':types.SimpleNamespace(offset=0,chunk_size=4),
                                                'b':types.SimpleNamespace(offset=4,chunk_size=4)}}
        meta1={'is_last':True,'bucket_meta':{'c':types.SimpleNamespace(offset=0,chunk_size=4)}}
        for buf,meta in ((bytearray(b'aaaabbbb'),meta0),(bytearray(b'ccccxxxx'),meta1)):
            queues[0].put((buf,threading.Event(),meta))

        async def old_order(engine):
            send,recv=engine.send_buf,engine.recv_buf
            sd,rd=engine.send_descs,engine.recv_descs
            first=Read(engine.agent,engine.prev_agent,rd,engine.bucket_size)
            metadata=await first.read_metadata();first.begin_read();await first.wait_for_complete()
            send,recv=recv,send;sd,rd=rd,sd
            while not metadata['is_last']:
                outbound=Readable(engine.agent,engine.next_agent,sd,metadata) if engine.next_agent is not None else None
                next_read=Read(engine.agent,engine.prev_agent,rd,engine.bucket_size)
                following=await next_read.read_metadata();next_read.begin_read()
                for meta in metadata['bucket_meta'].values():yield meta,send[meta.offset:meta.offset+meta.chunk_size]
                if outbound is not None:await outbound.wait_for_complete()
                await next_read.wait_for_complete();metadata=following;send,recv=recv,send;sd,rd=rd,sd
            if engine.next_agent is not None:Readable(engine.agent,engine.next_agent,sd,metadata)
            for meta in metadata['bucket_meta'].values():yield meta,send[meta.offset:meta.offset+meta.chunk_size]

        async def actor(rank):
            engine=module.RtpGuardedNixl.__new__(module.RtpGuardedNixl)
            engine.agent=rank;engine.prev_agent='source' if rank==0 else rank-1
            engine.next_agent=rank+1 if rank<3 else None
            # Like a GPU tensor slice, memoryview retains its underlying
            # mutable bucket rather than silently copying on slicing.
            engine.send_buf=memoryview(bytearray(8));engine.recv_buf=memoryview(bytearray(8))
            engine.send_descs=engine.send_buf;engine.recv_descs=engine.recv_buf;engine.bucket_size=8
            generator=old_order(engine) if legacy else engine._receive_weight_chunks()
            async for meta,value in generator:
                before=bytes(value)
                barrier.wait()
                if bytes(value)!=before:raise AssertionError('buffer changed before ACK')
                with lock:events.append((rank,before))

        def worker(rank):
            try:asyncio.run(actor(rank))
            except BaseException as exc:
                with lock:errors.append((rank,type(exc).__name__))
        threads=[threading.Thread(target=worker,args=(rank,),daemon=True) for rank in range(4)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(6)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        return events,errors

    def test_original_order_stalls_before_all_rank_first_tensor(self):
        events,errors=self._run_ring(legacy=True)
        self.assertTrue(errors)
        self.assertFalse(events)

    def test_forward_then_yield_reaches_all_rank_ack_for_multiple_buckets(self):
        events,errors=self._run_ring(legacy=False)
        self.assertEqual(errors,[])
        for payload in (b'aaaa',b'bbbb',b'cccc'):
            self.assertEqual({rank for rank,value in events if value==payload},{0,1,2,3})
        self.assertEqual(len(events),12)


if __name__=='__main__':unittest.main()
