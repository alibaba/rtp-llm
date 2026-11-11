"""Mocked NIXL lease/timeout contracts. Does not claim NIXL/GPU coverage."""
import asyncio
import importlib.util
import sys
import types
import unittest
import gc
import weakref
from unittest.mock import patch


def load_plugin():
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda fn: fn)
    class Dtype:
        def __init__(self, size): self.itemsize = size
    torch.uint8, torch.bfloat16, torch.float32 = Dtype(1), Dtype(2), Dtype(4)
    base = types.ModuleType("verl.checkpoint_engine.base")
    class Registry:
        @staticmethod
        def register(_name): return lambda cls: cls
    base.CheckpointEngineRegistry = Registry
    upstream = types.ModuleType("verl.checkpoint_engine.nixl_checkpoint_engine")
    upstream.NIXLCheckpointEngine = type("NIXLCheckpointEngine", (), {})
    package = types.ModuleType("verl.checkpoint_engine")
    package.nixl_checkpoint_engine = upstream
    modules = {"torch":torch, "verl":types.ModuleType("verl"),
               "verl.checkpoint_engine":package, "verl.checkpoint_engine.base":base,
               "verl.checkpoint_engine.nixl_checkpoint_engine":upstream}
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("test_private_nixl", "verl_nixl_guarded_partial_lease.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class NixlContracts(unittest.TestCase):
    def setUp(self):
        self.module = load_plugin()
        self.engine = self.module.RtpGuardedNixl.__new__(self.module.RtpGuardedNixl)
        e = self.engine
        e.active, e.poisoned, e.finished = True, False, False
        e.pending = e.pending_name = e.pending_step = None
        e.partial_merge_value = None
        e.partial_chunks = []
        e.protocol_operations = []
        e.source_values = []
        e.native_acknowledged = False
        e.native_binding = None
        e.inflight_task = None
        e.bucket_size = 64
        e.max_tensor_bytes = 64
        e.timeout_seconds = 1
        e.transaction_timeout_seconds = 5
        self.module.tensor_digest = lambda _value: "0" * 64
        self.proof = dict(name="w", global_step=3, native_update_id="u", native_incarnation="i",
                          replica_world_size=2, acknowledged_ranks=[0, 1], all_rank_acked=True)

    def use_one_chunk(self):
        dtype = self.module.torch.bfloat16
        meta = types.SimpleNamespace(name=self.module._wire_name("w", "0" * 64), dtype=dtype,
                                     shape=(2,), chunk_offset=0, chunk_size=4)
        class Chunk:
            is_cuda = True
            device = "cuda"
            def __init__(self, dtype): self.dtype = dtype
            def is_contiguous(self): return True
            def numel(self): return 4
            def view(self, arg): return self
        chunk = Chunk(self.module.torch.uint8)
        async def chunks():
            yield meta, chunk
        self.engine._receive_weight_chunks = chunks

    def test_native_ack_gates_generator_advance(self):
        self.use_one_chunk()
        async def run():
            iterator = self.engine.receive_weights(3)
            name, value = await iterator.__anext__()
            self.assertEqual(name, "w")
            self.assertIs(self.engine.pending, value)
            with self.assertRaises(RuntimeError): await iterator.__anext__()
        asyncio.run(run())
        self.assertTrue(self.engine.poisoned)
        self.assertIsNotNone(self.engine.pending)

    def test_verified_ack_releases_and_finishes(self):
        self.use_one_chunk()
        async def run():
            iterator = self.engine.receive_weights(3)
            name, _ = await iterator.__anext__()
            iterator.confirm_native_ack(name, self.proof)
            with self.assertRaises(StopAsyncIteration): await iterator.__anext__()
        asyncio.run(run())
        self.assertTrue(self.engine.finished)
        self.assertIsNone(self.engine.pending)

    def test_oversized_tensor_is_rejected_before_reassembly(self):
        self.use_one_chunk()
        self.engine.max_tensor_bytes = 3
        async def run():
            with self.assertRaises(RuntimeError): await self.engine.receive_weights(3).__anext__()
        asyncio.run(run())
        self.assertTrue(self.engine.poisoned)

    def test_trainer_digest_mismatch_is_rejected(self):
        self.use_one_chunk()
        self.module.tensor_digest = lambda _value: "1" * 64
        async def run():
            with self.assertRaises(RuntimeError): await self.engine.receive_weights(3).__anext__()
        asyncio.run(run())
        self.assertTrue(self.engine.poisoned)

    def test_timeout_preserves_uncancelled_nixl_task(self):
        async def never_finishes():
            await asyncio.Event().wait()
            yield None
        self.engine._receive_weight_chunks = never_finishes
        self.engine.timeout_seconds = 0.01
        async def run():
            with self.assertRaises(asyncio.TimeoutError): await self.engine.receive_weights(3).__anext__()
            task = self.engine.inflight_task
            self.assertIsNotNone(task)
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError): await task
        asyncio.run(run())
        self.assertTrue(self.engine.poisoned)

    def test_read_operation_is_retained_when_init_fails(self):
        class Partial:
            def __init__(self):
                self.handle=object()
                raise RuntimeError('after beginning read')
        with self.assertRaises(RuntimeError):self.engine._new_operation(Partial)
        obj=self.engine.protocol_operations[0]
        ref=weakref.ref(obj);del obj;gc.collect()
        self.assertIsNotNone(ref());self.assertEqual(len(self.engine.protocol_operations),1)
        with self.assertRaises(RuntimeError):self.engine.finalize()

    def test_chunk_view_retained_after_failed_digest(self):
        self.use_one_chunk();self.module.tensor_digest=lambda _value:'1'*64
        async def run():
            with self.assertRaises(RuntimeError):await self.engine.receive_weights(3).__anext__()
        asyncio.run(run());gc.collect()
        self.assertEqual(len(self.engine.partial_chunks),1)
        self.assertTrue(self.engine.poisoned)
        with self.assertRaises(RuntimeError):self.engine.finalize()

    def test_partial_merge_and_inflight_task_survive_timeout_and_gc(self):
        e=self.engine;e.bucket_size=4;e.max_tensor_bytes=16;e.timeout_seconds=.01
        dtype=self.module.torch.bfloat16
        meta=types.SimpleNamespace(name=self.module._wire_name('w','0'*64),dtype=dtype,
                                   shape=(4,),chunk_offset=0,chunk_size=4)
        class Chunk:
            is_cuda=True;device='cuda';dtype=self.module.torch.uint8
            def is_contiguous(self):return True
            def numel(self):return 4
        class Buffer:
            def view(self,*_):return self
            def __getitem__(self,_):return self
            def copy_(self,_):pass
        ref=[]
        def empty(*_args,**_kw):
            b=Buffer();ref.append(weakref.ref(b));return b
        self.module.torch.empty=empty
        async def chunks():
            yield meta,Chunk()
            await asyncio.Event().wait()
        e._receive_weight_chunks=chunks
        loop=asyncio.new_event_loop()
        try:
            with self.assertRaises(asyncio.TimeoutError):loop.run_until_complete(e.receive_weights(3).__anext__())
            gc.collect()
            self.assertIs(ref[0](),e.partial_merge_value)
            self.assertEqual(len(e.partial_chunks),1)
            self.assertIsNotNone(e.inflight_task);self.assertFalse(e.inflight_task.done())
            with self.assertRaises(RuntimeError):e.finalize()
            # Explicit cancellation belongs only to this isolated mock test.
            e.inflight_task.cancel()
            with self.assertRaises(asyncio.CancelledError):loop.run_until_complete(e.inflight_task)
        finally:loop.close()

    def test_submitted_read_handle_survives_wait_exception(self):
        class Handle:pass
        refs=[]
        class Read:
            def __init__(self,*_):self.xfer_handle=None
            async def read_metadata(self):return {'bucket_meta':{},'is_last':False}
            def begin_read(self):
                self.xfer_handle=Handle();refs.append(weakref.ref(self.xfer_handle))
            async def wait_for_complete(self):raise RuntimeError('injected after READ submission')
        self.module.upstream.ReadOperation=Read
        e=self.engine;e.prev_agent='upstream';e.agent=object();e.send_buf=e.recv_buf=object()
        e.send_descs=e.recv_descs=object()
        async def run():
            with self.assertRaisesRegex(RuntimeError,'after READ submission'):
                await e._receive_weight_chunks().__anext__()
        asyncio.run(run());gc.collect()
        self.assertIsNotNone(refs[0]());self.assertIs(e.protocol_operations[0].xfer_handle,refs[0]())
        with self.assertRaises(RuntimeError):e.finalize()

    def test_sender_synchronizes_before_exporter_reuses_same_buffer(self):
        events=[];shared=types.SimpleNamespace(value=0)
        self.module.torch.cuda=types.SimpleNamespace(synchronize=lambda:events.append(('sync',shared.value)))
        self.engine.rank=-1
        def exporter():
            shared.value=1;events.append(('produce',1));yield 'a',shared
            shared.value=2;events.append(('produce',2));yield 'b',shared
        stream=self.engine._verified_sources(exporter())
        self.assertIs(next(stream)[1],shared)
        self.assertIs(next(stream)[1],shared)
        self.assertEqual(events,[('produce',1),('sync',1),('produce',2)])
        with self.assertRaises(StopIteration):next(stream)
        self.assertEqual(events[-1],('sync',2));self.assertEqual(self.engine.source_values,[])


if __name__ == "__main__": unittest.main()
