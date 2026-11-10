"""Mocked NIXL lease/timeout contracts. Does not claim NIXL/GPU coverage."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


RUNTIME_DIR = Path(__file__).resolve().parents[2] / "disaggregated"
sys.path.insert(0, str(RUNTIME_DIR))


def load_plugin():
    torch = types.ModuleType("torch")
    class Dtype:
        def __init__(self, size): self.itemsize = size
    torch.uint8, torch.bfloat16, torch.float32 = Dtype(1), Dtype(2), Dtype(4)
    base = types.ModuleType("verl.checkpoint_engine.base")
    class Registry:
        @staticmethod
        def register(_name): return lambda cls: cls
    base.CheckpointEngineRegistry = Registry
    upstream = types.ModuleType("verl.checkpoint_engine.nixl_checkpoint_engine")
    class PinnedTopology:
        @classmethod
        def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
            return ({"rank": [0] + [-1] * (actor_wg_world_size - 1)}, {
                "next_agent_metadata": metadata[-rollout_world_size + 1:] + [None]})
    upstream.NIXLCheckpointEngine = PinnedTopology
    package = types.ModuleType("verl.checkpoint_engine")
    package.nixl_checkpoint_engine = upstream
    modules = {"torch":torch, "verl":types.ModuleType("verl"),
               "verl.checkpoint_engine":package, "verl.checkpoint_engine.base":base,
               "verl.checkpoint_engine.nixl_checkpoint_engine":upstream}
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("test_private_nixl", RUNTIME_DIR / "verl_nixl_guarded.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class NixlContracts(unittest.TestCase):
    def test_topology_one_rollout_has_no_successor(self):
        plugin = load_plugin()
        actor, rollout = plugin.RtpGuardedNixl.build_topology(1, 1, ["actor", "rollout"])
        self.assertEqual(actor["rank"], [0])
        self.assertEqual(rollout["next_agent_metadata"], [None])

    def test_topology_multiple_rollouts_preserves_upstream(self):
        plugin = load_plugin()
        metadata = ["actor", "r0", "r1", "r2", "r3"]
        _, rollout = plugin.RtpGuardedNixl.build_topology(1, 4, metadata)
        self.assertEqual(rollout["next_agent_metadata"], ["r1", "r2", "r3", None])

    def setUp(self):
        self.module = load_plugin()
        self.engine = self.module.RtpGuardedNixl.__new__(self.module.RtpGuardedNixl)
        e = self.engine
        e.active, e.poisoned, e.finished = True, False, False
        e.pending = e.pending_name = e.pending_step = None
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


if __name__ == "__main__": unittest.main()
