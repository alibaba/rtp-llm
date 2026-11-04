"""Eager Decode position metadata stays alive on its consuming CUDA stream."""

import gc
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import (
    DsaKVExecution,
    DsaPreparedKVBlock,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DsaDecodePositionsLifetimeTest(unittest.TestCase):
    def test_position_allocation_is_leased_to_consumer_stream(self):
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        device = torch.device("cuda", 0)

        class Phase:
            def reserve(self, shapes):
                pass

        with torch.cuda.stream(producer):
            positions = torch.arange(65536, device=device, dtype=torch.int32)
            ptr = positions.data_ptr()
            ready = torch.cuda.Event()
            ready.record()
        block = DsaPreparedKVBlock(
            None, 0, 8192, None, indexer_decode_positions=positions
        )
        execution = object.__new__(DsaKVExecution)
        execution.metadata_ready = ready
        execution.metadata_stream = producer.cuda_stream
        execution.owner = SimpleNamespace(device=device)
        execution.blocks = [block]
        execution.graph = False
        execution.workspace = Phase()
        execution.stages = None
        execution.shapes = {}
        with torch.cuda.stream(consumer):
            execution.begin()
            torch.cuda._sleep(1000000000)
            result = positions.sum()
        # Releasing an eager plan must leave the input's allocation awaiting consumer work.
        block.indexer_decode_positions = None
        execution.blocks = []
        positions = None
        gc.collect()
        blocks = [
            b
            for s in torch.cuda.memory_snapshot()
            for b in s["blocks"]
            if b["address"] == ptr
        ]
        state = blocks[0]["state"] if blocks else "missing"
        consumer.synchronize()
        value = int(result.item())
        expected = 65536 * 65535 // 2
        self.assertIn(state, ("active_awaiting_free", "active_pending_free"))
        self.assertEqual(value, expected)


if __name__ == "__main__":
    unittest.main()
