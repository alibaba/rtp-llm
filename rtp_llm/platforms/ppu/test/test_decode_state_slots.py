"""State-ring mapping across page boundaries and changing Graph inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class StateSlotsTest(unittest.TestCase):
    @torch.inference_mode()
    def test_strided_ring_tables_and_int64_slots(self):
        from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
            _compute_state_pool_slot_mapping,
        )
        from rtp_llm.platforms.ppu.kernels.ppu_decode_state_slots import (
            update_compressor_state_slots,
        )

        batch, q_len, entries, tokens = 7, 2, 8, 16
        n = batch * q_len
        table = torch.empty((batch * 2, 6), device="cuda", dtype=torch.int32)[::2, ::2]
        positions = torch.zeros(n, device="cuda", dtype=torch.int64)
        requests = torch.arange(batch - 1, -1, -1, device="cuda").repeat_interleave(
            q_len
        )
        output = torch.full((n + 3,), -99, device="cuda", dtype=torch.int64)
        meta = SimpleNamespace(
            q_len_per_req=q_len,
            position_ids_long=positions,
            req_id_per_token_long=requests,
            pool_block_tables={"state": table},
            compressor_state_slot_mappings={"state": output},
            paged_pool_tokens_per_block={"state": tokens},
        )
        table.fill_(1)
        update = lambda: update_compressor_state_slots(meta, batch, {"state": entries})
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            update()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            update()
        torch.cuda.current_stream().wait_stream(stream)
        pointer = output.data_ptr()
        for step, offset in enumerate((0, 7, 8, 15, 16, 31, 32, 47, 48, 16384)):
            positions.copy_(torch.arange(n, device="cuda") + offset)
            table.random_(1, 1 << 30)
            table[step % batch, 0] = 0
            table[(step + 1) % batch, 1] = -1
            output.fill_(-99)
            graph.replay()
            expected = _compute_state_pool_slot_mapping(
                table, positions, requests, entries, tokens
            )
            self.assertTrue(torch.equal(output[:n], expected))
            self.assertTrue(torch.equal(output[n:], torch.full_like(output[n:], -99)))
            self.assertEqual(output.data_ptr(), pointer)
        # Empty batches must not enqueue a kernel or touch the retained output.
        with patch(
            "rtp_llm.platforms.ppu.kernels.ppu_decode_state_slots._state_slots_kernel"
        ) as kernel:
            update_compressor_state_slots(meta, 0, {"state": entries})
            kernel.__getitem__.assert_not_called()
        original = meta.pool_block_tables
        meta.pool_block_tables = {}
        before = output.clone()
        update()
        self.assertTrue(torch.equal(output, before))
        meta.pool_block_tables = original
        for invalid in (table[:, :0], table[:1], table.float()):
            meta.pool_block_tables = {"state": invalid}
            with self.assertRaisesRegex(ValueError, "block geometry"):
                update()
        meta.pool_block_tables = original
        with self.assertRaisesRegex(ValueError, "block geometry"):
            update_compressor_state_slots(meta, batch, {"state": 0})
        meta.position_ids_long = positions.to(torch.int32)
        with self.assertRaisesRegex(ValueError, "int64 device vectors"):
            update()


if __name__ == "__main__":
    unittest.main()
