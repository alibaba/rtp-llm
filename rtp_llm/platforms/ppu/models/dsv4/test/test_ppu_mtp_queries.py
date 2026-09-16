import dataclasses
import unittest

import torch

from rtp_llm.platforms.ppu.models.dsv4.decode_query import slice_compressor_query


@dataclasses.dataclass(frozen=True)
class Metadata:
    positions: torch.Tensor
    b_idx: torch.Tensor
    state_slots: torch.Tensor
    kv_slots: torch.Tensor
    token_to_req: torch.Tensor
    has_prefix: bool = True
    is_batched: bool = True
    seq_start_per_req: torch.Tensor = None
    cu_seq_per_req: torch.Tensor = None
    compressed_lens_per_token: torch.Tensor = None


class QuerySliceTest(unittest.TestCase):
    def test_request_order_and_padding_survive_each_query_slice(self):
        positions = torch.tensor([[254, 255, 256, 257], [510, 511, 512, 513], [-1] * 4])
        requests = torch.arange(3).repeat_interleave(4)
        slots = torch.arange(12).reshape(3, 4) + 100
        slots[2].fill_(-1)
        meta = Metadata(
            positions.flatten(),
            requests,
            slots.flatten(),
            slots.flatten() + 10,
            requests.int(),
            compressed_lens_per_token=(positions + 1) // 4,
        )
        columns = []
        for q in range(4):
            step = slice_compressor_query(meta, 3, 4, q)
            columns.append(step.positions)
            self.assertEqual(step.b_idx.tolist(), [0, 1, 2])
            self.assertEqual(step.state_slots.tolist(), [100 + q, 104 + q, -1])
            self.assertEqual(
                step.compressed_lens_per_token.tolist(),
                [
                    (254 + q + 1) // 4,
                    (510 + q + 1) // 4,
                    0,
                ],
            )
            self.assertTrue(step.positions.is_contiguous())
            self.assertFalse(step.is_batched)
        self.assertTrue(torch.equal(torch.stack(columns, dim=1), positions))
        self.assertTrue(meta.is_batched)
        with self.assertRaisesRegex(ValueError, "row counts differ"):
            slice_compressor_query(meta, 2, 4, 0)


@unittest.skipUnless(torch.cuda.is_available(), "requires a PPU test allocation")
class C4SpeculativeRingTest(unittest.TestCase):
    def test_native_compression_across_blocks_and_mtp_rollback(self):
        if torch.cuda.get_device_name() != "ZW-M890P":
            self.skipTest("requires M890P native kernels")
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import compress4_decode
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import (
            build_decode_plan,
        )

        torch.manual_seed(701)
        raw = torch.randn(2, 516, 512, device="cuda") * 0.2
        ape = torch.randn(8, 128, device="cuda") * 0.1
        table = torch.tensor([[5, 2], [3, 4]], device="cuda", dtype=torch.int32)
        starts = [254, 510]
        for entries in (8, 12):
            state = torch.zeros(6, entries, 512, device="cuda")
            # Preserve the last ring contents of each allocated physical block.
            for b, start in enumerate(starts):
                for pos in range(start):
                    physical = (
                        [5, 2][(pos // 256) % 2] if b == 0 else [3, 4][(pos // 256) % 2]
                    )
                    state[physical, pos % entries].copy_(raw[b, pos])

            def step(offset):
                pos = torch.tensor([s + offset for s in starts], device="cuda")
                req = torch.arange(2, device="cuda")
                physical = table[req, (pos // 256) % 2].long()
                slots = physical * entries + pos % entries
                meta = Metadata(pos, req, slots, slots, req.int())
                plan, _ = build_decode_plan(meta, table, entries, 256)
                actual_plan = plan.view(torch.int32).cpu()
                for b, p in enumerate(pos.tolist()):
                    self.assertEqual(actual_plan[b, 1].item(), slots[b].item())
                    if (p + 1) % 4 == 0:
                        for part in range(2):
                            window = p + 1 - 8 + 4 * part
                            block = table[b, (window // 256) % 2].item()
                            self.assertEqual(
                                actual_plan[b, 2 + part].item(),
                                block * (entries // 4) + (window % entries) // 4,
                            )
                fused = torch.stack([raw[b, s + offset] for b, s in enumerate(starts)])
                output = compress4_decode(state, fused, ape, plan)
                for b, p in enumerate(pos.tolist()):
                    if (p + 1) % 4:
                        continue
                    history = raw[b, p - 7 : p + 1]
                    values = torch.cat([history[:4, :128], history[4:, 128:256]])
                    scores = (
                        torch.cat([history[:4, 256:384], history[4:, 384:512]]) + ape
                    )
                    expected = (scores.softmax(dim=0) * values).sum(dim=0)
                    torch.testing.assert_close(
                        output[b], expected, atol=2e-6, rtol=2e-5
                    )

            for offset in range(4):
                step(offset)
            if entries == 12:
                # Reject the last three proposals; overwrite the next token
                # and recompute its C4 boundary from the retained prefix.
                for b, start in enumerate(starts):
                    raw[b, start + 1].add_(0.15)
                step(1)


if __name__ == "__main__":
    unittest.main()
