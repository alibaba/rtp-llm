"""C4 prefill output and retained state across aligned chunks and short tails."""
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.device.device_type import is_ppu


@unittest.skipUnless(
    is_ppu(),
    "requires a PPU",
)
class ChunkStateTest(unittest.TestCase):
    @torch.inference_mode()
    def test_reused_fp4_pages_preserve_payload_and_scales(self):
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import gather_k

        entries = 64
        cache = torch.randint(0, 256, (10, entries, 68), device="cuda", dtype=torch.uint8)
        table = torch.tensor(
            [[4, 2, 6], [4, 3, 7], [8, 5, 1]], device="cuda", dtype=torch.int32
        )
        lengths = (63, 65, 129)
        cu = torch.tensor([0, 63, 128, 257], device="cuda", dtype=torch.int32)
        raw = cache.view(10, entries * 68)
        payload = raw[:, :entries * 64].reshape(10, entries, 64).view(torch.int8)
        scales = raw[:, entries * 64:].contiguous().view(torch.int32)
        expected_k, expected_s = [], []
        for row, length in enumerate(lengths):
            pos = torch.arange(length, device="cuda")
            blocks = table[row, pos // entries].long()
            expected_k.append(payload[blocks, pos % entries])
            expected_s.append(scales[blocks, pos % entries])
        # Native cache transfers must copy all 68 bytes per entry, including scales.
        cache[9].copy_(cache[4])
        table[1, 0] = 9
        k, s = gather_k(cache, table, cu, sum(lengths), entries)
        torch.testing.assert_close(k, torch.cat(expected_k), rtol=0, atol=0)
        torch.testing.assert_close(s[:, 0], torch.cat(expected_s), rtol=0, atol=0)

    def test_whole_and_chunk_state(self):
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import build_plans
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import compress4

        torch.manual_seed(42)
        for rows in (8193, 16384, 16448, 16960):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 512, device="cuda", dtype=torch.float32)
                ape = torch.randn(8, 128, device="cuda", dtype=torch.float32)
                blocks = (rows + 255) // 256
                table = torch.full((1, blocks), -1, device="cuda", dtype=torch.int32)
                table[0, -2:] = torch.tensor([1, 2], device="cuda")
                whole = torch.zeros(9, 8, 512, device="cuda")
                split = torch.zeros_like(whole)

                def run(pool, bt, start, end):
                    pos = torch.arange(start, end, device="cuda", dtype=torch.int64)
                    ids = bt[0, pos // 256].long()
                    block_end = torch.minimum(
                        (pos // 256 + 1) * 256, torch.full_like(pos, end)
                    )
                    slots = torch.where(
                        (ids > 0) & (pos + 8 >= block_end), ids * 8 + pos % 8, -1
                    )
                    meta = SimpleNamespace(
                        positions=pos, b_idx=torch.zeros_like(pos), is_batched=True,
                        seq_start_per_req=torch.tensor([start], device="cuda"),
                        state_slots=slots, kv_slots=pos // 4,
                    )
                    compute, write, _ = build_plans(meta, bt, 8, 256, None)
                    return compress4(pool, x[start:end], ape, compute, write)[3::4].clone()

                expected = run(whole, table, 0, rows)
                chunk_table = table.clone()
                next_id = 3
                outputs = []
                for start in range(0, rows, 8192):
                    end = min(start + 8192, rows)
                    last = (end + 255) // 256
                    for column in range(max(0, last - 2), last):
                        if int(chunk_table[0, column]) < 0:
                            chunk_table[0, column] = next_id
                            next_id += 1
                    outputs.append(run(split, chunk_table, start, end))
                torch.testing.assert_close(torch.cat(outputs), expected, rtol=0, atol=0)
                torch.testing.assert_close(split[1:3], whole[1:3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
