"""C4 prefill output and retained state across aligned chunks and short tails."""
import unittest
from types import SimpleNamespace

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
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

    @torch.inference_mode()
    def test_reused_prefix_with_remapped_pages_and_mixed_requests(self):
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import build_plans
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import compress4

        torch.manual_seed(91)
        for entries in (8, 12):
            for prefix in (256, 512, 8192, 65536, 71680, 1048320):
                for suffix in (1, 3, 4, 7, 257):
                    with self.subTest(entries=entries, prefix=prefix, suffix=suffix):
                        # C4 needs only its trailing window. Keep the absolute
                        # position large without exceeding the 65536-row plan ABI.
                        retained = prefix if prefix <= 8192 else 512
                        origin = prefix - retained
                        common = torch.randn(retained, 512, device="cuda")
                        tails = [torch.randn(suffix, 512, device="cuda") for _ in range(2)]
                        cold = torch.randn(13, 512, device="cuda")
                        ape = torch.randn(8, 128, device="cuda")
                        columns = (prefix + suffix + 255) // 256
                        table = torch.arange(
                            1, columns + 1, device="cuda", dtype=torch.int32
                        ).unsqueeze(0)

                        def run(pool, bt, inputs, starts):
                            positions, requests, writes = [], [], []
                            for request, (x, start) in enumerate(zip(inputs, starts)):
                                end = start + len(x)
                                pos = torch.arange(start, end, device="cuda")
                                physical = bt[request, pos // 256].long()
                                block_end = torch.minimum(
                                    (pos // 256 + 1) * 256, torch.full_like(pos, end)
                                )
                                positions.append(pos)
                                requests.append(torch.full_like(pos, request))
                                writes.append(torch.where(
                                    pos + entries >= block_end,
                                    physical * entries + pos % entries, -1,
                                ))
                            pos = torch.cat(positions)
                            meta = SimpleNamespace(
                                positions=pos, b_idx=torch.cat(requests),
                                is_batched=True,
                                seq_start_per_req=torch.tensor(starts, device="cuda"),
                                state_slots=torch.cat(writes), kv_slots=pos // 4,
                            )
                            compute, write, _ = build_plans(meta, bt, entries, 256, None)
                            output = compress4(pool, torch.cat(inputs), ape, compute, write)
                            return output[(pos + 1) % 4 == 0].clone()

                        cached = torch.zeros(columns + 1, entries, 512, device="cuda")
                        run(cached, table, [common], [origin])
                        expected = []
                        expected_states = []
                        for tail in tails:
                            pool = torch.zeros_like(cached)
                            complete = run(pool, table, [torch.cat((common, tail))], [origin])
                            expected.append(complete[retained // 4:])
                            expected_states.append(pool)
                        cold_pool = torch.zeros_like(cached)
                        expected.append(run(cold_pool, table, [cold], [0]))

                        # Restore the same cached prefix to independent physical pages.
                        # Different suffixes and a cold request share a single launch.
                        remapped = torch.arange(
                            1, 3 * columns + 1, device="cuda", dtype=torch.int32
                        ).reshape(3, columns).flip(1).contiguous()
                        restored = torch.zeros(3 * columns + 1, entries, 512, device="cuda")
                        cached_before = cached.clone()
                        for request in range(2):
                            for column in range(origin // 256, prefix // 256):
                                restored[remapped[request, column]] = cached[table[0, column]]
                        actual = run(restored, remapped, [*tails, cold], [prefix, prefix, 0])
                        torch.testing.assert_close(actual, torch.cat(expected), rtol=0, atol=0)
                        torch.testing.assert_close(cached, cached_before, rtol=0, atol=0)
                        for request, reference in enumerate([*expected_states, cold_pool]):
                            end = prefix + suffix if request < 2 else len(cold)
                            # Only live ring rows are defined in a partial block.
                            for pos in range(max(end // 256 * 256, end - entries), end):
                                column = pos // 256
                                torch.testing.assert_close(
                                    restored[remapped[request, column], pos % entries],
                                    reference[table[0, column], pos % entries], rtol=0, atol=0,
                                )

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
