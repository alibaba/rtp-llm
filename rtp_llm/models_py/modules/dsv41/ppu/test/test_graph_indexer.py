import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import indexer_topk
from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import (
    position_slots,
    read_packed_rows,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import (
    indexer_scores,
    select_topk,
    verify_indexer_scores,
)


class PagedGraphIndexerTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_replay_changes_visible_lengths_and_planar_pages(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        import deep_gemm

        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.assertTrue(hasattr(deep_gemm, "fp8_fp4_paged_sparse_mqa_logits"))
        self.assertTrue(hasattr(rtp_llm_ops, "dsv4_persistent_topk"))
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self._check_replay(ratio)

    def _check_replay(self, ratio):
        device = torch.device("cuda", torch.cuda.current_device())
        _quant.prepare_e2m1_codebooks(device)
        physical_page, pages, batch, topk = 256, 8, 4, 512
        pool = torch.empty(
            (pages + 1, physical_page * 68), dtype=torch.uint8, device=device
        )
        # Distinct scales and payloads detect addressing errors in both planes.
        token = torch.arange(physical_page, device=device)
        page = torch.arange(pages + 1, device=device)
        codes = torch.where((token[None, :] + page[:, None]) % 2 == 0, 0x22, 0x44)
        scales = 126 + (token[None, :] + 2 * page[:, None]) % 3
        pool[:, : physical_page * 64].copy_(codes.repeat_interleave(64, dim=1))
        pool[:, physical_page * 64 :].copy_(scales.repeat_interleave(4, dim=1))
        table = torch.empty((batch, pages), dtype=torch.int32, device=device)
        positions = torch.empty(batch, dtype=torch.int32, device=device)
        valid = torch.empty(batch, dtype=torch.bool, device=device)
        query = torch.ones((batch, 32, 128), dtype=torch.bfloat16, device=device)
        weights = torch.zeros((batch, 32), dtype=torch.bfloat16, device=device)
        weights[:, 5] = 1
        tag = "indexer_kv"
        cache = SimpleNamespace(
            max_seq_len=physical_page * pages,
            tables={tag: table},
            entries={tag: physical_page},
            spans={tag: physical_page},
            pool=lambda _layer, _tag: pool,
            slots=lambda _layer, _tag, raw: position_slots(
                raw,
                table,
                span=physical_page,
                entries=physical_page,
                ratio=ratio,
                full=True,
                pool_blocks=pool.shape[0],
            ),
            read_rows=lambda _layer, _tag, slots: read_packed_rows(
                pool,
                slots,
                width=68,
                entries=physical_page,
                planar=True,
            ),
        )

        def update(lengths, active, tables):
            raw_positions = [length * ratio - 1 + ratio // 2 for length in lengths]
            positions.copy_(torch.tensor(raw_positions, dtype=positions.dtype))
            valid.copy_(torch.tensor(active))
            table.copy_(torch.tensor(tables, dtype=table.dtype))

        def forward():
            scores = indexer_scores(
                query,
                weights,
                cache=cache,
                source_layer=0,
                ratio=ratio,
                positions=positions,
                valid_mask=valid,
            )
            selected = select_topk(scores, (positions + 1) // ratio, valid, topk)
            return scores, selected

        def check(scores, selected):
            expected = indexer_topk(
                query,
                weights,
                cache=cache,
                source_layer=0,
                ratio=ratio,
                positions=positions,
                valid_mask=valid,
                topk=topk,
                max_seq_len=cache.max_seq_len,
                scores_only=True,
            )
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
            for row, length in enumerate(((positions + 1) // ratio).tolist()):
                if not valid[row].item() or length <= topk:
                    ordered = torch.arange(topk, dtype=torch.int32, device=device)
                    count = length if valid[row].item() else 0
                    torch.testing.assert_close(
                        selected[row], ordered.masked_fill(ordered >= count, -1)
                    )
                else:
                    indices = selected[row][selected[row] >= 0].long()
                    self.assertEqual(indices.numel(), indices.unique().numel())
                    self.assertTrue((indices < length).all().item())
                    top_values = expected[row].topk(topk).values
                    torch.testing.assert_close(
                        expected[row, indices].sort(descending=True).values,
                        top_values[torch.isfinite(top_values)],
                        rtol=0,
                        atol=0,
                    )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            update(
                [600, 257, 600, 641],
                [False, True, True, True],
                [
                    list(range(1, 9)),
                    list(range(1, 9)),
                    [2, 0, 3, 4, 5, 6, 7, 8],
                    list(range(8, 0, -1)),
                ],
            )
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward()

        with torch.cuda.stream(stream):
            graph.replay()
            check(*captured)
            update(
                [513, 512, 7, 800],
                [True, True, True, False],
                [
                    [8, 3, -1, 4, 6, 5, 7, 2],
                    list(range(8, 0, -1)),
                    [7, 8, 0, 6, 5, 4, 3, 2],
                    [0] * pages,
                ],
            )
            graph.replay()
            check(*captured)
            eager = forward()
            torch.testing.assert_close(captured[0], eager[0], rtol=0, atol=0)
            # The native long-row kernel emits via atomics, so compare the
            # exact IDs without imposing an ordering on that row.
            torch.testing.assert_close(
                captured[1].sort(dim=1).values, eager[1].sort(dim=1).values
            )
            valid.zero_()
            graph.replay()
            check(*captured)
            torch.testing.assert_close(captured, forward(), rtol=0, atol=0)
        stream.synchronize()

        # Verify rows share an owner's committed pages but see different
        # prefixes of its speculative journal, including ratio-2 parity.
        width = 4
        requests = torch.arange(batch, device=device).repeat_interleave(width)
        prefixes = torch.tensor([255, 512, 767, 1024], device=device)
        offsets = torch.arange(width, device=device).repeat(batch)
        verify_query = query.repeat_interleave(width, dim=0)
        verify_weights = weights.repeat_interleave(width, dim=0)
        verify_valid = requests != 3
        table.copy_(torch.arange(1, pages + 1, device=device).expand(batch, pages))
        replacement = torch.full((68,), 0x66, dtype=torch.uint8, device=device)
        replacement[64:] = 127

        def physical_slots(raw):
            return position_slots(
                raw, table[requests], span=physical_page, entries=physical_page,
                ratio=ratio, full=True, pool_blocks=pool.shape[0],
            )

        def read_journal(layer, tag, raw):
            base = cache.read_rows(layer, tag, physical_slots(raw))
            first = (prefixes[requests] // ratio * ratio)[:, None]
            return torch.where((raw >= first)[..., None], replacement, base)

        journal = SimpleNamespace(
            native_cache=cache, prefixes=prefixes, requests=requests,
            slots=lambda _layer, _tag, raw: torch.where(physical_slots(raw) >= 0, raw, -1),
            read_rows=read_journal,
        )

        def verify():
            return verify_indexer_scores(
                verify_query, verify_weights, cache=journal, source_layer=0,
                ratio=ratio, positions=prefixes[requests] + offsets, valid_mask=verify_valid,
            )

        with torch.cuda.stream(stream):
            verify()
        stream.synchronize()
        verify_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(verify_graph, stream=stream):
            actual = verify()
        with torch.cuda.stream(stream):
            for _ in range(2):
                verify_graph.replay()
                expected = indexer_topk(
                    verify_query, verify_weights, cache=journal, source_layer=0,
                    ratio=ratio, positions=prefixes[requests] + offsets,
                    valid_mask=verify_valid, topk=topk, max_seq_len=cache.max_seq_len,
                    scores_only=True,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                prefixes.add_(1)
        stream.synchronize()

    def test_topk_cpu_ordinal_and_invalid(self):
        scores = torch.tensor([[2.0, 1.0, 3.0], [2.0, 3.0, 1.0], [1.0, 2.0, 3.0]])
        selected = select_topk(
            scores,
            torch.tensor([2, 3, 3]),
            torch.tensor([1, 1, 0], dtype=torch.bool),
            2,
        )
        torch.testing.assert_close(
            selected, torch.tensor([[0, 1], [1, 0], [-1, -1]], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
