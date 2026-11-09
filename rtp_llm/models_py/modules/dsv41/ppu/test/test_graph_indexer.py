import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
    select_visible_indices,
    verify_indexer_scores,
)


class PagedGraphIndexerTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_packed_metadata_replay_matches_native_slot_mapping(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_cache import NativeGraphCache
        from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import GraphPackedAttention

        device = torch.device("cuda", torch.cuda.current_device())
        batch, window, width, blocks = 8, 128, 512, 5
        layouts = {
            (0, "swa_kv"): ("decoder_swa_kv", 130, 256),
            (1, "global_kv_1"): ("global_kv_1", 128, 128),
            (2, "global_kv_2"): ("global_kv_2", 64, 128),
        }
        cache = NativeGraphCache.__new__(NativeGraphCache)
        cache.device = device
        cache.geometry = SimpleNamespace(ratio=lambda owner: owner)
        cache._framework_tags = {key: domain for key, (domain, _, _) in layouts.items()}
        cache._pools = {key: torch.empty(blocks, 1, device=device) for key in layouts}
        cache.entries = {domain: entries for domain, entries, _ in layouts.values()}
        cache.spans = {domain: span for domain, _, span in layouts.values()}
        cache.tables = {
            domain: torch.tensor([2, 0, 3, 4, 5, -1], dtype=torch.int32, device=device)
            .repeat(batch, 1)
            for domain in cache.entries
        }
        op = GraphPackedAttention.__new__(GraphPackedAttention)
        op.cache = cache
        op.offsets = torch.arange(window, dtype=torch.int64, device=device)
        op.swa_domains = {0: ("decoder_swa_kv", blocks)}

        def reject_fallback(*args, **kwargs):
            self.fail("packed metadata unexpectedly used the fallback")

        cache.slots = reject_fallback
        positions = torch.tensor([0, 63, 129, 256, 384, 527, 1035, 2037], device=device)
        valid = torch.tensor([True, True, False, True, True, False, True, True], device=device)
        selected = torch.arange(width * 2, dtype=torch.int32, device=device).repeat(batch, 1)[:, ::2]
        selected[:, 1] = -1

        def forward():
            shared = {}
            return (
                op._swa_indices(0, positions, valid, shared),
                op._global_indices(1, 1, "global_kv_1", 1, selected, positions, valid, shared),
                op._global_indices(2, 2, "global_kv_2", 2, selected, positions, valid, shared),
            )

        def check(actual):
            end = positions + 1
            for index, ((owner, tag), (slots, lengths)) in enumerate(zip(layouts, actual)):
                if index == 0:
                    live_lengths = torch.where(valid, end.clamp_max(window), 0)
                    raw = (end - window).clamp_min(0)[:, None] + op.offsets[None]
                    raw = torch.where(op.offsets[None] < live_lengths[:, None], raw, -1)
                else:
                    visible = end // owner
                    live_lengths = torch.where(valid, visible.clamp_max(width), 0)
                    live = valid[:, None] & (selected >= 0) & (selected < visible[:, None])
                    raw = torch.where(live, selected.long() * owner, -1)
                reference = NativeGraphCache.slots(cache, owner, tag, raw).int()[:, None]
                torch.testing.assert_close(slots, reference, rtol=0, atol=0)
                torch.testing.assert_close(lengths, live_lengths.int(), rtol=0, atol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            check(forward())
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = forward()
        with torch.cuda.stream(stream):
            graph.replay()
            check(actual)
            positions.add_(129)
            valid.logical_not_()
            selected[:, 0] = 67
            for table in cache.tables.values():
                table[:, 1] = 1
                table[:, 2] = 0
            graph.replay()
            check(actual)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_compact_merge_replay_keeps_scores_order_and_padding(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import _compact_kernels

        device = torch.device("cuda", torch.cuda.current_device())
        batch, topk, width = 8, 512, 1536
        columns = torch.arange(width, device=device, dtype=torch.int32)
        ids = (2 * columns + 7).repeat(batch, 1)
        scores = (columns % 23 - 11).float().repeat(batch, 1)
        scores[:, 0] = float("inf")
        scores[:, 1] = float("nan")
        scores[:, 2] = -float("inf")
        scores[4].zero_()
        scores[4, ::2] = -0.0
        scores[4, 3] = 2 ** -133
        scores[4, 4] = -(2 ** -133)
        visible = torch.tensor([900, 7, 511, 512, 513, 1536, 0, 1536], device=device)
        valid = torch.tensor([True] * 7 + [False], device=device)
        selected = torch.empty(batch, topk, dtype=torch.int32, device=device)
        kernels = _compact_kernels()

        def forward():
            kernels.merge_topk[(batch,)](
                scores, ids, visible, valid, selected, 1, 1, topk, width,
                kernels.triton.next_power_of_2(width), num_warps=8,
            )

        def check():
            lengths = torch.where(valid, visible, 0)
            ranks = scores.argsort(dim=1, descending=True, stable=True)[:, :topk]
            values = scores.gather(1, ranks)
            ranked_ids = torch.where(torch.isfinite(values), ids.gather(1, ranks), -1)
            ordinal = torch.where(columns[:topk][None, :] < lengths[:, None], columns[:topk], -1)
            expected = torch.where(lengths[:, None] <= topk, ordinal, ranked_ids)
            torch.testing.assert_close(selected, expected, rtol=0, atol=0)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            check()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward()
        with torch.cuda.stream(stream):
            graph.replay()
            check()
            scores[:, 0] = -8.0
            scores[:, -1] = float("inf")
            visible[0] = 17
            visible[2] = 513
            valid[7] = True
            ids.add_(3)
            graph.replay()
            check()
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_compact_decode_replay_across_score_tiles(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import (
            mask_candidate_logits,
        )
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import (
            _candidate_flags,
            _quantize_score_query,
        )

        device = torch.device("cuda", torch.cuda.current_device())
        _quant.prepare_e2m1_codebooks(device)
        batch, page, ratio, topk = 8, 256, 2, 512
        block_size, topk_blocks = 8, 512
        capacity, pages = 65557, 513
        tag = "indexer_kv"
        pool = torch.empty((pages + 1, page * 68), dtype=torch.uint8, device=device)
        row = torch.arange(pages + 1, device=device)[:, None]
        token = torch.arange(page, device=device)[None, :]
        payload = torch.where((token + row) % 3 == 0, 0x22, 0x44)
        scales = 126 + ((token + 2 * row) % 3)
        pool[:, : page * 64].copy_(payload.repeat_interleave(64, dim=1))
        pool[:, page * 64 :].copy_(scales.repeat_interleave(4, dim=1))
        table = torch.arange(1, pages + 1, dtype=torch.int32, device=device).repeat(batch, 1)
        query = torch.ones(batch, 32, 128, dtype=torch.bfloat16, device=device)
        query[0, 5, :4] = torch.tensor(
            [-0.0, 2 ** -133, -(2 ** -133), 0.25], dtype=torch.bfloat16, device=device
        )
        weights = torch.zeros(batch, 32, dtype=torch.bfloat16, device=device)
        weights[:, 5] = 1
        visible = torch.tensor([0, 7, 512, 513, 65535, 65536, 65557, 65557], device=device)
        positions = visible * ratio - 1
        valid = torch.tensor([True] * 7 + [False], device=device)
        candidates = torch.tensor(
            [[0, 1, 67, 8150, 8192, 8193, -1, -1]] * batch,
            dtype=torch.int32, device=device,
        )
        active = torch.tensor([False, False, False, True, True, False, True, False], device=device)
        cache = SimpleNamespace(
            max_seq_len=capacity * ratio,
            tables={tag: table},
            entries={tag: page},
            spans={tag: page},
            pool=lambda _layer, _tag: pool,
        )
        args = dict(cache=cache, source_layer=0, ratio=ratio, positions=positions, valid_mask=valid)

        def forward():
            shared_visible = (positions + 1) // ratio
            bitmap_cache = {}
            source, published = select_visible_indices(
                query, weights, **args, topk=topk, candidate_block_size=block_size,
                candidate_topk_blocks=topk_blocks, visible=shared_visible,
            )
            consumer, _ = select_visible_indices(
                query, weights, **args, topk=topk, candidate_block_size=block_size,
                candidates=candidates, candidate_active=active, visible=shared_visible,
                candidate_flags_cache=bitmap_cache,
            )
            bitmap_entry, = bitmap_cache.values()
            reused, _ = select_visible_indices(
                query, weights, **args, topk=topk, candidate_block_size=block_size,
                candidates=candidates, candidate_active=active, visible=shared_visible,
                candidate_flags_cache=bitmap_cache,
            )
            reused_entry, = bitmap_cache.values()
            self.assertIs(bitmap_entry, reused_entry)
            return source, published, consumer, reused

        def check(actual):
            self.assertTrue(_quant._ppu_fast_path(query))
            torch.testing.assert_close(
                _quantize_score_query(query, compact=True), _quant.quantize_indexer_q(query),
                rtol=0, atol=0,
            )
            flags = _candidate_flags(candidates, capacity, block_size)
            ragged_flags = _candidate_flags(candidates[:, :-1].contiguous(), capacity, block_size)
            torch.testing.assert_close(flags, ragged_flags[:, :-1].to(flags.dtype), rtol=0, atol=0)
            live = (positions + 1) // ratio
            scores = indexer_scores(query, weights, **args)
            padding = (-scores.shape[1]) % block_size
            pooled = torch.nn.functional.pad(scores, (0, padding), value=-torch.inf)
            pooled = pooled.view(batch, -1, block_size).amax(-1)
            newest = ((live - 1).clamp_min(0) // block_size).long()
            pooled.scatter_(1, newest[:, None], torch.where(live > 0, torch.inf, -torch.inf)[:, None])
            ranks = pooled.argsort(dim=1, descending=True, stable=True)[:, :topk_blocks]
            expected_published = torch.where(pooled.gather(1, ranks) > -torch.inf, ranks, -1).int()
            torch.testing.assert_close(actual[1], expected_published, rtol=0, atol=0)
            masked = mask_candidate_logits(scores.clone(), candidates, block_size)
            masked = torch.where(active[:, None], masked, scores)
            for selected, logits in ((actual[0], scores), (actual[2], masked), (actual[3], masked)):
                expected = select_topk(logits, live, valid, topk)
                for index, (length, live_row) in enumerate(zip(live.tolist(), valid.tolist())):
                    if length <= topk or not live_row:
                        torch.testing.assert_close(selected[index], expected[index], rtol=0, atol=0)
                        continue
                    ids = selected[index][selected[index] >= 0].long()
                    expected_ids = expected[index][expected[index] >= 0].long()
                    self.assertEqual(ids.numel(), ids.unique().numel())
                    self.assertTrue((ids < length).all().item())
                    actual_values = logits[index, ids]
                    self.assertTrue((actual_values[:-1] >= actual_values[1:]).all().item())
                    # The persistent kernel may emit tied IDs in different orders.
                    torch.testing.assert_close(
                        actual_values.sort().values,
                        logits[index, expected_ids].sort().values,
                        rtol=0, atol=0,
                    )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            check(forward())
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = forward()
        with torch.cuda.stream(stream):
            graph.replay()
            check(captured)
            positions[0] = 2 * 65542 - 1
            positions[6] = 2 * 507 - 1
            valid[7] = True
            valid[5] = False
            table[0, 0] = 0
            table[7, -1] = -1
            query.mul_(0.5)
            weights[:, 5].mul_(0.75)
            candidates[0, 1] = 8194
            active[0] = True
            graph.replay()
            check(captured)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_sparse_candidates_replay_matches_full_scoring(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import mask_candidate_logits

        device = torch.device("cuda", torch.cuda.current_device())
        _quant.prepare_e2m1_codebooks(device)
        batch, page, ratio, topk = 8, 256, 2, 512
        tile, count, capacity, pages = 8, 81, 8441, 66
        tag = "indexer_kv"
        row = torch.arange(pages + 1, device=device)[:, None]
        token = torch.arange(page, device=device)[None, :]
        payload = 0x11 * (1 + (token * 7 + row * 3) % 6)
        scales = 124 + ((token + 2 * row) % 5)
        pool = torch.empty((pages + 1, page * 68), dtype=torch.uint8, device=device)
        pool[:, : page * 64].copy_(payload.repeat_interleave(64, dim=1))
        pool[:, page * 64 :].copy_(scales.repeat_interleave(4, dim=1))
        table = torch.arange(1, pages + 1, dtype=torch.int32, device=device).repeat(batch, 1)
        table[1] = table[1].flip(0)
        table[2, :3] = torch.tensor([0, -1, pages + 1], dtype=torch.int32, device=device)
        table[5, -1] = 0
        query = torch.ones(batch, 32, 128, dtype=torch.bfloat16, device=device)
        weights = torch.zeros(batch, 32, dtype=torch.bfloat16, device=device)
        weights[:, 5] = 1
        visible = torch.tensor([0, 7, 642, 643, 8441, 8440, 8441, 8441], device=device)
        positions = visible * ratio - 1
        valid = torch.tensor([True] * 7 + [False], device=device)
        active = torch.tensor([False, False, False, True, True, True, True, True], device=device)
        candidates = ((torch.arange(count, device=device) * 29) % 1054).int().repeat(batch, 1)
        candidates[:, :5] = torch.tensor([1055, 1055, 16, 15, -1], dtype=torch.int32, device=device)
        candidates[4, 2:] = -1
        candidates[6, :] = -1
        cache = SimpleNamespace(
            max_seq_len=capacity * ratio,
            tables={tag: table, "global_kv_2": torch.zeros_like(table)},
            entries={tag: page},
            spans={tag: page},
            pool=lambda _layer, _tag: pool,
        )
        args = dict(cache=cache, ratio=ratio, positions=positions, valid_mask=valid)

        def forward():
            live = (positions + 1) // ratio
            plans = {}
            sparse_args = dict(
                **args, topk=topk, candidate_block_size=tile, candidates=candidates,
                candidate_active=active, visible=live, candidate_flags_cache=plans,
                candidate_inactive_limit=count * tile,
            )
            selected, published = select_visible_indices(query, weights, source_layer=0, **sparse_args)
            self.assertIsNone(published)
            key, = plans
            self.assertEqual(key[0], "indexer_sparse_schedule")
            plan = plans[key]
            reused, _ = select_visible_indices(query, weights, source_layer=1, **sparse_args)
            self.assertEqual(len(plans), 1)
            self.assertIs(plans[key], plan)
            return selected, reused

        def check(outputs):
            live = (positions + 1) // ratio
            dense = indexer_scores(query, weights, source_layer=0, **args)
            masked = mask_candidate_logits(dense.clone(), candidates, tile)
            logits = torch.where(active[:, None], masked, dense)
            ranked = logits.argsort(dim=1, descending=True, stable=True)[:, :topk]
            scored = torch.where(logits.gather(1, ranked).isfinite(), ranked, -1)
            ordinal = torch.arange(topk, device=device)[None, :]
            short = torch.where(ordinal < live[:, None], ordinal, -1)
            expected = torch.where((live <= topk)[:, None], short, scored)
            expected = torch.where(valid[:, None], expected, -1).int()
            for selected in outputs:
                torch.testing.assert_close(selected, expected, rtol=0, atol=0)
            newest = (live[4] - 1) // tile * tile
            ids = outputs[0][4]
            self.assertTrue((ids == newest).any().item())
            self.assertFalse((ids >= live[4]).any().item())

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with patch.dict("os.environ", {"RTP_V41_PPU_GRAPH_SPARSE_INDEXER": "1"}):
            with torch.cuda.stream(stream):
                check(forward())
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = forward()
            with torch.cuda.stream(stream):
                graph.replay()
                check(captured)
                positions[0] = 2 * 8440 - 1
                positions[3] = 2 * 644 - 1
                positions[4] = 2 * 8439 - 1
                positions[7] = 2 * 511 - 1
                valid[5] = False
                valid[7] = True
                active[0] = True
                active[3] = False
                candidates[0, 3] = 1054
                candidates[4, 0] = 1054
                candidates[4, 1] = 1054
                table[0, -1] = 3
                table[2, 0] = 6
                table[4, -1] = 1
                query.mul_(0.5)
                weights[:, 5].mul_(0.75)
                graph.replay()
                check(captured)
        stream.synchronize()

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
