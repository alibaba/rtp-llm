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
    def test_verify_fullwidth_native_topk_math_and_graph(self):
        self._assert_verify_fullwidth_topk_math_and_graph()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_bf16_fullwidth_topk_math_and_graph(self):
        with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "1"}):
            self._assert_verify_fullwidth_topk_math_and_graph(bf16_api=True)

    def _assert_verify_fullwidth_topk_math_and_graph(self, *, bf16_api=False):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import (
            _select_verify_topk,
            _select_verify_topk_tiled,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        rows, topk = 32, 512
        api = "dsv4_persistent_topk_bf16_scores" if bf16_api else "dsv4_persistent_topk"
        if not hasattr(rtp_llm_ops, api):
            self.skipTest("BF16 TopK API required")
        native = getattr(rtp_llm_ops, api)
        for width in (131840, 65920):
            with self.subTest(width=width):
                column = torch.arange(width, device=device)
                row = torch.arange(rows, device=device)[:, None]
                scores = (column.float()[None, :] / 256 + row / 8).contiguous()
                if bf16_api:
                    scores = scores.to(torch.bfloat16).float()
                scores[:, ::17] = -float("inf")
                visible = torch.tensor([0, 7, 511, 512, 513, 65535, 65536, width] * 4, device=device)
                valid = torch.tensor([True] * 7 + [False], device=device).repeat(4)
                valid[7] = True
                tiles = [scores[:, start:start + 65536].contiguous()
                         for start in range(0, width, 65536)]
                scored = [(tile, None, bf16_api) for tile in tiles]
                calls = []
                raw_picks = torch.empty((rows, topk), dtype=torch.int32, device=device)
                native_lengths = torch.empty(rows, dtype=torch.int32, device=device)

                def invoke(logits, lengths, output, workspace, k, max_width):
                    calls.append((logits.shape, logits.stride(), k, max_width))
                    result = native(logits, lengths, output, workspace, k, max_width)
                    raw_picks.copy_(output)
                    native_lengths.copy_(lengths)
                    return result

                def forward():
                    calls.clear()
                    result = _select_verify_topk(
                        scored, visible, valid, topk, bf16_tail_provenance=bf16_api,
                    )
                    self.assertEqual(calls, [(torch.Size((rows, width)), (width, 1), topk, width)])
                    return result

                def check(actual, source="fullwidth"):
                    lengths = torch.where(valid, visible.clamp(0, width), 0)
                    self.assertTrue((actual >= -1).all().item())
                    for index, length in enumerate(lengths.tolist()):
                        if length <= topk:
                            ordinal = torch.arange(topk, dtype=torch.int32, device=device)
                            expected = torch.where(ordinal < length, ordinal, -1)
                            torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
                            continue
                        ids = actual[index][actual[index] >= 0].long()
                        self.assertEqual(ids.numel(), ids.unique().numel())
                        self.assertTrue((ids < length).all().item())
                        got = scores[index, ids]
                        expected = scores[index, :length].topk(topk).values
                        expected = expected[expected.isfinite()]
                        if got.numel() != expected.numel():
                            diagnostic = {"source": source, "width": width,
                                          "row": index, "length": length,
                                          "selected_finite": got.numel(),
                                          "expected_finite": expected.numel()}
                            if source == "fullwidth":
                                raw = raw_picks[index].long()
                                raw_live = (raw >= 0) & (raw < length)
                                raw_values = scores[index, raw.clamp(0, width - 1)]
                                diagnostic.update(
                                    native_length=native_lengths[index].item(),
                                    raw_head=raw[:32].tolist(), raw_tail=raw[-32:].tolist(),
                                    raw_invalid=raw[~raw_live][:32].tolist(),
                                    raw_nonfinite=raw[raw_live & ~raw_values.isfinite()][:32].tolist(),
                                )
                            print({"verify_fullwidth_topk_diagnostic": diagnostic}, flush=True)
                        # Native ordering and tied IDs are unspecified; the
                        # selected score multiset remains exactly TopK.
                        torch.testing.assert_close(got.sort().values, expected.sort().values, rtol=0, atol=0)
                        if ids.numel() and expected.unique().numel() == expected.numel():
                            ref_ids = scores[index, :length].topk(topk).indices
                            ref_ids = ref_ids[scores[index, ref_ids].isfinite()]
                            torch.testing.assert_close(ids.sort().values, ref_ids.sort().values, rtol=0, atol=0)

                def check_tiled():
                    # Keep the old native tile geometry as an independent
                    # provider entry point, in addition to the dense oracle.
                    check(_select_verify_topk_tiled(scored, visible, valid, topk), source="tiled")

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    check_tiled()
                stream.synchronize()
                with patch.object(rtp_llm_ops, api, side_effect=invoke):
                    with torch.cuda.stream(stream):
                        check(forward())
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = forward()
                    with torch.cuda.stream(stream):
                        graph.replay()
                        check(actual)
                        scores.copy_((column[None, :] % 19 - 9).expand(rows, width))
                        scores[:, ::7] = -float("inf")
                        scores[:, 3::11] = -0.0
                        visible.add_(3).clamp_max_(width)
                        visible[0] = width - 3
                        valid.logical_not_()
                        valid[0] = True
                        valid[3] = True
                        for index, tile in enumerate(tiles):
                            tile.copy_(scores[:, index * 65536:index * 65536 + tile.shape[1]])
                        graph.replay()
                        check(actual)
                    stream.synchronize()
                # Outside the one-call wrapper: the old path intentionally
                # invokes native once per tile and keeps its Torch merge.
                with torch.cuda.stream(stream):
                    check_tiled()
                stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_fullwidth_native_fp32_bf16_histogram_reuse(self):
        """Separate score representation from mixed-row persistent state."""
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        rows, width, topk = 32, 131840, 512
        column = torch.arange(width, device=device)
        row = torch.arange(rows, device=device)[:, None]
        original = (column.float()[None, :] / 256 + row / 8).contiguous()
        original[:, ::17] = -float("inf")
        for representation in (torch.float32, torch.bfloat16):
            scores = original.to(representation).float()
            for pattern in ("all-large", "mixed"):
                with self.subTest(representation=representation, pattern=pattern):
                    if pattern == "all-large":
                        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
                    else:
                        lengths = torch.tensor(
                            [0, 7, 511, 512, 513, 65535, 65536, 0] * 4,
                            dtype=torch.int32, device=device,
                        )
                        lengths[7] = width
                    output = torch.full((rows, topk), -777777, dtype=torch.int32, device=device)
                    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)
                    rtp_llm_ops.dsv4_persistent_topk(scores, lengths, output, workspace, topk, width)
                    for index, length in enumerate(lengths.tolist()):
                        raw = output[index].long()
                        context = {"representation": str(representation), "pattern": pattern,
                                   "row": index, "length": length,
                                   "raw_head": raw[:32].tolist(), "raw_tail": raw[-32:].tolist()}
                        if length <= topk:
                            ordinal = torch.arange(topk, device=device)
                            torch.testing.assert_close(
                                raw, torch.where(ordinal < length, ordinal, -1), rtol=0, atol=0,
                                msg=str(context),
                            )
                            continue
                        live = (raw >= 0) & (raw < length)
                        context["raw_invalid"] = raw[~live][:32].tolist()
                        self.assertTrue(live.all().item(), str(context))
                        self.assertEqual(raw.numel(), raw.unique().numel(), str(context))
                        values = scores[index, raw]
                        expected = scores[index, :length].topk(topk).values
                        finite = values[values.isfinite()]
                        expected = expected[expected.isfinite()]
                        context["raw_nonfinite"] = raw[~values.isfinite()][:32].tolist()
                        context["selected_finite"] = finite.numel()
                        context["expected_finite"] = expected.numel()
                        torch.testing.assert_close(
                            finite.sort().values, expected.sort().values, rtol=0, atol=0,
                            msg=str(context),
                        )

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_native_topk_partial_vector_refinement_graph(self):
        """Padding lanes must never enter the small-row refinement buffer."""
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        rows, topk = 32, 512
        for width in (131840, 65920):
            with self.subTest(width=width):
                column = torch.arange(width, device=device)
                row = torch.arange(rows, device=device)[:, None]
                base = column.float()[None, :] / 256 + row / 8
                lengths = torch.tensor(
                    [513, 514, 515, 0, 7, 511, 512, 513] * 4,
                    dtype=torch.int32, device=device,
                )
                scores = torch.empty((rows, width), dtype=torch.float32, device=device)
                output = torch.empty((rows, topk), dtype=torch.int32, device=device)
                workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

                def populate(sign, poison):
                    live = column[None, :] < lengths[:, None]
                    values = (base * sign).masked_fill(column[None, :] % 17 == 0, -float("inf"))
                    # Refinement would see these large finite values if an
                    # invalid padded ID reached its logits[buffered_id] read.
                    scores.copy_(torch.where(live, values, poison))

                def forward():
                    output.fill_(-777777)
                    rtp_llm_ops.dsv4_persistent_topk(scores, lengths, output, workspace, topk, width)

                def check():
                    for index, length in enumerate(lengths.tolist()):
                        ids = output[index].long()
                        context = f"width={width}, row={index}, length={length}, ids={ids[-16:].tolist()}"
                        if length <= topk:
                            ordinal = torch.arange(topk, device=device)
                            torch.testing.assert_close(
                                ids, torch.where(ordinal < length, ordinal, -1), rtol=0, atol=0,
                                msg=context,
                            )
                            continue
                        self.assertTrue(((ids >= 0) & (ids < length)).all().item(), context)
                        self.assertEqual(ids.unique().numel(), topk, context)
                        # Fewer than K finite entries force the -Inf threshold
                        # tie through deferred refinement, including at 513/4/5.
                        self.assertLess(scores[index, :length].isfinite().sum().item(), topk)
                        expected = scores[index, :length].topk(topk).values
                        torch.testing.assert_close(
                            scores[index, ids].sort().values, expected.sort().values,
                            rtol=0, atol=0, msg=context,
                        )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    populate(1, 1e6)
                    forward()
                    check()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    forward()
                with torch.cuda.stream(stream):
                    graph.replay()
                    check()
                    lengths.copy_(torch.tensor(
                        [515, 513, 514, 517, 518, 519, 520, 513] * 4,
                        dtype=torch.int32, device=device,
                    ))
                    populate(-1, -1e6)
                    graph.replay()
                    check()
                    # Change only masked storage: the logical score multiset
                    # and every valid-domain assertion must remain unchanged.
                    scores.masked_fill_(column[None, :] >= lengths[:, None], 1e6)
                    graph.replay()
                    check()
                stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_fullwidth_topk_nonfinite_epilogue_and_ordinal(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import _select_verify_topk

        device = torch.device("cuda", torch.cuda.current_device())
        rows, width, topk = 32, 131840, 512
        scores = torch.full((rows, width), -float("inf"), device=device)
        scores[0].fill_(float("inf"))
        scores[2].fill_(float("nan"))
        scores[3].fill_(-float("nan"))
        scores[4, :600] = float("inf")
        scores[4, 600:] = 0.0
        scores[5].copy_(torch.arange(width, device=device))
        scores[5, :4] = torch.tensor([float("nan"), -float("nan"), float("inf"), -float("inf")], device=device)
        scores[6].zero_()
        scores[6, ::2] = -0.0
        visible = torch.full((rows,), width, device=device)
        valid = torch.arange(rows, device=device) < 7
        tiles = [scores[:, start:start + 65536].contiguous() for start in range(0, width, 65536)]

        def forward():
            return _select_verify_topk([(tile, None, False) for tile in tiles], visible, valid, topk)

        def check(actual, changed=False):
            self.assertTrue((actual >= -1).all().item())
            for row in (2, 3, 4):
                self.assertTrue((actual[row] == -1).all().item())
            if changed:
                torch.testing.assert_close(actual[0, :7], torch.arange(7, dtype=torch.int32, device=device))
                self.assertTrue((actual[0, 7:] == -1).all().item())
                self.assertEqual((actual[1] >= 0).sum().item(), topk)
            else:
                self.assertTrue((actual[0] == -1).all().item())
                self.assertTrue((actual[1] == -1).all().item())
            self.assertTrue((actual[7:] == -1).all().item())
            for row in (5, 6):
                ids = actual[row][actual[row] >= 0].long()
                self.assertEqual(ids.numel(), ids.unique().numel())
                self.assertTrue((ids < width).all().item())
                values = scores[row, ids]
                self.assertTrue(values.isfinite().all().item())
            self.assertEqual((actual[6] >= 0).sum().item(), topk)

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
            visible[0] = 7
            scores[1].copy_(torch.arange(width, device=device))
            for index, tile in enumerate(tiles):
                tile.copy_(scores[:, index * 65536:index * 65536 + tile.shape[1]])
            graph.replay()
            check(actual, changed=True)
        stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_topk_preserves_native_outputs_and_torch_merge(self):
        """Inject identical native tile results to isolate the fused consumer.

        The native long-row TopK may emit tied IDs via atomics. Feeding the
        same native outputs to both consumers proves exact merge/tie behavior
        without asserting a stronger determinism contract on that provider.
        """
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import (
            _select_verify_topk_tiled as _select_verify_topk,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        rows, topk = 32, 512
        for width in (131840, 65920):
            with self.subTest(width=width):
                scores = ((torch.arange(width, device=device) % 19) - 9).float().repeat(rows, 1)
                scores[:, 0] = float("nan")
                scores[:, 1] = float("inf")
                scores[:, 2] = -float("inf")
                scores[:, 3] = -0.0
                visible = torch.tensor(
                    [0, 7, 511, 512, 513, 65535, 65536, width] * 4, device=device,
                )
                valid = torch.tensor([True] * 7 + [False], device=device).repeat(4)
                tiles = [scores[:, start:start + 65536].contiguous()
                         for start in range(0, width, 65536)]
                # Committed lengths are intentionally shorter than the
                # current visible journal; selection must ignore these.
                score_tiles = [(tile, torch.zeros(rows, dtype=torch.int32, device=device), True)
                               for tile in tiles]
                picks = []
                snapshots = []
                for tile in tiles:
                    picked = (torch.arange(topk, device=device) * 13 % tile.shape[1]).int().repeat(rows, 1)
                    picked[:, -1] = -1
                    picks.append(picked)
                    snapshots.append(torch.empty(rows, dtype=torch.int32, device=device))
                calls = []

                def native_result(logits, lengths, output, _workspace, k, columns):
                    index = len(calls) % len(tiles)
                    self.assertEqual((logits.shape, logits.stride(), k, columns),
                                     (tiles[index].shape, tiles[index].stride(), topk, tiles[index].shape[1]))
                    self.assertTrue(lengths.is_contiguous())
                    snapshots[index].copy_(lengths)
                    output.copy_(picks[index])
                    calls.append(index)

                def candidate():
                    calls.clear()
                    return _select_verify_topk(score_tiles, visible, valid, topk)

                def reference():
                    calls.clear()
                    return select_topk(torch.cat(tiles, dim=1), visible, valid, topk)

                def check(actual):
                    expected = reference()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    length = torch.where(valid, visible.clamp_min(0), 0).int()
                    for index, tile in enumerate(tiles):
                        wanted = (length - index * 65536).clamp(0, tile.shape[1])
                        torch.testing.assert_close(snapshots[index], wanted, rtol=0, atol=0)

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with patch.object(rtp_llm_ops, "dsv4_persistent_topk", side_effect=native_result):
                    with torch.cuda.stream(stream):
                        check(candidate())
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = candidate()
                    with torch.cuda.stream(stream):
                        graph.replay()
                        check(actual)
                        visible.add_(3).clamp_max_(width)
                        valid.logical_not_()
                        tiles[0][:, 0] = 17.0
                        picks[0][:, 0] = 0
                        graph.replay()
                        check(actual)
                    stream.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_sparse_tiles_match_dense_journal_and_live_pages(self):
        self._assert_verify_sparse_tiles_match_dense(nonuniform=False)

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_sparse_tiles_nonuniform_head_reduce_replay(self):
        self._assert_verify_sparse_tiles_match_dense(nonuniform=True)

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_compact_provider_tail_math_and_replay(self):
        from rtp_llm.models_py.modules.dsv41.ppu import graph_indexer
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        with patch.object(graph_indexer, "_verify_topk_op", return_value=(rtp_llm_ops.dsv4_persistent_topk, False)):
            self._assert_verify_sparse_tiles_match_dense(nonuniform=True, compact=True)

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_bf16_compact_provider_tail_math_and_replay(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        if not hasattr(rtp_llm_ops, "dsv4_persistent_topk_bf16_scores"):
            self.skipTest("BF16 TopK API required")
        native = rtp_llm_ops.dsv4_persistent_topk_bf16_scores
        with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "1"}), patch.object(
            rtp_llm_ops, "dsv4_persistent_topk_bf16_scores", wraps=native,
        ) as calls, patch.object(
            rtp_llm_ops, "dsv4_persistent_topk",
            side_effect=AssertionError("proven BF16 provider must use BF16 TopK"),
        ):
            self._assert_verify_sparse_tiles_match_dense(nonuniform=True, compact=True, bf16_api=True)
        self.assertEqual(calls.call_count, 4)
        for call in calls.call_args_list:
            self.assertEqual(call.args[0].shape, (32, 4100))
            self.assertEqual(call.args[-2:], (512, 4100))

    def test_verify_topk_api_selection_uses_static_provenance(self):
        from rtp_llm.models_py.modules.dsv41.ppu import graph_indexer

        generic, specialized = object(), object()
        ops = SimpleNamespace(dsv4_persistent_topk=generic, dsv4_persistent_topk_bf16_scores=specialized)
        with patch("rtp_llm.ops.compute_ops.rtp_llm_ops", ops), patch.object(
            _quant, "is_ppu", return_value=True,
        ), patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "1"}):
            def select(rows=32, k=512, device="cuda", proof=True):
                return graph_indexer._verify_topk_op(rows, torch.device(device), k, bf16_provenance=proof)

            self.assertEqual(select(), (specialized, True))
            for args in ({"proof": False}, {"rows": 33}, {"rows": 0}, {"k": 1024}, {"device": "cpu"}):
                self.assertEqual(select(**args), (generic, False))
            with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "0"}):
                self.assertEqual(select(), (specialized, True))
            del ops.dsv4_persistent_topk_bf16_scores
            self.assertEqual(select(), (generic, False))

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_compact_unique_fp32_scores_map_exact_ids(self):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu import graph_indexer
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        device = torch.device("cuda", torch.cuda.current_device())
        rows, blocks, tile, capacity, topk = 32, 128, 8, 8192, 512
        width = blocks * tile
        column = torch.arange(width, device=device)[None, :]
        logical_ids = (torch.arange(blocks, device=device) * 2).int().repeat(rows, 1)
        scores = (column.float() + torch.arange(rows, device=device)[:, None] / 32).contiguous()
        lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
        first = torch.full((rows,), (2 * blocks - 1) * tile, device=device)
        visible = first + 4
        valid = torch.ones(rows, dtype=torch.bool, device=device)
        allowed = torch.ones((rows, capacity // tile + 1), dtype=torch.bool, device=device)
        tentative = (width + 0.125 + torch.arange(4, device=device)).expand(rows, 4).contiguous()
        scored = SimpleNamespace(
            scores=scores, logical_ids=logical_ids, lengths=lengths,
            blocks=blocks, tile=tile, capacity=capacity, kernels=graph_indexer._compact_kernels(),
        )
        for prefix_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(prefix_dtype=prefix_dtype), patch.dict(
                "os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "1"},
            ), patch.object(
                rtp_llm_ops, "dsv4_persistent_topk_bf16_scores", create=True,
                side_effect=AssertionError("arbitrary FP32 tail must stay on generic TopK"),
            ):
                scored.scores = scores.to(prefix_dtype)
                actual = graph_indexer._select_verify_compact_topk(
                    scored, first, tentative, visible, valid, allowed, valid, topk,
                )
                dense = torch.full((rows, capacity), -float("inf"), device=device)
                logical = logical_ids.long().repeat_interleave(tile, dim=1) * tile + column % tile
                dense.scatter_(1, logical, scored.scores.float())
                tail = first[:, None] + torch.arange(4, device=device)[None, :]
                dense.scatter_(1, tail, tentative)
                expected = dense.topk(topk, dim=1)
                if prefix_dtype == torch.float32:
                    torch.testing.assert_close(
                        actual.sort(dim=1).values, expected.indices.int().sort(dim=1).values, rtol=0, atol=0,
                    )
                self.assertTrue((actual >= 0).all().item())
                self.assertTrue((actual < visible[:, None]).all().item())
                for ids in actual:
                    self.assertEqual(ids.unique().numel(), topk)
                torch.testing.assert_close(
                    dense.gather(1, actual.long()).sort(dim=1).values,
                    expected.values.sort(dim=1).values, rtol=0, atol=0,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_compact_layout_boundaries_and_coarse_bin_graph(self):
        self._assert_verify_compact_layout_boundaries_and_coarse_bin_graph()

    @unittest.skipUnless(torch.cuda.is_available(), "PPU required")
    def test_verify_bf16_compact_small_coarse_bin_graph(self):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        if not hasattr(rtp_llm_ops, "dsv4_persistent_topk_bf16_scores"):
            self.skipTest("BF16 TopK API required")
        with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_BF16_TOPK": "1"}):
            self._assert_verify_compact_layout_boundaries_and_coarse_bin_graph(bf16_api=True)

    def _assert_verify_compact_layout_boundaries_and_coarse_bin_graph(self, *, bf16_api=False):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu import graph_indexer

        device = torch.device("cuda", torch.cuda.current_device())
        rows, capacity, topk = 32, 131840, 512
        geometries = ((512, 8, 4), (1024, 8, 2)) if bf16_api else ((4096, 8, 2), (4096, 16, 4))
        for blocks, tile, tail_width in geometries:
            with self.subTest(blocks=blocks, tile=tile, tail_width=tail_width):
                width = blocks * tile
                column = torch.arange(width, device=device)[None, :]
                logical_ids = (torch.arange(blocks, device=device) * 2).int().repeat(rows, 1)
                scores = torch.ones((rows, width), dtype=torch.bfloat16, device=device)
                # All four values share one FP16 coarse histogram bin. The
                # high scores occur after both bounded native buffers, and
                # after compact index 32767 in the wide case.
                scores[:, -2048:] = (1 + (column[:, -2048:] % 4).float() / 128)
                lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
                lengths[2] = 0
                first = torch.full((rows,), (2 * blocks - 2) * tile + tile // 2, device=device)
                visible = first + tail_width
                visible[:4] = torch.tensor([0, 7, 511, 512], device=device)
                visible[4] = 513
                valid = torch.ones(rows, dtype=torch.bool, device=device)
                valid[-1] = False
                active = torch.ones_like(valid)
                active[5] = False
                allowed = torch.ones((rows, (capacity + tile - 1) // tile + 1), dtype=torch.bool, device=device)
                allowed[::2] = False
                tentative = torch.full((rows, tail_width), 2.0, device=device)
                scored = SimpleNamespace(
                    scores=scores, logical_ids=logical_ids, lengths=lengths,
                    blocks=blocks, tile=tile, capacity=capacity, kernels=graph_indexer._compact_kernels(),
                )
                pack = graph_indexer._pack_verify_compact_logits

                def forward():
                    captured = []

                    def capture(*args, **kwargs):
                        self.assertEqual(kwargs.get("cooperative_bf16"), bf16_api)
                        result = pack(*args, **kwargs)
                        captured.extend(result)
                        return result

                    with patch.object(graph_indexer, "_pack_verify_compact_logits", side_effect=capture):
                        selected = graph_indexer._select_verify_compact_topk(
                            scored, first, tentative, visible, valid, allowed, active, topk,
                            bf16_tail_provenance=bf16_api,
                        )
                    return selected, *captured

                def check(actual):
                    selected, packed, native_lengths = actual
                    mapped = logical_ids.long().repeat_interleave(tile, dim=1) * tile + column % tile
                    prefix_live = (column < lengths[:, None]) & (mapped < first[:, None])
                    prefix_live &= valid[:, None] & (mapped >= 0) & (mapped < capacity)
                    prefix_live &= mapped < visible[:, None]
                    prefix_score = scores.float().masked_fill(~prefix_live, -float("inf"))
                    tail = first[:, None] + torch.arange(tail_width, device=device)[None, :]
                    tail_live = valid[:, None] & (tail >= 0) & (tail < capacity) & (tail < visible[:, None])
                    permitted = allowed.gather(1, (tail // tile).clamp(0, allowed.shape[1] - 1))
                    tail_live &= ~active[:, None] | permitted
                    tail_score = tentative.masked_fill(~tail_live, -float("inf"))
                    expected_pack = torch.nn.functional.pad(
                        torch.cat((prefix_score, tail_score), dim=1),
                        (0, packed.shape[1] - width - tail_width), value=-float("inf"),
                    )
                    torch.testing.assert_close(packed, expected_pack, rtol=0, atol=0)
                    self.assertEqual(
                        packed.shape[1], ((max(width + tail_width, 0 if bf16_api else 32772) + 3) // 4) * 4,
                    )
                    torch.testing.assert_close(
                        native_lengths, torch.where(valid & (visible > topk), packed.shape[1], 0).int(),
                        rtol=0, atol=0,
                    )
                    dense = torch.full((rows, capacity), -float("inf"), device=device)
                    row_ids = torch.arange(rows, device=device)[:, None]
                    prefix_rows = row_ids.expand_as(mapped)
                    dense[prefix_rows[prefix_live], mapped[prefix_live]] = prefix_score[prefix_live]
                    tail_rows = row_ids.expand_as(tail)
                    dense[tail_rows[tail_live], tail[tail_live]] = tail_score[tail_live]
                    self.assertTrue((selected >= -1).all().item())
                    for index, length in enumerate(torch.where(valid, visible.clamp_min(0), 0).tolist()):
                        ids = selected[index].long()
                        if length <= topk:
                            ordinal = torch.arange(topk, device=device)
                            torch.testing.assert_close(ids, torch.where(ordinal < length, ordinal, -1), rtol=0, atol=0)
                            continue
                        ids = ids[ids >= 0]
                        self.assertEqual(ids.numel(), ids.unique().numel())
                        self.assertTrue((ids < min(length, capacity)).all().item())
                        wanted = dense[index, :length].topk(topk).values
                        torch.testing.assert_close(
                            dense[index, ids].sort().values, wanted[wanted.isfinite()].sort().values,
                            rtol=0, atol=0,
                        )
                    if tile == 16:
                        self.assertTrue((selected[7] > 65536).any().item())

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
                    scores[:, -2048:].fill_(-1.0)
                    scores[:, -1024:] = 3.0
                    logical_ids.add_(1)
                    lengths.sub_(tile)
                    first.sub_(1)
                    visible.copy_(first + tail_width)
                    visible[0] = 511
                    active.logical_not_()
                    allowed.logical_not_()
                    valid[1] = False
                    tentative.neg_()
                    graph.replay()
                    check(actual)
                stream.synchronize()

    def _assert_verify_sparse_tiles_match_dense(self, *, nonuniform, compact=False, bf16_api=False):
        if "ZW-" not in torch.cuda.get_device_name():
            self.skipTest("PPU required")
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import _verify_score_tiles
        from rtp_llm.models_py.modules.dsv41.ppu import graph_indexer

        device = torch.device("cuda", torch.cuda.current_device())
        _quant.prepare_e2m1_codebooks(device)
        owners, width, capacity, page, topk = 8, 4, 131840, 256, 512
        block, count = 8, 512
        pages = (capacity + page - 1) // page
        pool = torch.empty((pages + 1, page * 68), dtype=torch.uint8, device=device)
        row = torch.arange(pages + 1, device=device)[:, None]
        token = torch.arange(page, device=device)[None, :]
        pool[:, :page * 64].copy_((0x11 * (1 + (row + token * 3) % 6)).repeat_interleave(64, dim=1))
        pool[:, page * 64:].copy_((124 + (row * 2 + token) % 5).repeat_interleave(4, dim=1))
        table = torch.arange(1, pages + 1, dtype=torch.int32, device=device).repeat(owners, 1)
        requests = torch.arange(owners, device=device).repeat_interleave(width)
        offsets = torch.arange(width, device=device).repeat(owners)
        query = torch.ones(owners * width, 32, 128, dtype=torch.bfloat16, device=device)
        weights = torch.zeros(owners * width, 32, dtype=torch.bfloat16, device=device)
        weights[:, 5] = 1
        if nonuniform:
            # Exercise all four scale groups and both signs in the dot
            # products, rather than a constant vector/single-head reduction.
            feature = torch.arange(64, device=device)[None, None, :]
            low = (row[:, :, None] + token[:, :, None] * 3 + feature * 5) % 16
            high = (row[:, :, None] * 7 + token[:, :, None] + feature * 3) % 16
            pool[:, :page * 64].copy_((low | (high << 4)).reshape(pages + 1, page * 64))
            groups = torch.arange(4, device=device)[None, None, :]
            scales = 124 + (row[:, :, None] * 2 + token[:, :, None] + groups * 3) % 5
            pool[:, page * 64:].copy_(scales.reshape(pages + 1, page * 4))
            element = torch.arange(query.numel(), device=device).reshape_as(query)
            query.copy_((((element * 17 + element // 128 * 3) % 65) - 32).float() / 16)
            query[:, ::5, ::7] = -0.0
            query[:, ::7, :] = 0.0
            heads = torch.arange(32, device=device)[None, :]
            queries = torch.arange(owners * width, device=device)[:, None]
            weights.copy_((((queries * 7 + heads * 5) % 17) - 8).float() / 32)
            weights[:, ::6] = 0.0
            self.assertTrue((query > 0).any().item())
            self.assertTrue((query < 0).any().item())
            self.assertTrue((weights > 0).any().item())
            self.assertTrue((weights < 0).any().item())
        replacement = torch.full((68,), 0x66, dtype=torch.uint8, device=device)
        replacement[64:] = 127
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                prefixes = torch.tensor([131072, 131071, 4090 * ratio, 510 * ratio,
                                         131070, 0, 131072, 131072], device=device)
                valid = (requests != 7)
                live_table = table[requests].contiguous()
                candidate_ids = ((torch.arange(count, device=device) * 29) % (capacity // ratio // block)).int()
                candidates = candidate_ids.repeat(owners * width, 1)
                candidates[:, :4] = torch.tensor([3, 3, -1, 0], device=device, dtype=torch.int32)
                candidates[4, :] = -1
                native = SimpleNamespace(max_seq_len=capacity, tables={"indexer_kv": table},
                                         entries={"indexer_kv": page}, spans={"indexer_kv": page},
                                         pool=lambda _layer, _tag: pool)

                def physical(raw):
                    return position_slots(raw, live_table, span=page, entries=page, ratio=ratio,
                                          full=True, pool_blocks=pool.shape[0])

                def journal_read(_layer, _tag, raw):
                    base = read_packed_rows(pool, physical(raw), width=68, entries=page, planar=True)
                    first = (prefixes[requests] // ratio * ratio)[:, None]
                    return torch.where((raw >= first)[..., None], replacement, base)

                journal = SimpleNamespace(native_cache=native, prefixes=prefixes, requests=requests,
                                          row_tables={"indexer_kv": live_table},
                                          slots=lambda _layer, _tag, raw: torch.where(physical(raw) >= 0, raw, -1),
                                          read_rows=journal_read)

                def forward(sparse):
                    positions = prefixes[requests] + offsets
                    visible = (positions + 1) // ratio
                    active = (visible > count * block) & (visible > topk)
                    if sparse and compact:
                        captured = []
                        pack = graph_indexer._pack_verify_compact_logits

                        def capture(scored, first, tentative, visible, valid, allowed, active, k, **kwargs):
                            self.assertEqual(kwargs.get("cooperative_bf16"), bf16_api)
                            logits, lengths = pack(scored, first, tentative, visible, valid, allowed, active, k, **kwargs)
                            captured.extend((logits, lengths, scored.logical_ids, scored.lengths,
                                             first, visible, scored.blocks, scored.tile))
                            return logits, lengths

                        with patch.dict("os.environ", {"RTP_V41_PPU_VERIFY_COMPACT_TOPK": "1"}), patch.object(
                            graph_indexer, "_pack_verify_compact_logits", side_effect=capture,
                        ):
                            result = graph_indexer.select_verify_indices(
                                query, weights, cache=journal, source_layer=0, ratio=ratio,
                                positions=positions, valid_mask=valid, topk=topk,
                                candidates=candidates, candidate_active=active, candidate_block_size=block,
                                shared={}, candidate_inactive_limit=count * block,
                            )
                        self.assertIsNotNone(result)
                        self.assertIsNone(result[1])
                        self.assertEqual(len(captured), 8, "compact provider unexpectedly fell back")
                        return (result[0], *captured)
                    tiles, _, _ = _verify_score_tiles(
                        query, weights, cache=journal, source_layer=0, ratio=ratio,
                        positions=positions, valid_mask=valid, fused=True,
                        candidates=candidates, candidate_active=active, candidate_block_size=block,
                        shared={}, candidate_inactive_limit=count * block if sparse else None, topk=topk,
                    )
                    return tuple(tile for tile, _, _ in tiles)

                def check(actual):
                    expected = forward(False)
                    if not compact:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        return
                    selected, packed, native_lengths, logical_ids, lengths, first, visible, blocks, tile = actual
                    dense = torch.cat(expected, dim=1)
                    column = torch.arange(blocks * tile, device=device)[None, :]
                    mapped = logical_ids.long().repeat_interleave(tile, dim=1) * tile + column % tile
                    prefix_live = (column < lengths[:, None]) & (mapped >= 0)
                    prefix_live &= (mapped < first[:, None]) & (mapped < visible[:, None])
                    prefix_live &= (mapped < dense.shape[1]) & valid[:, None]
                    prefix_score = dense.gather(1, mapped.clamp(0, dense.shape[1] - 1))
                    prefix_score.masked_fill_(~prefix_live, -float("inf"))
                    tail_width = (width + ratio - 1) // ratio
                    tail = first[:, None] + torch.arange(tail_width, device=device)[None, :]
                    tail_live = valid[:, None] & (tail >= 0) & (tail < dense.shape[1])
                    tail_live &= tail < visible[:, None]
                    tail_score = dense.gather(1, tail.clamp(0, dense.shape[1] - 1))
                    tail_score.masked_fill_(~tail_live, -float("inf"))
                    wanted = torch.nn.functional.pad(
                        torch.cat((prefix_score, tail_score), dim=1),
                        (0, packed.shape[1] - blocks * tile - tail_width), value=-float("inf"),
                    )
                    torch.testing.assert_close(packed, wanted, rtol=0, atol=0)
                    wanted_lengths = torch.where(valid & (visible > topk), packed.shape[1], 0).int()
                    torch.testing.assert_close(native_lengths, wanted_lengths, rtol=0, atol=0)
                    if bf16_api:
                        self.assertEqual(packed.shape[1], ((blocks * tile + tail_width + 3) // 4) * 4)
                        self.assertLess(packed.shape[1], 32768)
                    else:
                        self.assertGreater(packed.shape[1], 32768)
                    self.assertTrue((selected >= -1).all().item())
                    for index, length in enumerate(torch.where(valid, visible, 0).tolist()):
                        ids = selected[index].long()
                        if length <= topk:
                            ordinal = torch.arange(topk, device=device)
                            torch.testing.assert_close(ids, torch.where(ordinal < length, ordinal, -1), rtol=0, atol=0)
                            continue
                        ids = ids[ids >= 0]
                        self.assertEqual(ids.numel(), ids.unique().numel())
                        self.assertTrue((ids < min(length, dense.shape[1])).all().item())
                        values = dense[index, ids]
                        reference = dense[index, :length].topk(topk).values
                        torch.testing.assert_close(
                            values.sort().values, reference[reference.isfinite()].sort().values, rtol=0, atol=0,
                        )

                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    check(forward(True))
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = forward(True)
                with torch.cuda.stream(stream):
                    graph.replay()
                    check(captured)
                    prefixes.add_(1)
                    prefixes[0] = 4091 * ratio
                    prefixes[2] = 131069
                    valid.logical_not_()
                    valid[:4] = True
                    valid[8:12] = True
                    candidates[:, 1] = 1
                    if nonuniform:
                        query.neg_().mul_(0.75)
                        query[:, 2::7, 1::4] = 0.0
                        weights.neg_().mul_(0.6875)
                        weights[:, 1::5].add_(0.03125)
                    else:
                        query.mul_(0.5)
                        weights[:, 5].mul_(0.75)
                    replacement.fill_(0x22)
                    replacement[64:] = 126
                    live_table[:, 0] = 0
                    live_table[:, -1] = 1
                    graph.replay()
                    check(captured)
                stream.synchronize()

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
        op.journal = None
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
