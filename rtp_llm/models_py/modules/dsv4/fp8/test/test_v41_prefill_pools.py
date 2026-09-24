"""Request views, host budgets, and the SM100 scorer's scale alignment."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_pools as pools


class PrefillPoolsHostTest(unittest.TestCase):
    def test_joint_dispatch_starts_at_qualified_batch_boundary(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_joint_pool as joint

        attn = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, kv_cache_sharded=True),
            compress_ratio=1,
        )
        pool = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.uint8)
        result = object()
        for batch in (2, 6, 31, 32):
            with self.subTest(batch=batch), patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 0)
            ), patch.object(joint, "try_gather", return_value=result) as attempt:
                actual = pools.try_gather_prefill_pools(
                    attn, pool, pool, [1] * batch, None
                )
                if batch < 32:
                    attempt.assert_not_called()
                    self.assertIsNone(actual)
                else:
                    attempt.assert_called_once()
                    self.assertIs(actual, result)

    def test_budget_boundary_preserves_whole_requests(self):
        limit = pools._MAX_ROWS
        self.assertLessEqual(limit * pools._BYTES_PER_ROW, pools._MAX_TEMP_BYTES)
        self.assertGreater((limit + 1) * pools._BYTES_PER_ROW, pools._MAX_TEMP_BYTES)
        aligned_limit = limit // 256 * 256
        counts = (0, 256, aligned_limit - 256, 0, 256)
        groups = pools._request_groups(counts)
        cursor = 0
        for first, stop, rows in groups:
            self.assertEqual(first, cursor)
            self.assertEqual(rows, sum(counts[first:stop]))
            self.assertLessEqual(rows, limit)
            cursor = stop
        self.assertEqual(cursor, len(counts))

    def test_negative_and_oversized_requests_fall_back(self):
        for counts in ((-1, 4), (4, pools._MAX_ROWS + 1)):
            with self.subTest(counts=counts):
                self.assertIsNone(pools._request_groups(counts))

    def test_zero_rows_keep_request_order(self):
        self.assertEqual(pools._request_groups((0, 0)), [(0, 2, 0)])

    def test_ragged_boundaries_are_aligned_with_bounded_padding(self):
        for counts in ((1, 128), (2, 128), (3, 128), (128, 129, 132), (1, 0)):
            groups = pools._request_groups(counts)
            self.assertIsNotNone(groups)
            for first, stop, rows in groups:
                logical = sum(counts[first:stop])
                self.assertGreaterEqual(rows, logical)
                self.assertLessEqual(rows - logical, 255 * (stop - first))
                self.assertEqual(rows % 256, 0)

    def test_unsupported_metadata_returns_before_device_work(self):
        def case():
            return (
                SimpleNamespace(
                    _cp_ctx=SimpleNamespace(cp_size=4, kv_cache_sharded=True),
                    compress_ratio=1,
                ),
                SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.uint8),
                SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.uint8),
                [4, 8],
            )

        for reason in (
            "one_request",
            "no_cp",
            "one_rank",
            "unsharded",
            "ratio",
            "cpu",
            "main_dtype",
            "index_dtype",
            "device",
            "architecture",
            "oversized",
            "metadata_device",
            "metadata_dtype",
            "metadata_shape",
            "metadata_stride",
            "batch_limit",
        ):
            attn, main, index, ends = case()
            seq_ends = Mock(spec=torch.Tensor)
            seq_ends.device = torch.device("cuda:0")
            seq_ends.dtype = torch.int64
            seq_ends.shape = (2,)
            seq_ends.stride.return_value = 1
            if reason == "one_request":
                ends = [4]
            elif reason == "no_cp":
                attn._cp_ctx = None
            elif reason == "one_rank":
                attn._cp_ctx.cp_size = 1
            elif reason == "unsharded":
                attn._cp_ctx.kv_cache_sharded = False
            elif reason == "ratio":
                attn.compress_ratio = 4
            elif reason == "cpu":
                main.device = torch.device("cpu")
            elif reason == "main_dtype":
                main.dtype = torch.bfloat16
            elif reason == "index_dtype":
                index.dtype = torch.int8
            elif reason == "device":
                index.device = torch.device("cuda:1")
            elif reason == "oversized":
                ends = [pools._MAX_ROWS + 1, 4]
            elif reason == "metadata_device":
                seq_ends.device = torch.device("cpu")
            elif reason == "metadata_dtype":
                seq_ends.dtype = torch.int32
            elif reason == "metadata_shape":
                seq_ends.shape = (2, 1)
            elif reason == "metadata_stride":
                seq_ends.stride.return_value = 0
            elif reason == "batch_limit":
                ends = [4] * 129
                seq_ends.shape = (129,)
            capability = (9, 0) if reason == "architecture" else (10, 0)
            with self.subTest(reason=reason), patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), patch.object(torch, "arange", side_effect=AssertionError("device work")):
                self.assertIsNone(
                    pools.try_gather_prefill_pools(attn, main, index, ends, seq_ends)
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PrefillPoolsCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("SM100 scorer contract")

    def test_fused_row_metadata_matches_padded_eager_groups(self):
        for counts in (
            (0, 1, 255, 256, 257),
            tuple((i * 137) % 1031 for i in range(128)),
            (0, 0),
        ):
            for ratio in (1, 2):
                data = torch.tensor(
                    [c * ratio + ratio - 1 for c in counts],
                    device="cuda",
                    dtype=torch.int64,
                )
                backing = torch.zeros(
                    len(counts) * 2 + 1, device="cuda", dtype=torch.int64
                )
                backing[1::2] = data
                for ends in (data, backing[1::2]):
                    for first, stop in ((0, len(counts)), (1, len(counts))):
                        sizes = ends[first:stop] // ratio
                        padded = (sizes + 255) // 256 * 256
                        rows = sum((c + 255) // 256 * 256 for c in counts[first:stop])
                        offsets = padded.cumsum(0)
                        row = torch.arange(rows, dtype=torch.int64, device="cuda")
                        req = torch.bucketize(row, offsets[:-1], right=True)
                        local = row - (offsets - padded)[req]
                        expected = torch.where(
                            local < sizes[req], (local + 1) * ratio - 1, -1
                        )
                        pos, ids = pools._group_row_metadata(
                            ends, first, stop, rows, ratio
                        )
                        torch.testing.assert_close(pos, expected, rtol=0, atol=0)
                        torch.testing.assert_close(ids, req + first, rtol=0, atol=0)

    def test_fused_row_metadata_graph_reads_changed_lengths(self):
        ends = torch.tensor([1, 255, 257], device="cuda", dtype=torch.int64)
        pools._group_row_metadata(ends, 0, 3, 1024, 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = pools._group_row_metadata(ends, 0, 3, 1024, 1)
        ends.copy_(torch.tensor([257, 1, 255], device="cuda", dtype=torch.int64))
        graph.replay()
        expected = pools._group_row_metadata(ends, 0, 3, 1024, 1)
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def _gather(self, counts, ratio, limit=None):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as fp4

        total = sum(counts)
        main_entries, index_entries = 128 // ratio, 128
        main = torch.zeros(
            (total + main_entries - 1) // main_entries + 1,
            main_entries,
            288,
            dtype=torch.uint8,
            device="cuda",
        )
        index = torch.zeros(
            (total + index_entries - 1) // index_entries + 1,
            index_entries,
            68,
            dtype=torch.uint8,
            device="cuda",
        )
        torch.manual_seed(913)
        slots = torch.arange(total, dtype=torch.int64, device="cuda")
        fp4.quantize_and_insert_k_cache_fp4(
            torch.randn(total, 512, device="cuda", dtype=torch.bfloat16), main, slots
        )
        fp4.quantize_indexer_k_fp4(
            torch.randn(total, 128, device="cuda", dtype=torch.bfloat16), slots, index
        )
        saved = main.clone(), index.clone()
        starts = [sum(counts[:i]) for i in range(len(counts))]
        bases = torch.tensor(starts, dtype=torch.int64, device="cuda")
        transported = []

        def gather_shards(value):
            # One logical owner holds all rows in this single-GPU fixture.
            # Real multi-rank collective transport is outside this test.
            self.assertEqual(value.dtype, torch.uint8)
            transported.append(value)

        attn = SimpleNamespace(
            _cp_ctx=SimpleNamespace(cp_size=4, kv_cache_sharded=True),
            compress_ratio=ratio,
            _global_region=lambda: 1,
            _slots=lambda region, positions, requests: torch.where(
                positions >= 0, bases[requests] + (positions + 1) // ratio - 1, -1
            ),
            _gather_shards=gather_shards,
        )
        ends = [count * ratio + (ratio - 1) for count in counts]
        seq_ends = torch.tensor(ends, dtype=torch.int64, device="cuda")
        split_inputs = []
        original_split = torch.Tensor.split

        def split(tensor, sizes, dim=0):
            split_inputs.append((tensor, tuple(sizes), dim))
            return original_split(tensor, sizes, dim)

        with patch.object(pools, "_MAX_ROWS", limit or pools._MAX_ROWS), patch.object(
            torch.Tensor, "split", split
        ):
            result = pools.try_gather_prefill_pools(attn, main, index, ends, seq_ends)
        self.assertTrue(torch.equal(main, saved[0]))
        self.assertTrue(torch.equal(index, saved[1]))
        if result is None:
            return None
        self.assertEqual(len(result), len(counts))
        self.assertEqual(len(transported), len(split_inputs))
        self.assertEqual(len(split_inputs) % 3, 0)
        request = 0
        for group in range(0, len(split_inputs), 3):
            parents = split_inputs[group : group + 3]
            sizes = parents[0][1]
            offset = 0
            for padded_count in sizes:
                count = counts[request]
                global_keys, keys = result[request]
                for child, (parent, actual_sizes, dim) in zip(
                    (global_keys, keys.quant, keys.scale), parents
                ):
                    self.assertEqual(actual_sizes, sizes)
                    self.assertEqual(dim, 0)
                    self.assertTrue(child.is_contiguous())
                    self.assertEqual(child.shape[0], count)
                    self.assertEqual(
                        child.untyped_storage().data_ptr(),
                        parent.untyped_storage().data_ptr(),
                    )
                    self.assertEqual(child.storage_offset(), offset * parent.stride(0))
                selected = slots[starts[request] : starts[request] + count]
                expected_global = fp4.dequantize_k_cache_slots_fp4(main, selected)
                expected_quant, expected_scale = fp4.gather_indexer_k_fp4(
                    index, selected
                )
                self.assertTrue(torch.equal(global_keys, expected_global))
                self.assertTrue(torch.equal(keys.quant, expected_quant))
                self.assertTrue(torch.equal(keys.scale, expected_scale))
                offset += padded_count
                request += 1
        self.assertEqual(request, len(counts))
        return result

    def test_actual_gather_views_and_forced_budget_groups(self):
        for ratio in (1, 2):
            for counts, limit in (((0, 4, 8, 0, 12), None), ((4, 8, 4), 512)):
                with self.subTest(ratio=ratio, counts=counts, limit=limit):
                    self.assertIsNotNone(self._gather(counts, ratio, limit))

    def test_all_empty_requests(self):
        self.assertIsNotNone(self._gather((0, 0), 1))

    def test_ragged_gather_scores_exact(self):
        from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import (
            quantize_indexer_q,
            score_indexer_chunk,
        )

        for ratio in (1, 2):
            for counts in (
                (0, 1, 1, 1, 33, 65, 127, 129),
                (4, 8, 33),
                (128, 132, 129),
            ):
                result = self._gather(counts, ratio)
                self.assertIsNotNone(result)
                for heads in (32, 64):
                    q, sf = quantize_indexer_q(
                        torch.randn(5, heads, 128, dtype=torch.bfloat16, device="cuda")
                    )
                    weights = torch.rand(5, heads, device="cuda")
                    for request, (_, keys) in enumerate(result):
                        if len(keys) == 0:
                            continue
                        with self.subTest(ratio=ratio, heads=heads, request=request):
                            self.assertEqual(
                                keys.scale.data_ptr() % 16,
                                0,
                                "DeepGEMM's scale TMA descriptor requires 16-byte alignment",
                            )
                            visible = torch.tensor(
                                [0, 1, len(keys) // 2, len(keys), len(keys) + 1],
                                dtype=torch.int32,
                                device="cuda",
                            )
                            actual = score_indexer_chunk(
                                q, sf, keys.quant, keys.scale, weights, visible
                            )
                            expected = score_indexer_chunk(
                                q,
                                sf,
                                keys.quant.clone(),
                                keys.scale.clone(),
                                weights,
                                visible,
                            )
                            self.assertTrue(
                                torch.equal(
                                    actual.view(torch.int32), expected.view(torch.int32)
                                )
                            )

    def test_three_splits_have_no_gpu_activity_or_allocation(self):
        counts = (0, 1, 2, 3, 33, 65, 127, 129)
        parents = (
            torch.empty(sum(counts), 512, dtype=torch.bfloat16, device="cuda"),
            torch.empty(sum(counts), 64, dtype=torch.int8, device="cuda"),
            torch.empty(sum(counts), dtype=torch.int32, device="cuda"),
        )

        def split():
            return tuple(parent.split(counts, dim=0) for parent in parents)

        for _ in range(10):
            split()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(100):
                views = split()
        self.assertEqual(torch.cuda.memory_allocated(), allocated)
        gpu = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertEqual(gpu, [])
        for parent, children in zip(parents, views):
            for child in children:
                self.assertTrue(child.is_contiguous())
                self.assertEqual(child.untyped_storage().data_ptr(), parent.data_ptr())


if __name__ == "__main__":
    unittest.main()
