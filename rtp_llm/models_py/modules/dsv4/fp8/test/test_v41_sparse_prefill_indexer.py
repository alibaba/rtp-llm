"""Independent sparse-MQA metadata, BF16-boundary, and remapping contracts.

Set DSV41_TEST_SPARSE_PREFILL_GPU=1 for the installed DeepGEMM CUDA path.
The old FP32 score rounded once to BF16 is a behavior comparison, not the
sparse kernel's arithmetic oracle: sparse MQA also rounds per-head weights,
ReLU values, head FMA accumulation, and the final pairwise additions to BF16.
"""

import importlib
import math
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _indexer_score as score_backend
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as dense_indexer


def sparse_module():
    return importlib.import_module(
        "rtp_llm.models_py.modules.dsv4.fp8._v41_sparse_prefill_indexer"
    )


def reference_plan(candidates, visible, key_count, block_size=8):
    """Scalar oracle: unique allowed positions, independent of GPU sorting."""
    indices, ends = [], []
    for row, raw_visible in zip(candidates.cpu().tolist(), visible.cpu().tolist()):
        length = min(max(raw_visible, 0), key_count)
        blocks = sorted({b for b in row if b >= 0 and b * block_size < length})
        end = sum(min(block_size, length - b * block_size) for b in blocks)
        pad = blocks[-1] if blocks else 0
        indices.append(blocks + [pad] * (len(row) - len(blocks)))
        ends.append(end)
    return torch.tensor(indices, dtype=torch.int32), torch.tensor(
        ends, dtype=torch.int32
    )


def reference_remap(columns, sparse_indices, ends, visible, key_count, logits=None):
    """Map columns by scalar arithmetic, filtering invalid/nonfinite entries."""
    result = torch.full(columns.shape, -1, dtype=torch.int32)
    columns = columns.cpu()
    sparse_indices = sparse_indices.cpu()
    ends, visible = ends.cpu(), visible.cpu()
    logits = None if logits is None else logits.cpu()
    for row in range(columns.shape[0]):
        length = min(max(int(visible[row]), 0), key_count)
        for slot in range(columns.shape[1]):
            column = int(columns[row, slot])
            if not 0 <= column < int(ends[row]):
                continue
            position = int(sparse_indices[row, column // 8]) * 8 + column % 8
            if position >= length:
                continue
            if logits is not None and not math.isfinite(float(logits[row, column])):
                continue
            result[row, slot] = position
    return result


def reference_bf16_head_reduce(dots, weights):
    """Exact BF16-FMA order used by both dense-BF16 and sparse DeepGEMM.

    Float64 intermediates represent the product and sum of BF16 operands
    before one BF16 rounding, avoiding an accidental separate product round.
    Head lanes 0/1 and 2/3 accumulate separately, then pairwise BF16-add.
    """
    values = dots.relu().bfloat16().double()
    weights = weights.bfloat16().double()
    lanes = [torch.zeros_like(values[:, 0]) for _ in range(4)]
    for head in range(values.shape[1]):
        lane = head % 4
        lanes[lane] = (
            (values[:, head] * weights[:, head, None] + lanes[lane]).bfloat16().double()
        )
    even = (lanes[0] + lanes[2]).bfloat16().double()
    odd = (lanes[1] + lanes[3]).bfloat16().double()
    return (even + odd).bfloat16()


def reference_bf16_logits(q, k, weights):
    q_fake = dense_indexer._fp4_rows_torch(q)[2].view(q.shape).double()
    k_fake = dense_indexer._fp4_rows_torch(k)[2].double()
    dots = torch.einsum("mhd,nd->mhn", q_fake, k_fake).float()
    return reference_bf16_head_reduce(dots, weights)


def make_candidates(rows, width, key_count, seed=811):
    """Unsorted/duplicate IDs, newest blocks, invalid IDs, and empty rows."""
    rng = random.Random(seed)
    max_block = (key_count + 7) // 8
    candidates = []
    for row in range(rows):
        values = [-1] * width
        if row % 5:
            count = min(width, max(4, 2 * max_block))
            values[:count] = [rng.randrange(max_block + 2) for _ in range(count)]
            values[:4] = [max_block - 1, 0, 0, -1]
        candidates.append(values)
    return torch.tensor(candidates, dtype=torch.int32)


class SparsePrefillReferenceCPU(unittest.TestCase):
    def test_shared_slab_groups_are_aliases_and_bounded_without_device_work(self):
        sparse = sparse_module()
        with torch.inference_mode():
            q = torch.empty(32 * 32768, 64, dtype=torch.int8)
            sf = torch.empty(32 * 32768, dtype=torch.int32)
            keys = [
                dense_indexer.PrefillIndexerKeys(a, b)
                for a, b in zip(q.split(32768), sf.split(32768))
            ]
            slices = [slice(i * 512, (i + 1) * 512) for i in range(32)]
            with patch.object(
                torch, "cat", side_effect=AssertionError("copied K slab")
            ), patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")):
                groups = sparse._batch_groups(keys, slices, 16384)
            self.assertEqual(len(groups), 4)
            for pieces, slab, offsets in groups:
                self.assertEqual(pieces[-1][2] - pieces[0][1], 4096)
                self.assertEqual(offsets, tuple(i * 32768 for i in range(8)))
                self.assertEqual(
                    slab.quant.untyped_storage().data_ptr(),
                    q.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    slab.scale.untyped_storage().data_ptr(),
                    sf.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    slab.quant.data_ptr(), keys[pieces[0][0]].quant.data_ptr()
                )
                self.assertTrue(slab.quant.is_contiguous())
            self.assertIsNone(sparse._batch_groups(keys, slices[:-1], 16384))
            self.assertIsNone(sparse._joined_keys([keys[1], keys[0]]))
            ragged = [
                dense_indexer.PrefillIndexerKeys(q[:12], sf[:12]),
                dense_indexer.PrefillIndexerKeys(q[12:24], sf[12:24]),
            ]
            self.assertIsNone(sparse._joined_keys(ragged))

    def test_row_key_vectors_require_both_aligned_layouts_and_cpu_falls_back(self):
        sparse = sparse_module()
        candidates = torch.zeros(3, 64, dtype=torch.int32)
        offsets = torch.zeros(3, dtype=torch.int32)
        counts = torch.full((3,), 16, dtype=torch.int32)
        self.assertTrue(sparse._row_keys_supported(candidates, offsets, counts))
        for left, right in (
            (None, counts),
            (offsets, None),
            (offsets.long(), counts),
            (offsets, counts[:2]),
            (offsets, torch.zeros(6, dtype=torch.int32)[::2]),
        ):
            self.assertFalse(sparse._row_keys_supported(candidates, left, right))
        with patch.object(
            torch.cuda, "current_stream", side_effect=AssertionError("CPU CUDA query")
        ):
            self.assertIsNone(
                sparse.prepare_plan(
                    candidates,
                    counts,
                    48,
                    row_key_offsets=offsets,
                    row_key_counts=counts,
                )
            )

    def test_ragged_groups_keep_partial_query_offsets_and_padded_storage_gaps(self):
        sparse = sparse_module()
        counts, query_rows = (257, 509, 16, 1024), (3, 4097, 2, 19)
        starts, sizes, total = [], [], 0
        for count in counts:
            starts.append(total)
            size = (count + 255) // 256 * 256
            sizes.append(size)
            total += size
        quant = torch.empty(total, 64, dtype=torch.int8)
        scale = torch.empty(total, dtype=torch.int32)
        keys = [
            dense_indexer.PrefillIndexerKeys(
                quant[start : start + n], scale[start : start + n]
            )
            for start, n in zip(starts, counts)
        ]
        slices, cursor = [], 0
        for rows in query_rows:
            slices.append(slice(cursor, cursor + rows))
            cursor += rows
        groups = sparse._batch_groups(keys, slices, cursor)
        self.assertEqual(
            [[(r, a, b) for r, a, b in g[0]] for g in groups],
            [
                [(0, 0, 3), (1, 3, 4096)],
                [(1, 4096, 4100), (2, 4100, 4102), (3, 4102, 4121)],
            ],
        )
        self.assertEqual(groups[0][2], (0, 512))
        self.assertEqual(groups[1][2], (0, 512, 768))
        self.assertEqual(len(groups[1][1]), 1792)
        self.assertEqual(groups[1][1].scale.storage_offset(), 512)

    def test_batched_plan_snapshots_offsets_with_single_metadata_call(self):
        sparse = sparse_module()
        candidates = torch.tensor([[1, 0, 1, -1], [0, 1, 2, 3]], dtype=torch.int32)
        visible = torch.tensor([12, 40], dtype=torch.int32)
        offsets = torch.tensor([0, 16], dtype=torch.int32)
        counts = torch.tensor([12, 9], dtype=torch.int32)
        stream = SimpleNamespace(device=SimpleNamespace(index=17), cuda_stream=918274)

        def prepare(cand, vis, ks, ke, indices, end, *args, **kw):
            self.assertTrue(kw["BATCHED"])
            local_visible = torch.minimum(vis, kw["key_counts"])
            local, lengths = reference_plan(cand, local_visible, 25)
            ks.zero_()
            ke.copy_(local_visible)
            end.copy_(lengths)
            indices.copy_(local + kw["key_offsets"][:, None] // 8)
            kw["row_offsets"].copy_(kw["key_offsets"])

        kernel = MagicMock()
        kernel.__getitem__.return_value = prepare
        dg = SimpleNamespace(
            get_sparse_mqa_logits_metadata=MagicMock(
                return_value=torch.empty(17, dtype=torch.uint8)
            )
        )
        with patch.object(sparse, "_plan_supported", return_value=True), patch.object(
            torch.cuda, "current_stream", return_value=stream
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), patch.object(
            sparse, "_prepare_sparse_prefill_plan_kernel", kernel
        ), patch.object(
            sparse, "_get_deep_gemm", return_value=dg
        ), patch.object(
            sparse, "_WARMED_STREAMS", set()
        ):
            plan = sparse.prepare_plan(
                candidates, visible, 25, row_key_offsets=offsets, row_key_counts=counts
            )
        dg.get_sparse_mqa_logits_metadata.assert_called_once()
        self.assertIs(dg.get_sparse_mqa_logits_metadata.call_args.args[1], plan.end)
        self.assertEqual(plan.end.tolist(), [12, 9])
        self.assertEqual(plan.row_ke.tolist(), [12, 9])
        self.assertEqual(plan.nbytes, 4 * 2 * 4 + 2 * 4 * 4 + 17)
        offsets.fill_(999)
        self.assertEqual(plan.row_key_offsets.tolist(), [0, 16])

    def test_metadata_uses_compact_slot_count_and_execution_errors_propagate(self):
        sparse = sparse_module()
        candidates = torch.tensor([[7, 0, 7, -1], [-1, -1, -1, -1]], dtype=torch.int32)
        visible = torch.tensor([61, 61], dtype=torch.int32)
        expected_indices, expected_end = reference_plan(candidates, visible, 65)
        stream = SimpleNamespace(device=SimpleNamespace(index=17), cuda_stream=918273)

        def prepare(cand, vis, ks, ke, indices, end, *args, **kwargs):
            ks.zero_()
            ke.copy_(vis)
            indices.copy_(expected_indices)
            end.copy_(expected_end)

        kernel = MagicMock()
        kernel.__getitem__.return_value = prepare
        dg = SimpleNamespace(
            get_sparse_mqa_logits_metadata=MagicMock(
                return_value=torch.empty(17, dtype=torch.uint8)
            )
        )
        with patch.object(sparse, "_plan_supported", return_value=True), patch.object(
            torch.cuda, "current_stream", return_value=stream
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ), patch.object(
            sparse, "_prepare_sparse_prefill_plan_kernel", kernel
        ), patch.object(
            sparse, "_get_deep_gemm", return_value=dg
        ), patch.object(
            sparse, "_WARMED_STREAMS", set()
        ):
            plan = sparse.prepare_plan(candidates, visible, 65)
            self.assertIs(dg.get_sparse_mqa_logits_metadata.call_args.args[1], plan.end)
            torch.testing.assert_close(plan.row_ke, visible)
            torch.testing.assert_close(
                plan.end, torch.tensor([13, 0], dtype=torch.int32)
            )
            self.assertEqual(plan.nbytes, 3 * 2 * 4 + 2 * 4 * 4 + 17)
            dg.get_sparse_mqa_logits_metadata.side_effect = RuntimeError(
                "injected DG metadata failure"
            )
            with self.assertRaisesRegex(RuntimeError, "injected DG metadata failure"):
                sparse.prepare_plan(candidates, visible, 65)

    def test_cpu_gate_does_not_query_cuda_or_load_deepgemm(self):
        sparse = sparse_module()
        candidates = torch.zeros(3, 64, dtype=torch.int32)
        visible = torch.ones(3, dtype=torch.int32)
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU CUDA query"),
        ), patch.object(
            sparse, "_get_deep_gemm", side_effect=AssertionError("CPU DeepGEMM load")
        ):
            self.assertIsNone(sparse.prepare_plan(candidates, visible, 257))

    def test_positions_ratio_rejects_unsupported_modes_before_device_work(self):
        sparse = sparse_module()
        candidates = torch.zeros(3, 64, dtype=torch.int32)
        positions = torch.ones(3, dtype=torch.int32)
        with patch.object(
            sparse, "_plan_supported", side_effect=AssertionError("device gate")
        ):
            for ratio in (0, 3, -1, True, 1.0):
                self.assertIsNone(
                    sparse.prepare_plan(
                        candidates, positions, 257, positions_ratio=ratio
                    )
                )

    def test_unwarmed_capture_returns_before_plan_allocation(self):
        sparse = sparse_module()
        candidates = torch.zeros(3, 64, dtype=torch.int32)
        visible = torch.ones(3, dtype=torch.int32)
        stream = SimpleNamespace(device=SimpleNamespace(index=17), cuda_stream=918273)
        with patch.object(sparse, "_plan_supported", return_value=True), patch.object(
            torch.cuda, "current_stream", return_value=stream
        ), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ), patch.object(
            sparse, "_WARMED_STREAMS", set()
        ), patch.object(
            torch, "empty", side_effect=AssertionError("Unwarmed capture allocation")
        ):
            self.assertIsNone(sparse.prepare_plan(candidates, visible, 257))

    def test_full_gate_checks_head_scale_and_selection_layout(self):
        sparse = sparse_module()
        q = torch.zeros(3, 32, 64, dtype=torch.int8)
        q_sf = torch.ones(3, 32, dtype=torch.int32)
        keys = dense_indexer.PrefillIndexerKeys(
            torch.zeros(257, 64, dtype=torch.int8), torch.ones(257, dtype=torch.int32)
        )
        weights = torch.ones(3, 32)
        candidates = torch.zeros(3, 64, dtype=torch.int32)
        visible = torch.ones(3, dtype=torch.int32)
        with patch.object(sparse, "_plan_supported", return_value=True):
            self.assertTrue(
                sparse.is_supported(q, q_sf, keys, weights, candidates, visible)
            )
            self.assertTrue(
                sparse.is_supported(
                    q, q_sf, keys, weights.bfloat16(), candidates, visible
                )
            )
            self.assertFalse(
                sparse.is_supported(
                    q[:, :16], q_sf[:, :16], keys, weights[:, :16], candidates, visible
                )
            )
            self.assertFalse(
                sparse.is_supported(q, q_sf.float(), keys, weights, candidates, visible)
            )
            self.assertFalse(
                sparse.is_supported(
                    q, q_sf, keys, weights.double(), candidates, visible
                )
            )
            self.assertFalse(
                sparse.is_supported(q, q_sf, keys, weights, candidates[:, :4], visible)
            )
            self.assertFalse(
                sparse.is_supported(
                    q, q_sf, keys, weights, candidates, visible, topk=511
                )
            )

    def test_plan_deduplicates_preserves_newest_and_counts_partial_tail(self):
        candidates = torch.tensor(
            [[2, 0, 2, 1], [4, -1, 0, 4], [-1, -1, -1, -1], [9, 1, 0, -1]],
            dtype=torch.int32,
        )
        indices, ends = reference_plan(candidates, torch.tensor([19, 33, 9, 0]), 33)
        torch.testing.assert_close(
            indices,
            torch.tensor(
                [[0, 1, 2, 2], [0, 4, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(ends, torch.tensor([19, 9, 0, 0], dtype=torch.int32))

    def test_remap_does_not_expose_padding_or_nonfinite_values(self):
        indices = torch.tensor([[0, 4, 4, 4], [0, 0, 0, 0]], dtype=torch.int32)
        columns = torch.tensor(
            [[0, 7, 8, 9, -1, 32], [0, 1, -1, 4, 8, 32]], dtype=torch.int32
        )
        logits = torch.ones(2, 32, dtype=torch.bfloat16)
        logits[0, 7] = torch.nan
        actual = reference_remap(
            columns, indices, torch.tensor([9, 0]), torch.tensor([33, 0]), 33, logits
        )
        torch.testing.assert_close(
            actual, torch.tensor([[0, -1, 32, -1, -1, -1], [-1] * 6], dtype=torch.int32)
        )

    def test_bf16_reduction_is_not_final_output_rounding(self):
        generator = torch.Generator().manual_seed(813)
        dots = torch.randn(2, 32, 19, generator=generator) * 13
        weights = torch.randn(2, 32, generator=generator) * 0.17
        actual = reference_bf16_head_reduce(dots, weights)
        final_round_only = (dots.relu() * weights[:, :, None]).sum(1).bfloat16()
        self.assertGreater(int((actual != final_round_only).sum()), 0)
        # Unit weights and exact small positive dots admit an independent
        # closed-form result without any reduction-order ambiguity.
        exact = reference_bf16_head_reduce(torch.ones(1, 32, 3), torch.ones(1, 32))
        torch.testing.assert_close(
            exact, torch.full((1, 3), 32.0, dtype=torch.bfloat16), rtol=0, atol=0
        )


@unittest.skipUnless(
    os.environ.get("DSV41_TEST_SPARSE_PREFILL_GPU") == "1",
    "Set DSV41_TEST_SPARSE_PREFILL_GPU=1 for DeepGEMM sparse CUDA tests",
)
class SparsePrefillCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("Sparse CUDA validation requires an SM100-family GPU")
        cls.sparse = sparse_module()
        cls.device = torch.device("cuda")

    def assert_plans_equal(self, actual, expected):
        self.assertIsNotNone(actual)
        self.assertIsNotNone(expected)
        for name in ("row_ks", "row_ke", "sparse_indices", "end", "row_key_offsets"):
            got, ref = getattr(actual, name), getattr(expected, name)
            if ref is None:
                self.assertIsNone(got)
            else:
                torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_positions_plan_signed_floor_overflow_strides_and_lookup(self):
        for dtype in (torch.int32, torch.int64):
            bounds = torch.iinfo(dtype)
            values = [
                bounds.min,
                bounds.min + 1,
                -9,
                -4,
                -3,
                -2,
                -1,
                0,
                1,
                7,
                8,
                17,
                256,
                511,
                bounds.max - 1,
                bounds.max,
            ]
            for width in (4, 64, 2048):
                with self.subTest(dtype=dtype, width=width):
                    rows = len(values)
                    backing = torch.empty(rows * 3, device=self.device, dtype=dtype)
                    positions = backing[1::3]
                    positions.copy_(
                        torch.tensor(values, device=self.device, dtype=dtype)
                    )
                    candidates = torch.full(
                        (rows, width + 7), -999, device=self.device, dtype=torch.int32
                    )
                    view = candidates[:, 3 : width + 3]
                    view.copy_(make_candidates(rows, width, 257).to(self.device))
                    ids = torch.arange(rows, device=self.device, dtype=torch.int64) % 3
                    counts = torch.tensor(
                        [257, 33, 509], device=self.device, dtype=torch.int32
                    )
                    for ratio in (1, 2):
                        visible = (positions + 1) // ratio
                        for lookup in (False, True):
                            options = (
                                dict(request_ids=ids, request_key_counts=counts)
                                if lookup
                                else {}
                            )
                            count = 1277 if lookup else 257
                            expected = self.sparse.prepare_plan(
                                view, visible, count, **options
                            )
                            actual = self.sparse.prepare_plan(
                                view, positions, count, positions_ratio=ratio, **options
                            )
                            self.assert_plans_equal(actual, expected)
                            if not lookup:
                                indices, end = reference_plan(view, visible, count)
                                torch.testing.assert_close(
                                    actual.sparse_indices.cpu(), indices
                                )
                                torch.testing.assert_close(actual.end.cpu(), end)
                    self.assertTrue((candidates[:, :3] == -999).all())
                    self.assertTrue((candidates[:, width + 3 :] == -999).all())

    def test_positions_plan_graph_replay_rebuilds_changed_bounds(self):
        rows, count = 17, 1025
        candidates = make_candidates(rows, 2048, count).to(self.device)
        positions = torch.arange(rows * 2, device=self.device, dtype=torch.int64)[::2]
        ids = torch.arange(rows, device=self.device, dtype=torch.int32) % 2
        counts = torch.tensor([513, 257], device=self.device, dtype=torch.int32)
        options = dict(request_ids=ids, request_key_counts=counts)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        call = lambda: self.sparse.prepare_plan(
            candidates, positions, count, positions_ratio=2, **options
        )
        with torch.cuda.stream(stream):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = call()
        for shift in (1100, -1200, 400):
            positions.add_(shift)
            counts.copy_(counts.flip(0))
            candidates[:, ::11] = -1
            graph.replay()
            expected = self.sparse.prepare_plan(
                candidates, (positions + 1) // 2, count, **options
            )
            self.assert_plans_equal(captured, expected)

    def inputs(self, rows, count, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        q = torch.randn(
            rows, 32, 128, generator=generator, device=self.device
        ).bfloat16()
        k = torch.randn(count, 128, generator=generator, device=self.device).bfloat16()
        q[:, 0] *= 8
        q[:, 1] *= 0.125
        k[::2] *= 4
        weights = torch.randn(rows, 32, generator=generator, device=self.device) / 64
        q_payload, q_sf = dense_indexer.quantize_indexer_q(q)
        k_payload, k_sf = dense_indexer.quantize_indexer_k_reference(k)
        return (
            q,
            k,
            weights,
            q_payload,
            q_sf,
            dense_indexer.PrefillIndexerKeys(k_payload, k_sf),
        )

    def dense_bf16(self, q_payload, q_sf, keys, weights, plan):
        return score_backend._deep_gemm.fp8_fp4_mqa_logits(
            (q_payload, q_sf),
            (keys.quant, keys.scale),
            weights.bfloat16(),
            plan.row_ks,
            plan.row_ke,
            clean_logits=True,
            max_seqlen_k=0,
            logits_dtype=torch.bfloat16,
        )

    def test_batched_scores_and_remap_equal_per_request_including_partial_blocks(self):
        counts, offsets = (257, 509, 16), (0, 264, 776)
        request_ids = [1, 0, 2, 1, 0, 2, 0, 1]
        count, rows = 792, len(request_ids)
        _, _, weights, q, sf, keys = self.inputs(rows, count, 912)
        visible = torch.tensor(
            [700, 257, 0, 9, -1, 16, 1, 509], device=self.device, dtype=torch.int64
        )
        candidates = make_candidates(rows, 64, 509, 917).to(self.device)
        starts = torch.tensor(
            [offsets[i] for i in request_ids], device=self.device, dtype=torch.int32
        )
        lengths = torch.tensor(
            [counts[i] for i in request_ids], device=self.device, dtype=torch.int32
        )
        plan = self.sparse.prepare_plan(
            candidates, visible, count, row_key_offsets=starts, row_key_counts=lengths
        )
        self.assertIsNotNone(plan)
        actual = self.sparse.score(q, sf, keys, weights, plan)
        columns = (
            torch.tensor(
                [0, 7, 8, 15, 16, -1, 511, 512], dtype=torch.int32, device=self.device
            )
            .expand(rows, -1)
            .contiguous()
        )
        mapped = self.sparse.remap(columns, plan, logits=actual)
        for row, request in enumerate(request_ids):
            start, length = offsets[request], counts[request]
            local_keys = dense_indexer.PrefillIndexerKeys(
                keys.quant[start : start + length], keys.scale[start : start + length]
            )
            local = self.sparse.prepare_plan(
                candidates[row : row + 1], visible[row : row + 1], length
            )
            expected = self.sparse.score(
                q[row : row + 1],
                sf[row : row + 1],
                local_keys,
                weights[row : row + 1],
                local,
            )
            end = int(local.end[0])
            torch.testing.assert_close(
                plan.end[row : row + 1], local.end, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[row, :end], expected[0, :end], rtol=0, atol=0
            )
            reference = self.sparse.remap(
                columns[row : row + 1], local, logits=expected
            )
            torch.testing.assert_close(mapped[row : row + 1], reference, rtol=0, atol=0)
        # The plan snapshots bounds; later mutation of source descriptors cannot
        # redirect a retained plan to another request's K range.
        starts.fill_(0)
        lengths.fill_(0)
        torch.testing.assert_close(
            self.sparse.remap(columns, plan, logits=actual), mapped, rtol=0, atol=0
        )

    def test_batched_invalid_spans_are_empty_without_neighbor_reads(self):
        starts = torch.tensor(
            [-8, 1, 24, 16, 2147483640, 32], device=self.device, dtype=torch.int32
        )
        counts = torch.tensor(
            [8, 8, 16, -1, 2147483647, 0], device=self.device, dtype=torch.int32
        )
        candidates = torch.zeros(6, 64, device=self.device, dtype=torch.int32)
        visible = torch.full((6,), 32, device=self.device, dtype=torch.int32)
        plan = self.sparse.prepare_plan(
            candidates, visible, 32, row_key_offsets=starts, row_key_counts=counts
        )
        self.assertTrue((plan.end == 0).all())
        self.assertTrue((plan.row_ke == 0).all())
        self.assertTrue((plan.sparse_indices == 0).all())
        columns = torch.zeros(6, 512, device=self.device, dtype=torch.int32)
        self.assertTrue((self.sparse.remap(columns, plan) == -1).all())

    def test_lookup_ragged_batch_no_host_upload_matches_each_request(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as selector

        if not selector.is_available(self.device):
            self.skipTest("native DeepSelect unavailable")
        counts, row_counts = (257, 509, 16, 1024), (3, 4097, 2, 19)
        starts = (0, 512, 1024, 1280)
        rows = sum(row_counts)
        _, _, weights, q, sf, keys = self.inputs(rows, 2304, 935)
        globals_by_req = [
            (
                None,
                dense_indexer.PrefillIndexerKeys(
                    keys.quant[a : a + n], keys.scale[a : a + n]
                ),
            )
            for a, n in zip(starts, counts)
        ]
        slices, cursor, ids, positions = [], 0, [], []
        for request, (count, size) in enumerate(zip(counts, row_counts)):
            slices.append(slice(cursor, cursor + size))
            cursor += size
            ids.extend([request] * size)
            positions.extend([max(0, count - 1 - i % 17) for i in range(size)])
        ids = torch.tensor(ids, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions, dtype=torch.int32, device=self.device)
        count_table = torch.tensor(counts, dtype=torch.int32, device=self.device)
        candidates = (
            torch.arange(64, device=self.device, dtype=torch.int32)
            .expand(rows, -1)
            .contiguous()
        )
        expected_logits, expected_end, expected_out = [], [], []
        for (_, local_keys), span in zip(globals_by_req, slices):
            for first in range(span.start, span.stop, 4096):
                part = slice(first, min(first + 4096, span.stop))
                plan = self.sparse.prepare_plan(
                    candidates[part], positions[part] + 1, len(local_keys)
                )
                logits = self.sparse.score(
                    q[part], sf[part], local_keys, weights[part], plan
                )
                columns = selector.try_select_sparse_tokens(logits, plan.end)
                expected_logits.append(logits)
                expected_end.append(plan.end)
                expected_out.append(self.sparse.remap(columns, plan, logits=logits))
        expected_logits = torch.cat(expected_logits)
        expected_end = torch.cat(expected_end)
        expected_out = torch.cat(expected_out)
        output = torch.full_like(expected_out, 1234567)
        shared = {"candidates": candidates}
        seen_logits, seen_end = [], []
        score_fn = self.sparse.score

        def capture(*args):
            result = score_fn(*args)
            seen_logits.append(result)
            seen_end.append(args[-1].end)
            return result

        with patch.object(
            torch, "tensor", side_effect=AssertionError("critical H2D metadata")
        ), patch.object(
            torch.Tensor, "cpu", side_effect=AssertionError("D2H metadata")
        ), patch.object(
            self.sparse, "score", side_effect=capture
        ), patch.object(
            self.sparse, "prepare_plan", wraps=self.sparse.prepare_plan
        ) as prepare:
            self.assertTrue(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    1,
                    candidates,
                    8,
                    512,
                    shared,
                    output,
                    req_ids=ids,
                    key_counts=count_table,
                )
            )
            self.assertEqual(prepare.call_count, 2)
        actual = torch.cat(seen_logits)
        valid = torch.arange(512, device=self.device)[None] < expected_end[:, None]
        torch.testing.assert_close(torch.cat(seen_end), expected_end, rtol=0, atol=0)
        torch.testing.assert_close(
            actual[valid], expected_logits[valid], rtol=0, atol=0
        )
        torch.testing.assert_close(output, expected_out, rtol=0, atol=0)

        # Invalid IDs and IDs belonging to another group are empty, even when
        # their storage-relative offsets would accidentally fit this slab.
        bad_ids = torch.tensor([-1, 4, 0, 3], device=self.device, dtype=torch.int32)
        bad = self.sparse.prepare_plan(
            candidates[:4],
            positions[:4] + 1,
            768,
            request_ids=bad_ids,
            request_key_counts=count_table,
            request_start=1,
            request_stop=3,
        )
        self.assertTrue((bad.end == 0).all())

    def test_batch_entry_cache_groups_and_outputs_match_per_request(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as selector

        if not selector.is_available(self.device):
            self.skipTest("native DeepSelect unavailable")
        batch, per_request, count = 4, 1025, 1024
        rows = batch * per_request
        request_ids = torch.arange(
            batch, device=self.device, dtype=torch.int32
        ).repeat_interleave(per_request)
        key_counts = torch.full((batch,), count, device=self.device, dtype=torch.int32)
        _, _, weights, q, sf, keys = self.inputs(rows, batch * count, 930)
        slices = [slice(i * per_request, (i + 1) * per_request) for i in range(batch)]
        globals_by_req = [
            (
                None,
                dense_indexer.PrefillIndexerKeys(
                    keys.quant[i * count : (i + 1) * count],
                    keys.scale[i * count : (i + 1) * count],
                ),
            )
            for i in range(batch)
        ]
        positions = torch.full(
            (rows,), count - 1, device=self.device, dtype=torch.int32
        )
        candidates = (
            torch.arange(64, device=self.device, dtype=torch.int32)
            .expand(rows, -1)
            .contiguous()
        )
        # Exactly 512 candidates: DeepSelect emits the complete prefix, so no
        # tied-cutoff membership/order ambiguity hides remap or scorer bugs.
        expected = torch.arange(512, device=self.device, dtype=torch.int32).expand(
            rows, -1
        )
        shared = {"candidates": candidates}
        out = torch.empty_like(expected)
        with patch.object(
            self.sparse, "prepare_plan", wraps=self.sparse.prepare_plan
        ) as prepare:
            self.assertTrue(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    1,
                    candidates,
                    8,
                    512,
                    shared,
                    out,
                    req_ids=request_ids,
                    key_counts=key_counts,
                )
            )
            self.assertEqual(prepare.call_count, 2)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            retained = shared["prefill_sparse_plans"][2]
            self.assertTrue(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    1,
                    candidates,
                    8,
                    512,
                    shared,
                    out,
                    req_ids=request_ids,
                    key_counts=key_counts,
                )
            )
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(shared["prefill_sparse_plans"][2], retained)
            # A new global source with the same logical K layout shares plan
            # metadata, but must score the new source's bytes, not old K.
            new_quant, new_scale = torch.zeros_like(keys.quant), keys.scale.clone()
            new_globals = [
                (
                    None,
                    dense_indexer.PrefillIndexerKeys(
                        new_quant[i * count : (i + 1) * count],
                        new_scale[i * count : (i + 1) * count],
                    ),
                )
                for i in range(batch)
            ]
            score_impl = self.sparse.score

            def score_new_source(*args):
                self.assertEqual(
                    args[2].quant.untyped_storage().data_ptr(),
                    new_quant.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    args[2].scale.untyped_storage().data_ptr(),
                    new_scale.untyped_storage().data_ptr(),
                )
                scores = score_impl(*args)
                self.assertTrue((scores == 0).all())
                return scores

            with patch.object(self.sparse, "score", side_effect=score_new_source):
                self.assertTrue(
                    self.sparse.try_batched_sparse(
                        q,
                        sf,
                        weights,
                        new_globals,
                        slices,
                        positions,
                        1,
                        candidates,
                        8,
                        512,
                        shared,
                        out,
                        req_ids=request_ids,
                        key_counts=key_counts,
                    )
                )
            self.assertEqual(prepare.call_count, 2)
            self.assertEqual(shared["prefill_sparse_plans"][2], retained)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

            # Equal-valued replacement descriptors are different snapshots;
            # each must rebuild even when shapes and logical layouts match.
            for alternate_positions, alternate_ids, alternate_counts in (
                (positions.clone(), request_ids, key_counts),
                (positions, request_ids.clone(), key_counts),
                (positions, request_ids, key_counts.clone()),
            ):
                calls = prepare.call_count
                self.assertTrue(
                    self.sparse.try_batched_sparse(
                        q,
                        sf,
                        weights,
                        globals_by_req,
                        slices,
                        alternate_positions,
                        1,
                        candidates,
                        8,
                        512,
                        shared,
                        out,
                        req_ids=alternate_ids,
                        key_counts=alternate_counts,
                    )
                )
                self.assertEqual(prepare.call_count, calls + 2)
            calls = prepare.call_count
            self.assertTrue(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    2,
                    candidates,
                    8,
                    512,
                    shared,
                    out,
                    req_ids=request_ids,
                    key_counts=key_counts,
                )
            )
            self.assertEqual(prepare.call_count, calls + 2)
        replacement = candidates.clone()
        shared["candidates"] = replacement
        with patch.object(
            self.sparse, "prepare_plan", wraps=self.sparse.prepare_plan
        ) as prepare:
            self.assertTrue(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    1,
                    replacement,
                    8,
                    512,
                    shared,
                    out,
                    req_ids=request_ids,
                    key_counts=key_counts,
                )
            )
            self.assertEqual(prepare.call_count, 2)
        with patch.object(
            self.sparse,
            "score",
            side_effect=AssertionError("invalid preflight launched"),
        ):
            self.assertFalse(
                self.sparse.try_batched_sparse(
                    q,
                    sf,
                    weights,
                    globals_by_req,
                    slices,
                    positions,
                    1,
                    replacement,
                    8,
                    511,
                    shared,
                    out,
                    req_ids=request_ids,
                    key_counts=key_counts,
                )
            )

    def test_metadata_and_scores_match_dense_bf16_for_ragged_boundaries(self):
        cases = [
            (1, 1, 64),
            (3, 255, 64),
            (7, 256, 64),
            (13, 257, 64),
            (13, 509, 64),
            (13, 1021, 64),
            (13, 16555, 2048),
        ]
        for rows, count, width in cases:
            with self.subTest(rows=rows, count=count, candidates=width):
                visible = torch.tensor(
                    (
                        [
                            0,
                            1,
                            7,
                            8,
                            9,
                            255,
                            256,
                            257,
                            count - 1,
                            count,
                            count + 3,
                            -1,
                            count,
                        ]
                        * 2
                    )[:rows],
                    dtype=torch.int32,
                    device=self.device,
                )
                candidates = make_candidates(rows, width, count).to(self.device)
                expected_indices, expected_end = reference_plan(
                    candidates, visible, count
                )
                plan = self.sparse.prepare_plan(candidates, visible, count)
                self.assertIsNotNone(plan)
                torch.testing.assert_close(plan.sparse_indices.cpu(), expected_indices)
                torch.testing.assert_close(plan.end.cpu(), expected_end)
                q, k, weights, q_payload, q_sf, keys = self.inputs(rows, count, 814)
                actual = self.sparse.score(q_payload, q_sf, keys, weights, plan)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertEqual(actual.shape, (rows, width * 8))
                expected_dense = self.dense_bf16(q_payload, q_sf, keys, weights, plan)
                logical = (
                    plan.sparse_indices.long()[:, :, None] * 8
                    + torch.arange(8, device=self.device)
                ).flatten(1)
                valid = (
                    torch.arange(width * 8, device=self.device)[None]
                    < plan.end[:, None]
                )
                expected = expected_dense.gather(1, logical.clamp(0, count - 1))
                # Sparse padded columns are unspecified; only the valid prefix
                # is input to DeepSelect. No tolerance hides score/remap bugs.
                torch.testing.assert_close(
                    actual[valid], expected[valid], rtol=0, atol=0
                )

    def test_sparse_scores_match_independent_step_bf16_oracle(self):
        rows, count = 3, 257
        q, k, weights, q_payload, q_sf, keys = self.inputs(rows, count, 815)
        candidates = (
            torch.arange(64, device=self.device, dtype=torch.int32)
            .expand(rows, -1)
            .contiguous()
        )
        visible = torch.full((rows,), count, device=self.device, dtype=torch.int32)
        plan = self.sparse.prepare_plan(candidates, visible, count)
        actual = self.sparse.score(q_payload, q_sf, keys, weights, plan)
        expected = reference_bf16_logits(q, k, weights)
        torch.testing.assert_close(actual[:, :count], expected, rtol=0, atol=0)

    def test_remap_filters_invalid_columns_nonfinite_and_keeps_output_stride(self):
        count = 257
        candidates = torch.full((3, 64), -1, device=self.device, dtype=torch.int32)
        candidates[0, :4] = torch.tensor([32, 0, 0, 2], device=self.device)
        candidates[1, :4] = torch.tensor([1, 0, 7, 32], device=self.device)
        visible = torch.tensor([257, 9, 0], device=self.device, dtype=torch.int32)
        plan = self.sparse.prepare_plan(candidates, visible, count)
        columns = torch.tensor(
            [
                [0, 7, 8, 16, 17, -1, 512, 10000],
                [0, 7, 8, 9, 10, -1, 512, 10000],
                [0, 1, 8, 9, 10, -1, 512, 10000],
            ],
            device=self.device,
            dtype=torch.int32,
        )
        logits = torch.ones((3, 512), device=self.device, dtype=torch.bfloat16)
        logits[0, 7] = torch.nan
        logits[0, 8] = torch.inf
        logits[1, 8] = -torch.inf
        output_storage = torch.full((3, 16), 991, device=self.device, dtype=torch.int32)
        expected = reference_remap(
            columns, plan.sparse_indices, plan.end, visible, count, logits
        )
        actual = self.sparse.remap(
            columns, plan, logits=logits, out=output_storage[:, :8]
        )
        torch.testing.assert_close(actual.cpu(), expected)
        self.assertEqual(actual.data_ptr(), output_storage.data_ptr())
        self.assertTrue((output_storage[:, 8:] == 991).all())

    def assert_selection_contract(self, logits, columns, selected, plan):
        expected = reference_remap(
            columns, plan.sparse_indices, plan.end, plan.row_ke, plan.key_count, logits
        )
        torch.testing.assert_close(selected.cpu(), expected)
        for row in range(logits.shape[0]):
            end = int(plan.end[row])
            chosen = columns[row][columns[row] >= 0].long()
            self.assertEqual(chosen.numel(), min(512, end))
            self.assertEqual(chosen.unique().numel(), chosen.numel())
            positions = selected[row][selected[row] >= 0]
            self.assertEqual(positions.unique().numel(), positions.numel())
            if not end:
                self.assertTrue((selected[row] == -1).all())
                continue
            cutoff = logits[row, :end].topk(min(512, end)).values[-1]
            self.assertTrue((logits[row, chosen] >= cutoff).all())

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_SPARSE_DEEPSELECT_GPU") == "1",
        "Native DeepSelect integration is opt-in",
    )
    def test_deepselect_cutoff_and_graph_replay_keep_logical_candidate_set(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as deepselect

        rows, count = 13, 32769
        q, k, weights, q_payload, q_sf, keys = self.inputs(rows, count, 816)
        candidates = make_candidates(rows, 2048, count, 817).to(self.device)
        visible = torch.tensor(
            [0, 1, 7, 8, 9, 511, 512, 513, count, count - 1, count, 1021, count],
            device=self.device,
            dtype=torch.int32,
        )
        plan = self.sparse.prepare_plan(candidates, visible, count)

        def select():
            logits = self.sparse.score(q_payload, q_sf, keys, weights, plan)
            columns = deepselect.try_select_sparse_tokens(logits, plan.end)
            self.assertIsNotNone(columns, "Requested native DeepSelect is unavailable")
            return logits, columns, self.sparse.remap(columns, plan, logits=logits)

        logits, columns, selected = select()
        self.assert_selection_contract(logits, columns, selected, plan)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                select()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits, columns, selected = select()
        torch.cuda.synchronize()
        for multiplier in (0.75, -1.0):
            weights.mul_(multiplier)
            logits.fill_(torch.nan)
            columns.fill_(999999)
            selected.fill_(999999)
            graph.replay()
            self.assert_selection_contract(logits, columns, selected, plan)


if __name__ == "__main__":
    unittest.main()
