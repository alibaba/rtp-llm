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

    def test_cpu_and_disabled_gates_do_not_query_cuda_or_load_deepgemm(self):
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
            for flag in ("0", "1"):
                with patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_INDEXER": flag}):
                    self.assertIsNone(sparse.prepare_plan(candidates, visible, 257))

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

    def setUp(self):
        self.flags = patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_INDEXER": "1"})
        self.flags.start()
        self.addCleanup(self.flags.stop)

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
