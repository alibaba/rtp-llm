"""CPU integration contracts for forward-local sparse prefill selection."""

import os
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import attention_v41 as attention


class SparseSelectionIntegrationCPU(unittest.TestCase):
    def test_candidate_plan_backend_is_selected_inside_indexer(self):
        indexer = attention.sparse_prefill_indexer
        candidates = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
        visible = object()
        for cuda, capability, logical in (
            (True, (9, 0), True),
            (True, (9, 1), False),
            (True, (10, 0), False),
            (False, (9, 0), False),
        ):
            with self.subTest(cuda=cuda, capability=capability), patch.object(
                indexer, "is_cuda", return_value=cuda
            ), patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ) as get_capability, patch.object(
                indexer, "_plan_supported", return_value=False
            ) as supported:
                self.assertIsNone(indexer.prepare_plan(candidates, visible, 2048))
                supported.assert_called_once_with(
                    candidates, visible, 2048, 8, sm90=logical
                )
                if not cuda:
                    get_capability.assert_not_called()

    def test_prefill_indexer_does_not_select_sm90_on_hip(self):
        indexer = attention.prefill_indexer
        with patch.object(indexer, "is_cuda", return_value=False), patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ) as get_capability:
            self.assertFalse(indexer.is_supported(torch.device("cuda:0"), 32, 128))
            get_capability.assert_not_called()

    def test_cache_budget_key_isolation_and_candidate_identity(self):
        candidates = torch.zeros(3, 128, dtype=torch.int32)
        visible = torch.ones(3, dtype=torch.int32)
        shared = {"candidates": candidates}
        plans = [SimpleNamespace(nbytes=n) for n in (10, 14, 1, 1, 2)]
        keys = [(0, 0, 3, 1537, 1, 8), (1, 0, 3, 1537, 1, 8)]
        with patch.dict(
            os.environ, {"DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES": "24"}
        ), patch.object(
            attention.sparse_prefill_indexer, "prepare_plan", side_effect=plans
        ) as prepare:
            for index, key in enumerate(keys):
                actual = attention._prefill_sparse_plan(
                    shared, key, candidates, visible, 1537, 8
                )
                self.assertIs(actual, plans[index])
                self.assertIs(
                    attention._prefill_sparse_plan(
                        shared, key, candidates, visible, 1537, 8
                    ),
                    actual,
                )
            self.assertEqual(shared["prefill_sparse_plans"][2], 24)
            # Compression differs: never alias the same request/chunk tuple.
            uncached_key = (0, 0, 3, 1537, 2, 8)
            for index in (2, 3):
                self.assertIs(
                    attention._prefill_sparse_plan(
                        shared, uncached_key, candidates, visible, 1537, 8
                    ),
                    plans[index],
                )
            self.assertNotIn(uncached_key, shared["prefill_sparse_plans"][1])
            # Publishing a new tensor invalidates even an identical key.
            shared["candidates"] = candidates.clone()
            actual = attention._prefill_sparse_plan(
                shared, keys[0], shared["candidates"], visible, 1537, 8
            )
            self.assertIs(actual, plans[4])
            self.assertEqual(shared["prefill_sparse_plans"][2], 2)
            self.assertEqual(prepare.call_count, 5)

    def test_zero_budget_rebuilds_and_rejected_plan_is_not_cached(self):
        candidates = torch.zeros(1, 128, dtype=torch.int32)
        visible = torch.ones(1, dtype=torch.int32)
        shared = {"candidates": candidates}
        key = (0, 0, 1, 1537, 1, 8)
        plan = SimpleNamespace(nbytes=1)
        with patch.dict(
            os.environ, {"DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES": "0"}
        ), patch.object(
            attention.sparse_prefill_indexer,
            "prepare_plan",
            side_effect=[plan, plan, None],
        ) as prepare:
            for _ in range(2):
                self.assertIs(
                    attention._prefill_sparse_plan(
                        shared, key, candidates, visible, 1537, 8
                    ),
                    plan,
                )
            with self.assertRaisesRegex(RuntimeError, "rejected a supported"):
                attention._prefill_sparse_plan(
                    shared, key, candidates, visible, 1537, 8
                )
            self.assertEqual(prepare.call_count, 3)
            self.assertEqual(shared["prefill_sparse_plans"][1], {})
            self.assertEqual(shared["prefill_sparse_plans"][2], 0)

    def test_begin_forward_invalidates_only_at_first_layer(self):
        marker = object()
        shared = {
            "layers": {0: None, 26: None},
            "prefill_sparse_plans": marker,
            "prefill_sparse_candidates": True,
            "candidates": torch.zeros(1, 128),
        }
        for layer, kept in ((26, True), (0, False)):
            attention.AttentionV41FP8._begin_forward(
                SimpleNamespace(layer_id=layer, _shared_attention=shared)
            )
            self.assertEqual("prefill_sparse_plans" in shared, kept)
        self.assertNotIn("prefill_sparse_candidates", shared)
        self.assertIsNone(shared["candidates"])

    def test_candidate_source_republication_drops_previous_plans(self):
        old_candidates = torch.ones(2, 128, dtype=torch.int32)
        shared = {
            "global": {20: [(None, torch.empty(17, 128))]},
            "candidates": old_candidates,
            "prefill_sparse_plans": [old_candidates, {"old": object()}, 100],
            "prefill_candidate_mask": object(),
        }
        source = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=20,
            kv_source_layer_id=20,
            layer_id=20,
            compress_ratio=1,
            index_topk=512,
            index_n_heads=32,
            v41_config=dict(
                candidate_source_layer_id=20,
                candidate_topk_blocks=128,
                candidate_block_size=8,
            ),
            _shared_attention=shared,
        )
        with patch.object(
            attention.prefill_deepselect, "is_available", return_value=True
        ):
            selected = attention.AttentionV41FP8._select_indices(
                source, torch.zeros(2, 1), None, torch.tensor([0, 16]), torch.zeros(2)
            )
        self.assertNotIn("prefill_sparse_plans", shared)
        self.assertNotIn("prefill_candidate_mask", shared)
        self.assertIsNone(shared["candidates"])
        self.assertTrue(shared["prefill_sparse_candidates"])
        self.assertEqual(int((selected[0] >= 0).sum()), 1)
        self.assertEqual(int((selected[1] >= 0).sum()), 17)

    def _check_selection(self, ragged, ratio, *, logical_output=False):
        rows = 7
        req_ids = torch.tensor([1, 0, 1, 0, 1, 0, 1] if ragged else [0] * rows)
        lengths = [1537, 1793] if ragged else [1793]
        visible = torch.tensor([0, 1, 9, 511, 512, 769, 1021])
        positions = visible * ratio - 1
        qr = torch.arange(1, rows + 1).float()[:, None]
        x = qr.clone()
        candidates = torch.arange(128, dtype=torch.int32).expand(rows, -1).clone()
        candidates[:, 0] = -1
        candidates[:, 2] = 1  # duplicate input must not duplicate logical tokens
        shared = {
            "global": {
                20: [
                    (
                        None,
                        attention.prefill_indexer.PrefillIndexerKeys(
                            torch.zeros(n, 64, dtype=torch.int8),
                            torch.zeros(n, dtype=torch.int32),
                        ),
                    )
                    for n in lengths
                ]
            },
            "candidates": candidates,
        }
        attn = SimpleNamespace(
            is_index_source=True,
            index_source_layer_id=26,
            kv_source_layer_id=20,
            layer_id=26,
            index_topk=512,
            index_n_heads=32,
            index_head_dim=128,
            rope_head_dim=0,
            compress_ratio=ratio,
            index_wq=None,
            index_weights=torch.ones(32, 1),
            freqs_cis=torch.ones(2048, 1, dtype=torch.complex64),
            _lin=lambda weight, value: value.expand(-1, 32 * 128),
            v41_config=dict(
                candidate_source_layer_id=20,
                candidate_topk_blocks=128,
                candidate_block_size=8,
            ),
            _shared_attention=shared,
        )
        destinations = []

        def prepare(cand, vis, key_count, block_size):
            logical = torch.full((len(cand), 1024), -1, dtype=torch.int32)
            ends = []
            for row, (blocks, length) in enumerate(zip(cand.tolist(), vis.tolist())):
                allowed = sorted(
                    {
                        b * 8 + offset
                        for b in blocks
                        for offset in range(8)
                        if b >= 0 and b * 8 + offset < min(length, key_count)
                    }
                )
                logical[row, : len(allowed)] = torch.tensor(allowed, dtype=torch.int32)
                ends.append(len(allowed))
            return SimpleNamespace(
                logical=logical, end=torch.tensor(ends, dtype=torch.int32), nbytes=128
            )

        def score(q, sf, keys, weights, plan):
            # Row-specific direction catches gather/scatter and chunk mistakes.
            sign = torch.where(q[:, 0, 0].int() % 2 == 0, 1, -1)
            return (plan.logical.float() * sign[:, None]).masked_fill(
                plan.logical < 0, -torch.inf
            )

        def score_logical(q, sf, keys, weights, plan):
            compact = score(q, sf, keys, weights, plan)
            logits = torch.full((len(q), len(keys)), -torch.inf)
            for row in range(len(q)):
                valid = plan.logical[row] >= 0
                logits[row, plan.logical[row, valid].long()] = compact[row, valid]
            return logits

        def select(logits, end):
            values, columns = logits.topk(512)
            return torch.where(values.isfinite(), columns, -1).int()

        def remap(columns, plan, *, logits, out):
            result = plan.logical.gather(1, columns.long().clamp_min(0))
            result = result.masked_fill(columns < 0, -1)
            destinations.append(out)
            return result if out is None else out.copy_(result)

        with ExitStack() as stack:
            stack.enter_context(
                patch.dict(os.environ, {"DSV41_SPARSE_PREFILL_PLAN_MAX_BYTES": "4096"})
            )
            stack.enter_context(
                patch.object(attention, "rope_only", side_effect=lambda q, *a: q)
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer,
                    "quantize_indexer_q",
                    side_effect=lambda q: (q, torch.zeros(rows, 32)),
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_deepselect, "is_available", return_value=True
                )
            )
            stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer,
                    "is_supported",
                    side_effect=lambda *args, **kwargs: kwargs.get(
                        "logical_output", False
                    )
                    == logical_output,
                )
            )
            stack.enter_context(
                patch.object(attention.sparse_prefill_indexer, "MAX_CHUNK_ROWS", 2)
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer, "logits_chunk_rows", return_value=2
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_topk, "try_select_tokens", return_value=None
                )
            )
            logical_score_mock = stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer,
                    "score_logical",
                    side_effect=score_logical,
                )
            )
            prepare_mock = stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer,
                    "prepare_plan",
                    side_effect=prepare,
                )
            )
            compact_score_mock = stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer, "score", side_effect=score
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_deepselect,
                    "try_select_sparse_tokens",
                    side_effect=select,
                )
            )
            remap_mock = stack.enter_context(
                patch.object(
                    attention.sparse_prefill_indexer, "remap", side_effect=remap
                )
            )
            stack.enter_context(
                patch.object(
                    attention.prefill_indexer,
                    "score_indexer_chunk",
                    side_effect=AssertionError("Unexpected dense fallback"),
                )
            )
            stack.enter_context(
                patch.object(
                    attention,
                    "_apply_prefill_candidates",
                    side_effect=AssertionError("Unexpected dense mask"),
                )
            )
            outputs = []
            for layer in (26, 32, 38, 39):
                attn.layer_id = layer
                outputs.append(
                    attention.AttentionV41FP8._select_indices(
                        attn, x, qr, positions, req_ids
                    )
                )
            self.assertEqual(prepare_mock.call_count, 4)
            if logical_output:
                self.assertEqual(logical_score_mock.call_count, 16)
                compact_score_mock.assert_not_called()
                remap_mock.assert_not_called()
            else:
                logical_score_mock.assert_not_called()
        expected = torch.full((rows, 512), -1, dtype=torch.int32)
        for row in range(rows):
            allowed = sorted(
                {
                    b * 8 + o
                    for b in candidates[row].tolist()
                    for o in range(8)
                    if b >= 0 and b * 8 + o < int(visible[row])
                },
                reverse=(row + 1) % 2 == 0,
            )[:512]
            expected[row, : len(allowed)] = torch.tensor(allowed, dtype=torch.int32)
        for actual in outputs:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(all((out is None) == ragged for out in destinations))
        self.assertIs(shared["topk"][39], outputs[-1])

    def test_logical_candidate_scores_skip_sparse_remap_and_second_mask(self):
        for ragged in (False, True):
            for ratio in (1, 2):
                with self.subTest(ragged=ragged, ratio=ratio):
                    self._check_selection(ragged, ratio, logical_output=True)

    def test_single_request_slices_write_in_place_and_reuse_four_consumers(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self._check_selection(False, ratio)

    def test_ragged_requests_scatter_and_isolate_request_plans(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self._check_selection(True, ratio)


if __name__ == "__main__":
    unittest.main()
