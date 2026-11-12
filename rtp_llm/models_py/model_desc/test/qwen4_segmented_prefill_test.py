"""CPU policy tests; CUDA numerical checks run in the EP8 model harness."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

SOURCE = Path(
    os.environ.get("QWEN_SEGMENT_SOURCE")
    or str(
        Path(__file__).resolve().parents[2] / "modules/qwen4_exp/segmented_prefill.py"
    )
)
spec = importlib.util.spec_from_file_location("segmented_prefill_under_test", SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SegmentPolicyTest(unittest.TestCase):
    def test_exact_large_rows(self):
        budget = module.SegmentBudget(2, 6144 + 8192)
        self.assertTrue(budget.admit((6144, "bf16")))
        self.assertTrue(budget.admit((8192, "bf16")))
        self.assertEqual(budget.tokens, 14336)
        self.assertFalse(budget.admit((6145, "bf16")))

    def test_cached_key_does_not_consume_budget(self):
        budget = module.SegmentBudget(1, 8192)
        for _ in range(10):
            self.assertTrue(budget.admit((8192, "bf16")))
        self.assertEqual(budget.tokens, 8192)

    def test_token_overflow_is_transactional(self):
        budget = module.SegmentBudget(4, 8192)
        self.assertTrue(budget.admit((6144, "bf16")))
        self.assertFalse(budget.admit((2049, "bf16")))
        self.assertEqual(budget.tokens, 6144)
        self.assertTrue(budget.admit((2048, "bf16")))

    def test_entry_limit_without_eviction(self):
        budget = module.SegmentBudget(1, 32768)
        self.assertTrue(budget.admit((512, "bf16")))
        self.assertFalse(budget.admit((2048, "bf16")))
        self.assertEqual(budget.keys, {(512, "bf16")})

    def test_invalid_budgets(self):
        for entries, tokens in [(0, 2), (2, 0), (-1, 4)]:
            with self.assertRaises(ValueError):
                module.SegmentBudget(entries, tokens)

    def test_zero_rows(self):
        budget = module.SegmentBudget()
        self.assertFalse(budget.admit((0, "bf16")))

    def test_default_rows_and_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            admission = module.SegmentAdmission()
        self.assertEqual(admission.rows, {512, 2048, 6144, 8192})
        for rows in sorted(admission.rows):
            self.assertTrue(admission.budget.admit((rows, "bf16")))

    def test_invalid_rows(self):
        for rows in ["32", "8193", "0", ""]:
            with patch.dict(os.environ, {"QWEN_PREFILL_SEGMENT_ROWS": rows}):
                with self.assertRaises(ValueError):
                    module.SegmentAdmission()

    def test_cpu_fallback_does_not_collect_or_capture(self):
        segment = module.StaticPrefillSegment(0, "test", module.SegmentAdmission())
        x = torch.ones(512, 4)
        with patch.object(module, "require_rank_agreement") as consensus:
            output = segment.run(lambda t: t + 2, x)
            consensus.assert_not_called()
        self.assertTrue(torch.equal(output, x + 2))
        self.assertEqual(segment.graphs, {})

    def test_rank_agreement(self):
        def gather(peers, value):
            peers[:] = [value, value]

        with (
            patch.object(module.dist, "get_world_size", return_value=2),
            patch.object(module.dist, "all_gather_object", side_effect=gather),
        ):
            module.require_rank_agreement((6144, "bf16"))

    def test_rank_divergence_fails_before_capture(self):
        def gather(peers, value):
            peers[:] = [value, (8192, "bf16")]

        with (
            patch.object(module.dist, "get_world_size", return_value=2),
            patch.object(module.dist, "all_gather_object", side_effect=gather),
        ):
            with self.assertRaisesRegex(RuntimeError, "rank shape/cache disagreement"):
                module.require_rank_agreement((6144, "bf16"))

    def test_remote_low_memory_rejects_and_preserves_eager_output(self):
        admission = module.SegmentAdmission()
        segment = module.StaticPrefillSegment(0, "test", admission)
        x = torch.ones(512, 4)
        stream = SimpleNamespace(cuda_stream=7)
        with (
            patch.object(admission, "supports", return_value=True),
            patch.object(module, "require_rank_agreement"),
            patch.object(module.torch.cuda, "current_stream", return_value=stream),
            patch.object(
                module.torch.cuda, "mem_get_info", return_value=(8 * 2**30, 16 * 2**30)
            ),
            patch.object(
                module.dist, "all_reduce", side_effect=lambda t, op: t.fill_(0)
            ) as vote,
        ):
            for _ in range(2):
                result = segment.run(lambda t: t + 3, x)
                self.assertTrue(torch.equal(result, x + 3))
            self.assertEqual(vote.call_count, 1)
        self.assertEqual(len(segment.rejected), 1)
        self.assertFalse(segment.graphs)

    def test_private_pool_residency_includes_inactive_bytes_and_both_paths(self):
        snapshot = [
            {"segment_pool_id": (0, 1), "total_size": 1024},
            {"segment_pool_id": (0, 2), "total_size": 2048},
            {"segment_pool_id": (0, 0), "total_size": 100000},
        ]
        self.assertEqual(
            module.graph_memory_bytes(snapshot, {(0, 1), (0, 2)}, 256), 3328
        )

    def test_unknown_pool_ownership_fails_instead_of_undercounting(self):
        with self.assertRaisesRegex(RuntimeError, "pool ownership"):
            module.graph_memory_bytes([{"total_size": 1024}], {(0, 1)}, 0)

    def test_static_and_moe_shape_budget_is_shared_and_transactional(self):
        with patch.dict(os.environ, {"QWEN_PREFILL_GRAPH_MAX_ENTRIES": "1"}):
            admission = module.SegmentAdmission()

        def tensor(rows):
            return SimpleNamespace(
                is_cuda=True,
                dtype=torch.bfloat16,
                ndim=2,
                shape=(rows, 4),
                is_contiguous=lambda: True,
                requires_grad=False,
            )

        with (
            patch.object(module, "require_rank_agreement"),
            patch.object(
                module.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            self.assertTrue(admission.supports(tensor(768), moe=True))
            self.assertFalse(admission.supports(tensor(512)))
            self.assertTrue(admission.supports(tensor(768), moe=True))
        self.assertEqual(admission.budget.tokens, 768)
        self.assertFalse(admission.segment_budget.keys)

    def test_non_hot_shapes_never_promote_after_repeated_requests(self):
        admission = module.SegmentAdmission()
        with patch.object(module, "require_rank_agreement") as vote:
            for rows in (1536, 4608, 5376):
                for _ in range(20):
                    self.assertFalse(
                        admission.supports(SimpleNamespace(shape=(rows, 4)), moe=True)
                    )
            vote.assert_not_called()
        self.assertFalse(admission.budget.keys)

    def test_overlapping_hot_shapes_are_rejected(self):
        with patch.dict(os.environ, {"QWEN_PREFILL_GRAPH_MOE_ROWS": "512"}):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                module.SegmentAdmission()

    def test_invalid_memory_budget(self):
        for value in ("0", "-1", "512"):
            with patch.dict(os.environ, {"QWEN_PREFILL_GRAPH_MAX_MEMORY_MIB": value}):
                with self.assertRaises(ValueError):
                    module.SegmentAdmission()

    def test_budget_closes_cold_admission_without_touching_existing_graphs(self):
        admission = module.SegmentAdmission()
        admission.memory_used = admission.max_memory - admission.capture_reserve + 1
        with (
            patch.object(
                module.torch.cuda,
                "mem_get_info",
                return_value=(100 * 2**30, 140 * 2**30),
            ),
            patch.object(module.dist, "all_reduce") as vote,
        ):
            self.assertFalse(admission.allow_capture(torch.device("cpu"), 0))
            self.assertFalse(admission.allow_capture(torch.device("cpu"), 0))
            self.assertEqual(vote.call_count, 1)
        self.assertTrue(admission.capture_closed)

    def test_remote_rank_over_budget_fails_closed(self):
        admission = module.SegmentAdmission()
        with (
            patch.object(module.torch.cuda, "memory_snapshot", return_value=[]),
            patch.object(
                module.dist,
                "all_reduce",
                side_effect=lambda t, op: t.fill_(admission.max_memory + 1),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
                admission.record_capture(torch.device("cpu"))

    def test_closed_budget_still_replays_existing_graph_with_new_inputs(self):
        admission = module.SegmentAdmission()
        admission.capture_closed = True
        segment = module.StaticPrefillSegment(0, "hc_attn", admission)
        x = torch.ones(512, 4)
        static = torch.empty_like(x)
        output = torch.empty_like(x)
        key = ((tuple(x.shape), tuple(x.stride()), x.dtype, x.device),)
        graph = SimpleNamespace(replay=lambda: output.copy_(static + 1))
        segment.graphs[key] = (graph, (static,), output, None)

        def forbidden_eager(*args):
            self.fail("An admitted graph must remain usable when cold admission closes")

        with (
            patch.object(admission, "supports", return_value=True),
            patch.object(admission, "allow_capture") as cold,
            patch.object(module.dist, "all_reduce") as collective,
            patch.object(
                module.torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=7),
            ),
        ):
            for value in (3, 7):
                x.fill_(value)
                result = segment.run(forbidden_eager, x)
                self.assertTrue(torch.equal(result, x + 1))
            cold.assert_not_called()
            collective.assert_not_called()

    def test_static_subcap_rejection_does_not_charge_shared_budget(self):
        with patch.dict(os.environ, {"QWEN_PREFILL_SEGMENT_MAX_ENTRIES": "1"}):
            admission = module.SegmentAdmission()

        def tensor(rows):
            return SimpleNamespace(
                is_cuda=True,
                dtype=torch.bfloat16,
                ndim=2,
                shape=(rows, 4),
                is_contiguous=lambda: True,
                requires_grad=False,
            )

        with (
            patch.object(module, "require_rank_agreement"),
            patch.object(
                module.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            self.assertTrue(admission.supports(tensor(512)))
            self.assertFalse(admission.supports(tensor(2048)))
            self.assertTrue(admission.supports(tensor(768), moe=True))
        self.assertEqual(admission.budget.tokens, 1280)
        self.assertEqual(admission.segment_budget.tokens, 512)

    def test_input_storage_is_charged_once_even_with_multiple_references(self):
        admission = module.SegmentAdmission()
        x = torch.ones(4, 5)
        admission.input_slots = {"first": x, "second": x.view(5, 4)}
        admission.pools = {(512, x.dtype, x.device): (0, 1)}
        with (
            patch.object(
                module.torch.cuda,
                "memory_snapshot",
                return_value=[
                    {"segment_pool_id": (0, 1), "total_size": 1024},
                    {"segment_pool_id": (0, 0), "total_size": 4096},
                ],
            ),
            patch.object(module.dist, "all_reduce"),
        ):
            admission.record_capture(x.device)
        self.assertEqual(admission.memory_used, 1024 + x.untyped_storage().nbytes())

    def test_stream_switch_fails_before_replay(self):
        segment = module.StaticPrefillSegment(
            0, "test", SimpleNamespace(supports=lambda x: True)
        )
        segment.stream_id = 7
        with patch.object(
            module.torch.cuda,
            "current_stream",
            return_value=SimpleNamespace(cuda_stream=8),
        ):
            with self.assertRaisesRegex(RuntimeError, "serialized main stream"):
                segment.run(lambda t: t, torch.ones(512, 4))

    def test_strided_auxiliary_input_falls_back(self):
        segment = module.StaticPrefillSegment(
            0, "test", SimpleNamespace(supports=lambda x: True)
        )
        x = torch.ones(512, 4)
        auxiliary = torch.arange(4096).reshape(512, 8).float()[:, ::2]
        with patch.object(module.torch.cuda, "current_stream") as stream:
            result = segment.run(lambda a, b: a + b, x, auxiliary)
            stream.assert_not_called()
        self.assertTrue(torch.equal(result, x + auxiliary))

    def test_static_slots_share_across_layers_but_not_stages_or_shapes(self):
        admission = module.SegmentAdmission()
        x = torch.ones(512, 4)
        (first,) = admission.static_inputs("hc_attn", (x,))
        (again,) = admission.static_inputs("hc_attn", (x.clone(),))
        (other_stage,) = admission.static_inputs("hc_moe", (x,))
        (other_shape,) = admission.static_inputs("hc_attn", (torch.ones(2048, 4),))
        self.assertIs(first, again)
        self.assertIsNot(first, other_stage)
        self.assertIsNot(first, other_shape)

    def test_pending_injection_reuses_residual_slot_without_aliasing_inputs(self):
        admission = module.SegmentAdmission()
        x = torch.ones(512, 16)
        (first,) = admission.static_inputs("hc_attn", (x,))
        residual, output, gate = admission.static_inputs(
            "hc_attn", (x, torch.ones(512, 4), torch.ones(512, 4))
        )
        self.assertIs(first, residual)
        self.assertNotEqual(output.data_ptr(), gate.data_ptr())
        self.assertNotEqual(first.data_ptr(), x.data_ptr())


if __name__ == "__main__":
    unittest.main()
