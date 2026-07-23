"""Host-side (no-GPU) regression for the MoE warmup skew math.

The warmup skew logic decides how much reserved hot-expert load is folded into the
memory-traced warmup forward. It is pure Python arithmetic + CPU-tensor index shuffling, so
it can and should be verified without a GPU smoke run. All supported executors are slot-based,
so these tests pin:
  * the reserved-fraction formulas (default_slot_share / skew_reserve),
  * warmup_skew_topk_ids expert-id legality and rank-0 routing, including the
    n_hot==0 (no hot tokens) and all-hot boundaries.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs import (
    warmup_diagnostics as diagnostics_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.defs.warmup_diagnostics import (
    diagnostics,
)


class _FakeRouter:
    def __init__(
        self,
        ep_size,
        expert_num_per_rank,
        ep_rank=0,
        tp_size=1,
        dp_size=None,
        expert_num=None,
    ):
        dp_size = dp_size if dp_size is not None else ep_size
        self.config = SimpleNamespace(
            ep_size=ep_size,
            expert_num=(
                ep_size * expert_num_per_rank
                if expert_num is None
                else expert_num
            ),
            ep_rank=ep_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            world_size=tp_size * dp_size,
            parallelism_config=SimpleNamespace(
                ffn_disaggregate_config=SimpleNamespace(
                    enable_ffn_disaggregate=False
                )
            ),
        )


class _SlotExecutor:
    """Slot-based executor: memory scales with local expert slots."""


class SkewFractionMathTest(unittest.TestCase):
    def test_skew_reserve_default_and_clamp(self):
        # default MOE_SKEW_MULT=1.5, MOE_SKEW_ADD=0.1
        self.assertAlmostEqual(
            diagnostics.skew_reserve(0.2), 0.2 * 1.5 + 0.1, places=6
        )
        # clamped to 1.0
        self.assertEqual(diagnostics.skew_reserve(0.9), 1.0)

    def test_skew_reserve_env_override(self):
        prev_mult = os.environ.get("MOE_SKEW_MULT")
        prev_add = os.environ.get("MOE_SKEW_ADD")
        os.environ["MOE_SKEW_MULT"] = "2.0"
        os.environ["MOE_SKEW_ADD"] = "0.0"
        try:
            self.assertAlmostEqual(diagnostics.skew_reserve(0.3), 0.6, places=6)
        finally:
            for key, prev in (("MOE_SKEW_MULT", prev_mult), ("MOE_SKEW_ADD", prev_add)):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

    def test_skew_reserve_invalid_env_uses_defaults(self):
        with (
            patch.dict(
                os.environ,
                {"MOE_SKEW_MULT": "invalid", "MOE_SKEW_ADD": "invalid"},
            ),
            patch.object(diagnostics, "skew_env_warned", False),
        ):
            self.assertAlmostEqual(diagnostics.skew_reserve(0.2), 0.4, places=6)

    def test_default_slot_share(self):
        self.assertEqual(diagnostics.default_slot_share(1, 8, 2), 1.0)
        # experts <= top_k: every rank is guaranteed hit
        self.assertEqual(diagnostics.default_slot_share(2, 2, 2), 1.0)
        self.assertAlmostEqual(
            diagnostics.default_slot_share(4, 8, 2),
            min(1.0, 0.25 * 1.5 + 0.1),
            places=6,
        )


class WarmupSkewParamsTest(unittest.TestCase):
    def test_slot_executor_params(self):
        q, s = diagnostics.warmup_skew_params(ep_size=4, expert_num=8, top_k=2)
        self.assertEqual(s, 2)  # slot-based: one rank-0 slot per top_k
        self.assertGreaterEqual(q, 1.0 / 4)
        self.assertLessEqual(q, 1.0)


class WarmupSkewTopkIdsTest(unittest.TestCase):
    @staticmethod
    def _apply(topk_ids, ep_size, expert_num):
        return diagnostics.warmup_skew_topk_ids(
            topk_ids, ep_size, expert_num, "SlotExecutor"
        )

    def _assert_valid_ids(self, out, expert_num):
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out < expert_num))

    def test_single_ep_returns_unchanged(self):
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        out = self._apply(topk_ids, ep_size=1, expert_num=2)
        self.assertTrue(torch.equal(out, topk_ids))

    def test_invalid_expert_layout_is_rejected(self):
        router = _FakeRouter(ep_size=4, expert_num_per_rank=0)
        with self.assertRaisesRegex(ValueError, "does not match"):
            FusedMoe(router=router, fused_experts=_SlotExecutor(), expert_num=8)

    def test_non_divisible_expert_layout_is_rejected(self):
        router = _FakeRouter(
            ep_size=3,
            expert_num_per_rank=0,
            expert_num=8,
        )
        with self.assertRaisesRegex(ValueError, "divisible"):
            FusedMoe(router=router, fused_experts=_SlotExecutor(), expert_num=8)

    def test_slot_executor_hot_and_cold_routing(self):
        ep_size, n_local = 4, 2
        expert_num = ep_size * n_local
        topk_ids = torch.zeros((8, 2), dtype=torch.int64)

        out = self._apply(topk_ids, ep_size, expert_num)

        self.assertEqual(tuple(out.shape), (8, 2))
        self._assert_valid_ids(out, expert_num)
        # q≈0.475 over 8 tokens -> 4 hot rows routed entirely to rank 0 (all ids < n_local).
        hot = out[:4]
        self.assertTrue(torch.all(hot < n_local))
        # remaining rows are pushed off rank 0 (all ids >= n_local).
        cold = out[4:]
        self.assertTrue(torch.all(cold >= n_local))

    def test_n_hot_zero_boundary(self):
        # ep=4, top_k=1, single token -> round(1 * 0.475) == 0 hot rows, everything cold.
        ep_size, n_local = 4, 2
        topk_ids = torch.zeros((1, 1), dtype=torch.int64)

        out = self._apply(topk_ids, ep_size, ep_size * n_local)

        self._assert_valid_ids(out, ep_size * n_local)
        self.assertTrue(torch.all(out >= n_local))  # no hot row -> nothing on rank 0

    def test_all_hot_boundary(self):
        # experts == top_k -> slot share clamps to 1.0 -> every token is hot.
        ep_size, n_local = 2, 1
        topk_ids = torch.full((3, 2), 1, dtype=torch.int64)

        out = self._apply(topk_ids, ep_size, ep_size * n_local)

        self._assert_valid_ids(out, ep_size * n_local)
        # n_local == 1 -> the only rank-0 slot is expert id 0.
        self.assertTrue(torch.all(out == 0))

    def test_dtype_preserved(self):
        for dt in (torch.int32, torch.int64):
            out = self._apply(torch.zeros((6, 2), dtype=dt), 4, 8)
            self.assertEqual(out.dtype, dt)


class RuntimeSlotDistributionTest(unittest.TestCase):
    @patch("torch.cuda.is_current_stream_capturing", return_value=True)
    def test_capture_returns_before_tensor_or_collective_work(self, _capturing):
        topk_ids = MagicMock()

        diagnostics.log_runtime_slot_distribution(_FakeRouter(2, 2), topk_ids)

        topk_ids.reshape.assert_not_called()

    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    @patch(
        "rtp_llm.models_py.distributed.collective_torch.all_reduce",
        side_effect=lambda tensor, _group: tensor,
    )
    def test_updates_global_slot_share_peaks(self, all_reduce, _capturing):
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        with (
            patch.object(diagnostics, "runtime_slot_peaks", []),
            patch.object(diagnostics_module.logger, "warning") as warning,
        ):
            diagnostics.log_runtime_slot_distribution(_FakeRouter(2, 2), topk_ids)

            self.assertEqual(diagnostics.runtime_slot_peaks, [0.5, 0.5])
            self.assertIn("new_peak", warning.call_args.args[0])
        all_reduce.assert_called_once()
        self.assertEqual(all_reduce.call_args.args[1].name, "DP")

    @patch("torch.cuda.is_current_stream_capturing", return_value=False)
    def test_unsupported_topology_warns_before_tensor_work(self, _capturing):
        topk_ids = MagicMock()
        router = _FakeRouter(2, 2, tp_size=2, dp_size=1)
        with (
            patch.object(diagnostics, "runtime_slot_log_unsupported", False),
            patch.object(diagnostics_module.logger, "warning") as warning,
        ):
            diagnostics.log_runtime_slot_distribution(router, topk_ids)

        topk_ids.reshape.assert_not_called()
        self.assertIn("unsupported topology", warning.call_args.args[0])

    def test_enabled_slot_log_does_not_run_startup_collective(self):
        with (
            patch.object(diagnostics, "runtime_slot_log_requested", True),
            patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce",
            ) as all_reduce,
        ):
            self.assertTrue(diagnostics.resolve_runtime_slot_log(_FakeRouter(2, 2)))
        all_reduce.assert_not_called()

    def test_disabled_slot_log_does_not_run_startup_collective(self):
        with (
            patch.object(diagnostics, "runtime_slot_log_requested", False),
            patch.object(diagnostics, "warmup_enabled", False),
            patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce"
            ) as all_reduce,
        ):
            moe = FusedMoe(_FakeRouter(2, 2), _SlotExecutor(), expert_num=4)
            self.assertFalse(moe.runtime_slot_log_enabled)
        all_reduce.assert_not_called()


class TraceMemoryBindingTest(unittest.TestCase):
    def test_binding_is_importable_and_callable(self):
        from rtp_llm.ops.compute_ops import get_trace_memory_state, is_trace_memory

        self.assertTrue(callable(is_trace_memory))
        self.assertIsInstance(is_trace_memory(), bool)
        self.assertTrue(callable(get_trace_memory_state))
        self.assertIn(get_trace_memory_state(), (0, 1, 2))

    def test_final_warmup_config_resets_trace_latch(self):
        with (
            patch.object(diagnostics, "warmup_enabled", True),
            patch.object(diagnostics, "trace_memory_finished", False),
        ):
            diagnostics.configure_warmup_trace(False)
            self.assertFalse(diagnostics.warmup_enabled)
            self.assertTrue(diagnostics.trace_memory_finished)

            diagnostics.configure_warmup_trace(True)
            self.assertTrue(diagnostics.warmup_enabled)
            self.assertFalse(diagnostics.trace_memory_finished)

    def test_ep_warmup_requires_binding(self):
        router = _FakeRouter(ep_size=2, expert_num_per_rank=1)
        with (
            patch.object(diagnostics, "warmup_enabled", True),
            patch.object(diagnostics, "get_trace_memory_state", None),
        ):
            with self.assertRaisesRegex(RuntimeError, "get_trace_memory_state"):
                FusedMoe(
                    router=router,
                    fused_experts=_SlotExecutor(),
                    expert_num=2,
                )

    def test_disabled_warmup_skips_binding_requirement(self):
        router = _FakeRouter(ep_size=2, expert_num_per_rank=1)
        with (
            patch.object(diagnostics, "warmup_enabled", False),
            patch.object(diagnostics, "get_trace_memory_state", None),
        ):
            FusedMoe(
                router=router,
                fused_experts=_SlotExecutor(),
                expert_num=2,
            )

    def test_completed_startup_trace_stops_binding_queries(self):
        binding = MagicMock(side_effect=[0, 1, 1, 2])
        with (
            patch.object(diagnostics, "warmup_enabled", True),
            patch.object(diagnostics, "get_trace_memory_state", binding),
            patch.object(diagnostics, "trace_memory_finished", False),
        ):
            self.assertFalse(diagnostics.in_memory_trace(2))
            self.assertTrue(diagnostics.in_memory_trace(2))
            self.assertTrue(diagnostics.in_memory_trace(2))
            self.assertFalse(diagnostics.in_memory_trace(2))
            self.assertFalse(diagnostics.in_memory_trace(2))

        self.assertEqual(binding.call_count, 4)

    def test_finished_trace_does_not_query_binding(self):
        binding = MagicMock()
        with (
            patch.object(diagnostics, "get_trace_memory_state", binding),
            patch.object(diagnostics, "trace_memory_finished", True),
        ):
            self.assertFalse(diagnostics.in_memory_trace(2))
        binding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
