"""Device-free regression for the MoE warmup skew math.

The warmup skew logic decides how much reserved hot-expert load is folded into the
memory-traced warmup forward. It is pure Python arithmetic + CPU-tensor index shuffling;
no assertion touches a GPU device. The target still schedules on a GPU host
(exec_properties in BUILD): the import chain routes through compute_ops' arch
dispatch, which requires torch.cuda.is_available() in the CUDA build.
All supported executors are slot-based, so these tests pin:
  * the reserved-fraction formula (skew_fraction),
  * warmup_skew_topk_ids expert-id legality and rank-0 routing, including the
    n_hot==0 (no hot tokens) and all-hot boundaries, plus the fact that cold rows
    are spread over the non-rank-0 experts and only overflow back onto rank 0
    when row-wise uniqueness leaves them no other ids,
  * FusedMoeDataRouter.experts_per_ep_rank, the validated partition used by every
    router that slices experts by ep_size.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs import (
    warmup_diagnostics as diagnostics_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoe,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.warmup_diagnostics import (
    MoeWarmupDiagnostics,
    diagnostics,
)
from rtp_llm.ops import RoleType


class _FakeRouter:
    def __init__(
        self,
        ep_size,
        expert_num_per_rank,
        ep_rank=0,
        tp_size=1,
        dp_size=None,
        expert_num=None,
        enable_cuda_graph=False,
        role_type=RoleType.PREFILL,
    ):
        dp_size = dp_size if dp_size is not None else ep_size
        self.config = SimpleNamespace(
            ep_size=ep_size,
            expert_num=(
                ep_size * expert_num_per_rank if expert_num is None else expert_num
            ),
            ep_rank=ep_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            world_size=tp_size * dp_size,
            enable_cuda_graph=enable_cuda_graph,
            enable_moe_warmup_skew=(role_type == RoleType.PREFILL),
            parallelism_config=SimpleNamespace(
                ffn_disaggregate_config=SimpleNamespace(enable_ffn_disaggregate=False)
            ),
        )

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        return ExpertForwardPayload(
            expert_x=a1,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(
        self,
        payload,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        extra_finalize_args,
    ):
        return payload.fused_expert_output


class _EpPartitionRouter(FusedMoeDataRouter):
    """Minimal concrete router standing in for the real ep_size-slicing routers."""

    def __init__(self, expert_num, ep_size, phy_exp_num=None):
        super().__init__(
            SimpleNamespace(
                expert_num=expert_num,
                ep_size=ep_size,
                phy_exp_num=expert_num if phy_exp_num is None else phy_exp_num,
            ),
            quant_config=None,
        )

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        raise NotImplementedError

    def finalize(
        self,
        payload,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
        extra_finalize_args,
    ):
        raise NotImplementedError


class _SlotExecutor:
    """Slot-based executor: memory scales with local expert slots."""

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        return CombineForwardPayload(fused_expert_output=payload.expert_x)


def _natural_ids(num_tokens, top_k, expert_num, dtype=torch.int64):
    """Row-unique ids standing in for the model's own routing decision."""
    rows = torch.arange(num_tokens).unsqueeze(1)
    slots = torch.arange(top_k).unsqueeze(0)
    return ((rows + slots) % expert_num).to(dtype)


class DiagnosticsTestCase(unittest.TestCase):
    def setUp(self):
        # This file runs on the default (no-GPU) gate, where the compute_ops import
        # can fail and leave diagnostics.get_trace_memory_state as None. PREFILL EP
        # MoE construction would then trip require_trace_binding() with a RuntimeError
        # unrelated to most cases, so fake the binding by default. MagicMock()'s
        # __int__ is 1, matching the Active phase used by explicit gate tests.
        binding_patcher = patch.object(
            diagnostics, "get_trace_memory_state", MagicMock()
        )
        binding_patcher.start()
        self.addCleanup(binding_patcher.stop)

        diagnostics.reload_runtime_settings()


class EpRankPartitionTest(unittest.TestCase):
    """experts_per_ep_rank owns the divisibility contract for EP routers.

    It deliberately lives on the router rather than on FusedMoe: routers that
    partition along another dimension (BatchedDataRouter slices by tp_size) must
    not inherit an ep_size constraint they never honor.
    """

    def test_divisible_layout_returns_even_partition(self):
        self.assertEqual(
            _EpPartitionRouter(expert_num=64, ep_size=8).experts_per_ep_rank(), 8
        )
        self.assertEqual(
            _EpPartitionRouter(expert_num=7, ep_size=1).experts_per_ep_rank(), 7
        )

    def test_non_divisible_layout_is_rejected(self):
        router = _EpPartitionRouter(expert_num=8, ep_size=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            router.experts_per_ep_rank()

    def test_non_divisible_redundant_layout_is_a_known_gap(self):
        # Known gap, not endorsed behaviour: redundant layouts are exempted from
        # the divisibility check solely to keep their pre-existing partitioning
        # bit-for-bit. Nothing in this path consumes phy2log/phy_exp_num, so the
        # floor division below still drops tail experts (60 % 8 != 0 -> 7 per
        # rank, 4 experts unreachable). If redundant partitioning ever becomes
        # phy2log-aware, this test should start failing and be rewritten.
        router = _EpPartitionRouter(expert_num=60, ep_size=8, phy_exp_num=64)
        self.assertEqual(router.experts_per_ep_rank(), 7)

    def test_non_positive_layout_is_rejected(self):
        for expert_num, ep_size in ((0, 4), (8, 0)):
            with self.subTest(expert_num=expert_num, ep_size=ep_size):
                router = _EpPartitionRouter(expert_num, ep_size)
                with self.assertRaisesRegex(ValueError, "positive"):
                    router.experts_per_ep_rank()


class SkewFractionMathTest(DiagnosticsTestCase):
    def test_skew_fraction_default(self):
        # default MOE_SKEW_MULT=2.0: the hot rank carries exactly twice the
        # 1/ep_size mean share.
        self.assertAlmostEqual(
            diagnostics.skew_fraction(ep_size=5, expert_num=8, top_k=2),
            0.4,
            places=6,
        )
        # no EP / every rank hit anyway -> whole batch is hot
        self.assertEqual(
            diagnostics.skew_fraction(ep_size=1, expert_num=8, top_k=2), 1.0
        )
        self.assertEqual(
            diagnostics.skew_fraction(ep_size=2, expert_num=2, top_k=2), 1.0
        )

    def test_skew_fraction_clamped_to_one(self):
        diagnostics.reload_runtime_settings(skew_mult=3.0)
        self.assertEqual(
            diagnostics.skew_fraction(ep_size=2, expert_num=8, top_k=2), 1.0
        )

    def test_skew_fraction_structured_config_override(self):
        diagnostics.reload_runtime_settings(skew_mult=1.6)
        self.assertAlmostEqual(
            diagnostics.skew_fraction(ep_size=10, expert_num=8, top_k=2),
            0.16,
            places=6,
        )

    def test_aggregate_multiple_is_exactly_mult_at_every_ep_size(self):
        """Hot-rank load relative to the mean is exactly skew_mult (below clamp).

        The additive-with-floor formula this replaced grew with ep_size; the
        pure-mult formula pins the multiple everywhere the 1.0 clamp is loose.
        """
        mult = 2.0
        diagnostics.reload_runtime_settings(skew_mult=mult)
        for ep_size in (3, 4, 8, 16, 32, 64):
            with self.subTest(ep_size=ep_size):
                fraction = diagnostics.skew_fraction(
                    ep_size=ep_size, expert_num=256, top_k=8
                )
                self.assertAlmostEqual(fraction * ep_size, mult, places=6)

    def test_invalid_structured_config_is_rejected(self):
        # <= 1.0 degenerates to uniform routing, so it is rejected outright.
        for skew_mult in (-1.0, float("nan"), float("inf"), 0.5, 1.0):
            with self.subTest(skew_mult=skew_mult):
                with self.assertRaises(ValueError):
                    diagnostics.reload_runtime_settings(skew_mult)


class WarmupSkewTopkIdsTest(DiagnosticsTestCase):
    @staticmethod
    def _apply(topk_ids, ep_size, expert_num):
        return diagnostics.warmup_skew_topk_ids(
            topk_ids, ep_size, expert_num, "SlotExecutor"
        )

    def _assert_valid_ids(self, out, expert_num):
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out < expert_num))

    def _assert_unique_rows(self, out):
        for row in out:
            self.assertEqual(torch.unique(row).numel(), row.numel())

    def test_single_ep_returns_unchanged(self):
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        out = self._apply(topk_ids, ep_size=1, expert_num=2)
        self.assertTrue(torch.equal(out, topk_ids))

    def test_empty_batch_returns_unchanged_and_keeps_the_summary(self):
        # A DP rank can receive zero tokens. The rewrite must not touch the batch
        # and must not spend the once-per-lifecycle summary on hot_tokens=0.
        topk_ids = torch.empty((0, 2), dtype=torch.int64)

        with patch.object(diagnostics_module.logger, "info") as info:
            out = self._apply(topk_ids, ep_size=4, expert_num=8)

        self.assertIs(out, topk_ids)
        info.assert_not_called()

    def test_hot_rows_go_to_rank0_and_cold_rows_avoid_rank0(self):
        ep_size, n_local = 4, 2
        expert_num = ep_size * n_local
        topk_ids = _natural_ids(8, 2, expert_num)

        out = self._apply(topk_ids, ep_size, expert_num)

        self.assertEqual(tuple(out.shape), (8, 2))
        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        # q = 2.0/4 = 0.5 over 8 tokens -> 4 hot rows routed entirely to rank 0.
        self.assertTrue(torch.all(out[:4] < n_local))
        # Cold rows are rewritten off rank 0, so its token share is exactly q.
        self.assertTrue(torch.all(out[4:] >= n_local))

    def test_non_divisible_logical_layout_rewrite_stays_valid(self):
        expert_num, ep_size = 60, 8
        topk_ids = _natural_ids(8, 8, expert_num)

        out = self._apply(topk_ids, ep_size, expert_num)

        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        # ceil(60 / 8) reserves logical ids [0, 8) as the hot partition;
        # q = 2.0/8 = 0.25 over 8 tokens -> 2 hot rows.
        self.assertTrue(torch.all(out[:2] < 8))
        self.assertTrue(torch.all(out[2:] >= 8))
        # Documented upper-bound semantics (warmup_skew_topk_ids docstring): the
        # ceil hot window is wider than the floor partition EP routers derive
        # (60 // 8 == 7), so hot rows legitimately carry id 7 -- which a floor-
        # partitioned router dispatches off rank 0. That makes the logged
        # rank0_slot_share an upper bound for redundant non-divisible layouts,
        # not a bug in the rewrite; this assertion pins the window mismatch the
        # caveat describes.
        floor_window = expert_num // ep_size
        self.assertGreater(8, floor_window)
        self.assertTrue(torch.any(out[:2] >= floor_window))

    def test_nonempty_batch_always_has_a_hot_token(self):
        # ep=4, top_k=1, single token would round down without the lower bound.
        topk_ids = torch.tensor([[7]], dtype=torch.int64)

        out = self._apply(topk_ids, ep_size=4, expert_num=8)

        self.assertEqual(out.item(), 0)

    def test_skew_summary_logs_once_at_info(self):
        topk_ids = _natural_ids(2, 1, 8)
        with (
            patch.object(diagnostics_module.logger, "info") as info,
            patch.object(diagnostics_module.logger, "warning") as warning,
        ):
            self._apply(topk_ids, ep_size=4, expert_num=8)
            self._apply(topk_ids, ep_size=4, expert_num=8)

        info.assert_called_once()
        warning.assert_not_called()

    def test_all_hot_boundary(self):
        # experts == top_k -> slot share clamps to 1.0 -> every token is hot.
        ep_size, expert_num = 2, 2
        topk_ids = torch.full((3, 2), 1, dtype=torch.int64)

        out = self._apply(topk_ids, ep_size, expert_num)

        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        self.assertTrue(torch.all(out == torch.tensor([0, 1])))

    def test_top_k_larger_than_local_experts_stays_unique_and_countable(self):
        ep_size, n_local, top_k = 4, 2, 5
        expert_num = ep_size * n_local
        topk_ids = _natural_ids(6, top_k, expert_num)

        out = self._apply(topk_ids, ep_size, expert_num)

        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        # q = 2.0/4 = 0.5, dilution compensation scales the hot row count by
        # top_k/n_local = 2.5 and caps at the batch -> all 6 rows are hot: two
        # slots on rank 0, the remaining three overflow onto other ranks.
        self.assertTrue(torch.all(out[:, :n_local] < n_local))
        self.assertTrue(torch.all(out[:, n_local:] >= n_local))
        # Slot executors flatten and count these ids before dispatch.
        slots = torch.bincount(out.reshape(-1), minlength=expert_num)
        self.assertEqual(slots.numel(), expert_num)
        self.assertEqual(int(slots.sum()), out.numel())

    def test_dilution_compensation_restores_rank0_slot_share(self):
        # ep=8, experts=8 -> n_local=1, top_k=2: each hot row lands only half its
        # slots on rank 0, so the hot row count doubles: q=0.25 -> 4 of 8 rows.
        ep_size, expert_num, top_k = 8, 8, 2
        topk_ids = _natural_ids(8, top_k, expert_num)

        out = self._apply(topk_ids, ep_size, expert_num)

        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        self.assertTrue(torch.all(out[:4, 0] == 0))
        self.assertTrue(torch.all(out[4:] >= 1))
        # The dispatched-slot share on rank 0 is exactly mult/ep = 2/8.
        rank0_slots = int((out == 0).sum())
        self.assertEqual(rank0_slots / out.numel(), 0.25)

    def test_cold_rows_overflow_back_onto_rank0_when_they_cannot_avoid_it(self):
        # ep=3, experts=6 -> n_local=2, 4 non-rank0 experts. top_k=5 needs five
        # unique ids, so every cold row must place exactly one slot on rank 0.
        # mult=1.1 keeps the compensated hot fraction below 1.0 (0.9166) so cold
        # rows still exist: 12 tokens -> 11 hot, 1 cold.
        diagnostics.reload_runtime_settings(skew_mult=1.1)
        ep_size, expert_num, top_k = 3, 6, 5
        n_local = 2
        topk_ids = _natural_ids(12, top_k, expert_num)

        out = self._apply(topk_ids, ep_size, expert_num)

        self._assert_valid_ids(out, expert_num)
        self._assert_unique_rows(out)
        self.assertTrue(torch.all(out[:11, :n_local] < n_local))
        cold_rank0_slots = (out[11:] < n_local).sum(dim=1)
        self.assertTrue(torch.all(cold_rank0_slots == top_k - (expert_num - n_local)))

    def test_top_k_larger_than_total_experts_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot contain unique ids"):
            self._apply(torch.zeros((2, 5), dtype=torch.int64), 2, 4)

    def test_dtype_preserved(self):
        for dt in (torch.int32, torch.int64):
            out = self._apply(_natural_ids(6, 2, 8, dtype=dt), 4, 8)
            self.assertEqual(out.dtype, dt)

    def test_capture_skips_rewrite_and_warns_once(self):
        topk_ids = MagicMock()
        topk_ids.is_cuda = True
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(diagnostics, "warmup_capture_warned", False),
            patch.object(diagnostics_module.logger, "warning") as warning,
        ):
            self.assertIs(self._apply(topk_ids, 2, 4), topk_ids)
            self.assertIs(self._apply(topk_ids, 2, 4), topk_ids)

        warning.assert_called_once()
        self.assertIn("CUDA graph capture", warning.call_args.args[0])


class PrefillSkewGateTest(DiagnosticsTestCase):
    def test_prefill_warmup_forward_uses_the_rewritten_ids(self):
        router = _FakeRouter(2, 2)
        router.prepare = MagicMock(wraps=router.prepare)
        moe = FusedMoe(router, _SlotExecutor(), expert_num=4)
        hidden_states = torch.ones((2, 4))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.ones((2, 1), dtype=torch.int64)
        rewritten_ids = torch.zeros_like(topk_ids)

        with (
            patch.object(diagnostics, "is_moe_warmup_active", return_value=True),
            patch.object(
                diagnostics,
                "warmup_skew_topk_ids",
                return_value=rewritten_ids,
            ) as rewrite,
        ):
            output = moe(hidden_states, topk_weights, topk_ids)

        rewrite.assert_called_once_with(topk_ids, 2, 4, "_SlotExecutor")
        self.assertIs(router.prepare.call_args.args[4], rewritten_ids)
        self.assertTrue(torch.equal(output, hidden_states))


class WarmupRoleGateTest(DiagnosticsTestCase):
    def test_decode_warmup_keeps_natural_routing(self):
        router = _FakeRouter(
            2,
            2,
            enable_cuda_graph=True,
            role_type=RoleType.DECODE,
        )
        router.prepare = MagicMock(wraps=router.prepare)
        moe = FusedMoe(router, _SlotExecutor(), expert_num=4)
        hidden_states = torch.ones((2, 4))
        topk_weights = torch.ones((2, 1))
        topk_ids = torch.ones((2, 1), dtype=torch.int64)

        with (
            patch.object(
                diagnostics, "is_moe_warmup_active", return_value=True
            ) as warmup_active,
            patch.object(diagnostics, "warmup_skew_topk_ids") as rewrite,
        ):
            output = moe(hidden_states, topk_weights, topk_ids)

        warmup_active.assert_not_called()
        rewrite.assert_not_called()
        self.assertIs(router.prepare.call_args.args[4], topk_ids)
        self.assertTrue(torch.equal(output, hidden_states))


class TraceMemoryBindingTest(DiagnosticsTestCase):
    def test_structured_config_is_loaded_only_when_explicitly_reloaded(self):
        local_diagnostics = MoeWarmupDiagnostics()
        self.assertEqual(local_diagnostics.skew_mult, 2.0)

        local_diagnostics.reload_runtime_settings(skew_mult=2.5)
        self.assertEqual(local_diagnostics.skew_mult, 2.5)

    def test_reload_resets_model_build_trace_state(self):
        with (
            patch.object(diagnostics, "trace_memory_finished", True),
            patch.object(diagnostics, "warmup_skew_logged", True),
        ):
            diagnostics_module.reload_runtime_diagnostics()

            self.assertFalse(diagnostics.trace_memory_finished)
            self.assertFalse(diagnostics.warmup_skew_logged)

    def test_prefill_ep_warmup_requires_binding(self):
        router = _FakeRouter(ep_size=2, expert_num_per_rank=1)
        with patch.object(diagnostics, "get_trace_memory_state", None):
            with self.assertRaisesRegex(RuntimeError, "get_trace_memory_state"):
                FusedMoe(
                    router=router,
                    fused_experts=_SlotExecutor(),
                    expert_num=2,
                )

    def test_decode_does_not_require_trace_binding(self):
        router = _FakeRouter(
            ep_size=2,
            expert_num_per_rank=1,
            role_type=RoleType.DECODE,
        )
        with patch.object(diagnostics, "get_trace_memory_state", None):
            FusedMoe(
                router=router,
                fused_experts=_SlotExecutor(),
                expert_num=2,
            )

    def test_completed_startup_trace_stops_binding_queries(self):
        """Pending is re-queried, Active stays active, Finished latches for good.

        active_forwards is parameterised because a CUDA-graph capture issues many
        Active forwards before the native trace finishes; the latch behaviour is
        the same assertion at any count, so it is a subTest rather than a second
        test method.
        """
        for active_forwards in (2, 32):
            with self.subTest(active_forwards=active_forwards):
                binding = MagicMock(
                    side_effect=[0] + [1] * active_forwards + [2]
                )
                with (
                    patch.object(diagnostics, "get_trace_memory_state", binding),
                    patch.object(diagnostics, "trace_memory_finished", False),
                ):
                    self.assertFalse(diagnostics.is_moe_warmup_active(2))
                    for _ in range(active_forwards):
                        self.assertTrue(diagnostics.is_moe_warmup_active(2))
                    # Sees Finished, latches, and never queries the binding again.
                    self.assertFalse(diagnostics.is_moe_warmup_active(2))
                    self.assertFalse(diagnostics.is_moe_warmup_active(2))
                    self.assertTrue(diagnostics.trace_memory_finished)

                self.assertEqual(binding.call_count, active_forwards + 2)

    def test_finished_trace_does_not_query_binding(self):
        binding = MagicMock()
        with (
            patch.object(diagnostics, "get_trace_memory_state", binding),
            patch.object(diagnostics, "trace_memory_finished", True),
        ):
            self.assertFalse(diagnostics.is_moe_warmup_active(2))
        binding.assert_not_called()

    def test_single_ep_never_queries_native_trace_state(self):
        binding = MagicMock(return_value=1)
        with patch.object(diagnostics, "get_trace_memory_state", binding):
            self.assertFalse(diagnostics.is_moe_warmup_active(1))
        binding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
