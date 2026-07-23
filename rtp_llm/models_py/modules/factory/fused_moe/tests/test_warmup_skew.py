"""Host-side (no-GPU) regression for the MoE warmup skew math.

The warmup skew logic decides how much reserved hot-expert load is folded into the
memory-traced warmup forward. It is pure Python arithmetic + CPU-tensor index shuffling, so
it can and should be verified without a GPU smoke run. All supported executors are slot-based,
so these tests pin:
  * the reserved-fraction formulas (_default_slot_share / _skew_reserve),
  * _warmup_skew_topk_ids expert-id legality and rank-0 routing, including the
    n_hot==0 (no hot tokens) and all-hot boundaries.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoe,
    _default_slot_share,
    _log_runtime_slot_distribution,
    _skew_reserve,
)


class _FakeRouter:
    def __init__(
        self, ep_size, expert_num_per_rank, ep_rank=0, tp_size=1, dp_size=None
    ):
        self.ep_size = ep_size
        self.expert_num_per_rank = expert_num_per_rank
        self.ep_rank = ep_rank
        self.tp_size = tp_size
        self.dp_size = dp_size if dp_size is not None else ep_size


class _SlotExecutor:
    """Slot-based executor: memory scales with local expert slots."""


def _make_moe(ep_size, n_local, executor):
    router = _FakeRouter(ep_size=ep_size, expert_num_per_rank=n_local)
    return FusedMoe(router=router, fused_experts=executor, expert_num=ep_size * n_local)


class SkewFractionMathTest(unittest.TestCase):
    def test_skew_reserve_default_and_clamp(self):
        # default MOE_SKEW_MULT=1.5, MOE_SKEW_ADD=0.1
        self.assertAlmostEqual(_skew_reserve(0.2), 0.2 * 1.5 + 0.1, places=6)
        # clamped to 1.0
        self.assertEqual(_skew_reserve(0.9), 1.0)

    def test_skew_reserve_env_override(self):
        prev_mult = os.environ.get("MOE_SKEW_MULT")
        prev_add = os.environ.get("MOE_SKEW_ADD")
        os.environ["MOE_SKEW_MULT"] = "2.0"
        os.environ["MOE_SKEW_ADD"] = "0.0"
        try:
            self.assertAlmostEqual(_skew_reserve(0.3), 0.6, places=6)
        finally:
            for key, prev in (("MOE_SKEW_MULT", prev_mult), ("MOE_SKEW_ADD", prev_add)):
                if prev is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = prev

    def test_default_slot_share(self):
        self.assertEqual(_default_slot_share(1, 8, 2), 1.0)
        # experts <= top_k: every rank is guaranteed hit
        self.assertEqual(_default_slot_share(2, 2, 2), 1.0)
        self.assertAlmostEqual(
            _default_slot_share(4, 8, 2), min(1.0, 0.25 * 1.5 + 0.1), places=6
        )


class WarmupSkewParamsTest(unittest.TestCase):
    def test_slot_executor_params(self):
        moe = _make_moe(4, 2, _SlotExecutor())
        q, s = moe._warmup_skew_params(ep_size=4, expert_num=8, topk=2)
        self.assertEqual(s, 2)  # slot-based: one rank-0 slot per top_k
        self.assertGreaterEqual(q, 1.0 / 4)
        self.assertLessEqual(q, 1.0)


class WarmupSkewTopkIdsTest(unittest.TestCase):
    def _assert_valid_ids(self, out, expert_num):
        self.assertTrue(torch.all(out >= 0))
        self.assertTrue(torch.all(out < expert_num))

    def test_single_ep_returns_unchanged(self):
        moe = _make_moe(1, 2, _SlotExecutor())
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
        out = moe._warmup_skew_topk_ids(topk_ids)
        self.assertTrue(torch.equal(out, topk_ids))

    def test_missing_local_experts_returns_unchanged(self):
        router = _FakeRouter(ep_size=4, expert_num_per_rank=0)
        moe = FusedMoe(router=router, fused_experts=_SlotExecutor(), expert_num=8)
        topk_ids = torch.zeros((3, 2), dtype=torch.int64)
        out = moe._warmup_skew_topk_ids(topk_ids)
        self.assertTrue(torch.equal(out, topk_ids))

    def test_slot_executor_hot_and_cold_routing(self):
        ep_size, n_local = 4, 2
        expert_num = ep_size * n_local
        moe = _make_moe(ep_size, n_local, _SlotExecutor())
        topk_ids = torch.zeros((8, 2), dtype=torch.int64)

        out = moe._warmup_skew_topk_ids(topk_ids)

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
        moe = _make_moe(ep_size, n_local, _SlotExecutor())
        topk_ids = torch.zeros((1, 1), dtype=torch.int64)

        out = moe._warmup_skew_topk_ids(topk_ids)

        self._assert_valid_ids(out, ep_size * n_local)
        self.assertTrue(torch.all(out >= n_local))  # no hot row -> nothing on rank 0

    def test_all_hot_boundary(self):
        # experts == top_k -> slot share clamps to 1.0 -> every token is hot.
        ep_size, n_local = 2, 1
        moe = _make_moe(ep_size, n_local, _SlotExecutor())
        topk_ids = torch.full((3, 2), 1, dtype=torch.int64)

        out = moe._warmup_skew_topk_ids(topk_ids)

        self._assert_valid_ids(out, ep_size * n_local)
        # n_local == 1 -> the only rank-0 slot is expert id 0.
        self.assertTrue(torch.all(out == 0))

    def test_dtype_preserved(self):
        moe = _make_moe(4, 2, _SlotExecutor())
        for dt in (torch.int32, torch.int64):
            out = moe._warmup_skew_topk_ids(torch.zeros((6, 2), dtype=dt))
            self.assertEqual(out.dtype, dt)


class RuntimeSlotDistributionTest(unittest.TestCase):
    @patch(
        "rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe._MOE_RUNTIME_SLOT_LOG",
        True,
    )
    @patch("torch.cuda.is_current_stream_capturing", return_value=True)
    def test_capture_returns_before_tensor_or_collective_work(self, _capturing):
        topk_ids = MagicMock()

        _log_runtime_slot_distribution(_FakeRouter(2, 2), topk_ids)

        topk_ids.reshape.assert_not_called()


if __name__ == "__main__":
    unittest.main()
