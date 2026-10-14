"""CPU-only unit tests for the independent EPLB window-ratio oracle.

These tests pin the oracle to the contract's acceptance arithmetic so the
oracle itself can be trusted when it is used to validate production GPU
recorders and window artifacts.  No torch, no GPU.

Run directly:
    /opt/conda310/bin/python -I \\
        rtp_llm/test/eplb_monitoring/window_ratio_oracle_test.py
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

# Allow running as a loose script (python -I <path>) in addition to Bazel's
# package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_ratio_oracle import (  # noqa: E402
    StepRecorder,
    WindowReport,
    WindowTracker,
    sum_histograms,
    sum_step_histories,
)


def _tracker(window_size, num_layers=1, num_experts=4, e2r=(0, 0, 1, 1), ranks=2):
    return WindowTracker(
        num_layers=num_layers,
        num_physical_experts=num_experts,
        expert_to_rank=list(e2r),
        num_ranks=ranks,
        window_size=window_size,
    )


def _feed_rank_totals(tracker, steps_rank_totals, layer=0):
    """Feed one step per [rank0_total, rank1_total] via concrete expert ids."""
    for totals in steps_rank_totals:
        rec = tracker.begin_step()
        # rank0 owns experts {0,1}, rank1 owns {2,3}; use 0 and 2.
        rec.record(layer, [0] * totals[0])
        rec.record(layer, [2] * totals[1])
        tracker.finish_step()


class WindowRatioOracleTest(unittest.TestCase):
    def test_contract_case1_w2_four_steps(self):
        """W=2, steps [4,0],[0,4],[2,2],[0,4] -> (2,1) and (1.5,1.5)."""
        tracker = _tracker(window_size=2)
        _feed_rank_totals(tracker, [[4, 0], [0, 4], [2, 2], [0, 4]])

        w1 = tracker.close_window()
        self.assertIsNotNone(w1)
        r1 = w1.layer_ratios(0)
        self.assertTrue(r1["valid"])
        self.assertEqual(r1["moe_lb"], Fraction(2))
        self.assertEqual(r1["moe_lb_window"], Fraction(1))

        w2 = tracker.close_window()
        self.assertIsNotNone(w2)
        r2 = w2.layer_ratios(0)
        self.assertTrue(r2["valid"])
        self.assertEqual(r2["moe_lb"], Fraction(3, 2))
        self.assertEqual(r2["moe_lb_window"], Fraction(3, 2))

        # No residual: both windows fully consumed.
        self.assertEqual(tracker.pending_steps(), 0)
        self.assertIsNone(tracker.close_window())

    def test_contract_case2_cross_source_sum_before_ratio(self):
        """source0 [[2,3],[2,3]] + source1 [[0,1]] -> hist [1,1,2,2]."""
        src0 = StepRecorder(num_layers=1, num_physical_experts=4)
        src1 = StepRecorder(num_layers=1, num_physical_experts=4)
        src0.record(0, [2, 3, 2, 3])
        src1.record(0, [0, 1])
        hist = sum_histograms([src0.snapshot(), src1.snapshot()])
        self.assertEqual(hist[0], [1, 1, 2, 2])
        self.assertEqual(sum(hist[0]), 6)

        t0 = _tracker(window_size=1)
        t1 = _tracker(window_size=1)
        t0.begin_step().record(0, [2, 3, 2, 3])
        t0.finish_step()
        t1.begin_step().record(0, [0, 1])
        t1.finish_step()
        rep = WindowReport.from_sources([t0.close_window(), t1.close_window()])
        lr = rep.layer_ratios(0)
        self.assertEqual(lr["rank_totals"], [2, 4])
        self.assertEqual(lr["sample_count"], 6)
        # Ratios computed AFTER summing sources: max/mean over [2,4].
        self.assertEqual(lr["moe_lb"], Fraction(4, 3))
        self.assertEqual(lr["moe_lb_window"], Fraction(4, 3))

    def test_moe_lb_vs_moe_lb_window_differ(self):
        """The two ratio definitions must not be conflated.

        Steps [4,0],[0,4]: per-step maxes sum to 8 over sum of means 4 -> 2,
        while window totals [4,4] give 1.  A reducer that swaps the two
        definitions fails this test.
        """
        tracker = _tracker(window_size=2)
        _feed_rank_totals(tracker, [[4, 0], [0, 4]])
        r = tracker.close_window().layer_ratios(0)
        self.assertEqual(r["moe_lb"], Fraction(2))
        self.assertEqual(r["moe_lb_window"], Fraction(1))

    def test_all_zero_window_is_invalid_not_one(self):
        tracker = _tracker(window_size=2, num_layers=2)
        for _ in range(2):
            tracker.begin_step()
            tracker.finish_step()
        report = tracker.close_window()
        for layer in range(2):
            r = report.layer_ratios(layer)
            self.assertFalse(r["valid"])
            self.assertIsNone(r["moe_lb"])
            self.assertIsNone(r["moe_lb_window"])
            self.assertEqual(r["sample_count"], 0)

    def test_partially_zero_layer_invalid_independently(self):
        """Layer 0 has traffic, layer 1 has none: only layer 1 invalid."""
        tracker = _tracker(window_size=1, num_layers=2)
        rec = tracker.begin_step()
        rec.record(0, [0, 2])
        tracker.finish_step()
        report = tracker.close_window()
        self.assertTrue(report.layer_ratios(0)["valid"])
        self.assertFalse(report.layer_ratios(1)["valid"])

    def test_layers_reported_independently(self):
        tracker = _tracker(window_size=1, num_layers=2)
        rec = tracker.begin_step()
        rec.record(0, [0, 0, 0, 2])  # layer0 rank totals [3,1]
        rec.record(1, [2, 2, 3, 3])  # layer1 rank totals [0,4]
        tracker.finish_step()
        report = tracker.close_window()
        self.assertEqual(report.layer_ratios(0)["rank_totals"], [3, 1])
        self.assertEqual(report.layer_ratios(1)["rank_totals"], [0, 4])
        self.assertEqual(report.layer_ratios(0)["moe_lb"], Fraction(3, 2))
        self.assertEqual(report.layer_ratios(1)["moe_lb"], Fraction(2))

    def test_layer_chunks_accumulate_within_step(self):
        tracker = _tracker(window_size=1)
        rec = tracker.begin_step()
        rec.record(0, [0])
        rec.record(0, [0, 1])
        rec.record(0, [3])
        tracker.finish_step()
        r = tracker.close_window().layer_ratios(0)
        self.assertEqual(r["rank_totals"], [3, 1])
        self.assertEqual(r["sample_count"], 4)

    def test_padding_tokens_count(self):
        """Padding rows that are actually routed must be counted.

        Simulates graph bucket 4 with topk 2: 8 activations even when the
        logical batch is 1 (3 padding rows), all routed to expert 0.
        """
        tracker = _tracker(window_size=1)
        rec = tracker.begin_step()
        rec.record(0, [0] * 8)  # bucket 4 x topk 2, padding included
        tracker.finish_step()
        r = tracker.close_window().layer_ratios(0)
        self.assertEqual(r["rank_totals"], [8, 0])
        self.assertEqual(r["sample_count"], 8)

    def test_window_not_due_returns_none(self):
        tracker = _tracker(window_size=2)
        _feed_rank_totals(tracker, [[1, 1]])
        self.assertIsNone(tracker.close_window())
        self.assertEqual(tracker.pending_steps(), 1)

    def test_sum_step_histories_alignment_guard(self):
        t0 = _tracker(window_size=1)
        t1 = _tracker(window_size=1)
        t0.begin_step().record(0, [0])
        t0.finish_step()
        t1.begin_step().record(0, [1])
        t1.finish_step()
        t1.begin_step().record(0, [1])  # t1 has an extra step
        t1.finish_step()
        with self.assertRaises(AssertionError):
            sum_step_histories([t0.history, t1.history])

    def test_varying_load_window_ratio_is_sample_weighted(self):
        """Steps [8,2],[1,1] -> moe_lb = (8+1)/(5+1) = 3/2.

        An unweighted mean of per-step ratios would give (8/5 + 1)/2 = 1.3;
        the contract ratio weights by sample count.  This test fails if a
        reducer averages step ratios instead of summing max/mean.
        """
        tracker = _tracker(window_size=2)
        _feed_rank_totals(tracker, [[8, 2], [1, 1]])
        r = tracker.close_window().layer_ratios(0)
        self.assertTrue(r["valid"])
        self.assertEqual(r["moe_lb"], Fraction(3, 2))
        self.assertNotEqual(r["moe_lb"], Fraction(13, 10))  # unweighted trap
        # Window-sum ratio: totals [9,3] -> 9/6 = 3/2 as well here.
        self.assertEqual(r["moe_lb_window"], Fraction(3, 2))

    def test_three_ranks_mean_uses_num_ranks(self):
        """mean over ranks divides by num_ranks even when a rank is idle."""
        tracker = WindowTracker(
            num_layers=1,
            num_physical_experts=6,
            expert_to_rank=[0, 0, 1, 1, 2, 2],
            num_ranks=3,
            window_size=1,
        )
        rec = tracker.begin_step()
        rec.record(0, [0, 0, 0, 2])  # rank totals [3,1,0]
        tracker.finish_step()
        r = tracker.close_window().layer_ratios(0)
        self.assertEqual(r["rank_totals"], [3, 1, 0])
        # moe_lb = 3 / (4/3) = 9/4 ; window = 3 / (4/3) = 9/4
        self.assertEqual(r["moe_lb"], Fraction(9, 4))
        self.assertEqual(r["moe_lb_window"], Fraction(9, 4))


if __name__ == "__main__":
    unittest.main()
