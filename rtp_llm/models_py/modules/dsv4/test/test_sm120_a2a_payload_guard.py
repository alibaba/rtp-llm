"""CPU regression for the SM120 MoE all-to-all payload guard and its fallback warning.

The all-to-all dispatch rejects a batch whose summed recv_counts exceeds a fixed token bound and
falls back to the fixed-EP path. That fallback is correct, but it allocates an
O(world * tokens * hidden) FP32 reduction buffer per layer, so on a prefill leg it is a large
latency cliff -- and it used to be observable only under a debug env var, which let a prefill
admission bound larger than the guard roughly double time-to-first-token with nothing in the log.

These tests pin:

  * the guard's boundary semantics, so the fast all-to-all path is not accidentally narrowed and the
    safe fallback is still taken for an oversized or failed count exchange; and
  * the warning's content and rate limit, so the cliff stays observable in production without a
    per-call log flood.

Pure Python: no CUDA, no torch.distributed, no flashinfer.
"""

from __future__ import annotations

import logging
import unittest

from rtp_llm.models_py.modules.dsv4.moe.strategies import deepep

BOUND = deepep._A2A_PAYLOAD_TOKEN_BOUND


class A2APayloadGuardTest(unittest.TestCase):
    def setUp(self):
        # The warning counter is module state; reset it so each test starts unbudgeted.
        deepep._A2A_OVERSIZE_WARN_CT[0] = 0

    # ---------------------------------------------------------------- guard boundary

    def test_bound_is_the_documented_payload_limit(self):
        self.assertEqual(BOUND, 65536)

    def test_at_or_under_the_bound_stays_on_the_all_to_all_path(self):
        self.assertFalse(deepep._a2a_payload_bound_exceeded([BOUND // 4] * 4))  # sum == bound
        self.assertFalse(deepep._a2a_payload_bound_exceeded([BOUND // 4 - 1] * 4))
        self.assertFalse(deepep._a2a_payload_bound_exceeded([8000] * 4))  # one 32K prompt under CP4
        self.assertFalse(deepep._a2a_payload_bound_exceeded([16000] * 4))  # two 32K prompts under CP4
        self.assertFalse(deepep._a2a_payload_bound_exceeded([]))  # empty batch

    def test_over_the_bound_falls_back(self):
        # Exactly one token over the bound: 16385 + 3*16384 == 65537. Pins the boundary against the
        # sum==bound case above (65536 stays on the fast path).
        self.assertTrue(deepep._a2a_payload_bound_exceeded([BOUND // 4 + 1] + [BOUND // 4] * 3))
        # The recorded prefill shapes that hit the cliff: 3x32K and 4x32K sharded over CP4.
        self.assertTrue(deepep._a2a_payload_bound_exceeded([24000] * 4))  # sum 96000
        self.assertTrue(deepep._a2a_payload_bound_exceeded([32000] * 4))  # sum 128000

    def test_a_single_oversized_count_falls_back_even_when_the_sum_is_small(self):
        # The guard is per-count as well as on the sum: one corrupt count must not be dispatched.
        self.assertTrue(deepep._a2a_payload_bound_exceeded([BOUND + 1, 0, 0, 0]))

    def test_negative_count_sentinel_falls_back(self):
        self.assertTrue(deepep._a2a_payload_bound_exceeded([-1, 8000, 8000, 8000]))

    def test_an_unsummable_count_list_takes_the_safe_branch(self):
        self.assertTrue(deepep._a2a_payload_bound_exceeded(None))
        self.assertTrue(deepep._a2a_payload_bound_exceeded(["not-a-count"] * 4))

    # ---------------------------------------------------------------- warning

    def test_the_fallback_warns_and_names_the_knob_to_cap(self):
        with self.assertLogs(deepep._logger, level=logging.WARNING) as cm:
            deepep._warn_a2a_oversize([24000] * 4)
        self.assertEqual(len(cm.output), 1)
        msg = cm.output[0]
        self.assertIn("fixed-EP", msg)
        self.assertIn("max_batch_tokens_size", msg)  # actionable: the admission bound to lower
        self.assertIn("96000", msg)  # the observed sum
        self.assertIn("65536", msg)  # the bound

    def test_the_warning_is_rate_limited(self):
        budget = deepep._A2A_OVERSIZE_WARN_MAX
        with self.assertLogs(deepep._logger, level=logging.WARNING) as cm:
            for _ in range(budget + 5):
                deepep._warn_a2a_oversize([32000] * 4)
        self.assertEqual(len(cm.output), budget)
        self.assertEqual(deepep._A2A_OVERSIZE_WARN_CT[0], budget)

    def test_once_the_budget_is_spent_nothing_more_is_logged(self):
        deepep._A2A_OVERSIZE_WARN_CT[0] = deepep._A2A_OVERSIZE_WARN_MAX
        with self.assertNoLogs(deepep._logger, level=logging.WARNING):
            deepep._warn_a2a_oversize([32000] * 4)

    def test_an_unsummable_count_list_still_warns_without_raising(self):
        with self.assertLogs(deepep._logger, level=logging.WARNING) as cm:
            deepep._warn_a2a_oversize(None)
        self.assertIn("sum=-1", cm.output[0])

    def test_the_guard_and_the_warning_agree(self):
        # Every shape the guard rejects must produce a warning, and every shape it accepts must not.
        for counts in ([8000] * 4, [16000] * 4, [BOUND // 4] * 4, []):
            self.assertFalse(deepep._a2a_payload_bound_exceeded(counts), counts)
        for counts in ([24000] * 4, [32000] * 4, [-1, 0, 0, 0], [BOUND + 1]):
            deepep._A2A_OVERSIZE_WARN_CT[0] = 0
            self.assertTrue(deepep._a2a_payload_bound_exceeded(counts), counts)
            with self.assertLogs(deepep._logger, level=logging.WARNING):
                deepep._warn_a2a_oversize(counts)


if __name__ == "__main__":
    unittest.main()
