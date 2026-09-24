"""CPU tests for all-to-all capacity boundaries, fallback routing and warning limits.

Quantization and collectives are mocked; routing uses CPU tensors.
"""

from __future__ import annotations

import logging
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies import sm120_decode as deepep

BOUND = deepep._A2A_PAYLOAD_TOKEN_BOUND


class A2APayloadGuardTest(unittest.TestCase):
    def setUp(self):
        # The warning counter is module state; reset it so each test starts unbudgeted.
        deepep._A2A_OVERSIZE_WARN_CT[0] = 0

    # ---------------------------------------------------------------- guard boundary

    def test_bound_is_the_documented_payload_limit(self):
        self.assertEqual(BOUND, 65536)

    def test_at_or_under_the_bound_stays_on_the_all_to_all_path(self):
        self.assertFalse(
            deepep._a2a_payload_bound_exceeded([BOUND // 4] * 4)
        )  # sum == bound
        self.assertFalse(deepep._a2a_payload_bound_exceeded([BOUND // 4 - 1] * 4))
        self.assertFalse(
            deepep._a2a_payload_bound_exceeded([8000] * 4)
        )  # one 32K prompt under CP4
        self.assertFalse(
            deepep._a2a_payload_bound_exceeded([16000] * 4)
        )  # two 32K prompts under CP4
        self.assertFalse(deepep._a2a_payload_bound_exceeded([]))  # empty batch

    def test_over_the_bound_falls_back(self):
        # Exactly one token over the bound: 16385 + 3*16384 == 65537. Pins the boundary against the
        # sum==bound case above (65536 stays on the fast path).
        self.assertTrue(
            deepep._a2a_payload_bound_exceeded([BOUND // 4 + 1] + [BOUND // 4] * 3)
        )
        # Oversized prefill batches sharded over CP4.
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
        self.assertIn(
            "max_batch_tokens_size", msg
        )  # actionable: the admission bound to lower
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


class DispatchRoutingTest(unittest.TestCase):
    def test_prepared_fallback_does_not_require_a2a_payload(self):
        strategy = SimpleNamespace(
            _forward_sm120_fixed_ep=mock.Mock(return_value="fixed"),
            _run_sm120_all_to_all_prepared=mock.Mock(),
        )
        prep = dict(
            mode="fixed_ep", x=object(), weights=object(), indices=object(), pad_floor=8
        )
        result = deepep.Sm120DecodeStrategy.run_dispatch_prepared(strategy, prep)
        self.assertEqual(result, "fixed")
        strategy._forward_sm120_fixed_ep.assert_called_once_with(
            prep["x"], prep["weights"], prep["indices"], pad_floor=8
        )
        strategy._run_sm120_all_to_all_prepared.assert_not_called()

    def test_prepared_a2a_preserves_the_payload(self):
        strategy = SimpleNamespace(
            _forward_sm120_fixed_ep=mock.Mock(),
            _run_sm120_all_to_all_prepared=mock.Mock(return_value="a2a"),
        )
        prep = dict(mode="a2a", recv_counts=[1, 2])
        self.assertEqual(
            deepep.Sm120DecodeStrategy.run_dispatch_prepared(strategy, prep), "a2a"
        )
        strategy._run_sm120_all_to_all_prepared.assert_called_once_with(prep)
        strategy._forward_sm120_fixed_ep.assert_not_called()

    def test_stock_path_uses_the_same_mode_router(self):
        for mode in ("fixed_ep", "a2a"):
            with self.subTest(mode=mode):
                prep = dict(mode=mode)
                strategy = SimpleNamespace(
                    _prepare_sm120_all_to_all=mock.Mock(return_value=prep),
                    run_dispatch_prepared=mock.Mock(return_value="output"),
                )
                self.assertEqual(
                    deepep.Sm120DecodeStrategy._forward_sm120_all_to_all_impl(
                        strategy, "x", "weights", "indices"
                    ),
                    "output",
                )
                strategy.run_dispatch_prepared.assert_called_once_with(prep)

    def test_all_to_all_failure_is_logged_and_reraised(self):
        failure = RuntimeError("dispatch failure")
        strategy = SimpleNamespace(
            _forward_sm120_all_to_all_impl=mock.Mock(side_effect=failure)
        )
        with self.assertLogs(deepep._logger, level=logging.ERROR):
            with self.assertRaises(RuntimeError) as caught:
                deepep.Sm120DecodeStrategy._forward_sm120_all_to_all(
                    strategy, None, None, None
                )
        self.assertIs(caught.exception, failure)

    def test_every_prepare_exchanges_fresh_counts(self):
        strategy = SimpleNamespace(cfg=SimpleNamespace(n_routed_experts=4))
        x = torch.zeros((2, 128), dtype=torch.bfloat16)
        weights = torch.ones((2, 1), dtype=torch.float32)
        indices = torch.tensor([[0], [3]], dtype=torch.int64)

        def quantize(value, **kwargs):
            return value.to(torch.float8_e4m3fn), torch.zeros(
                (value.size(0), value.size(1) // 32), dtype=torch.uint8
            )

        counts = iter(([2, 3], [2, 4], [2, BOUND]))

        def exchange(output, local, **kwargs):
            self.assertEqual(local.tolist(), [2])
            output.copy_(torch.tensor(next(counts)).view(2, 1))

        flashinfer = SimpleNamespace(mxfp8_quantize=quantize)
        with mock.patch.dict(
            sys.modules, {"flashinfer": flashinfer}
        ), mock.patch.object(
            torch.distributed, "get_world_size", return_value=2
        ), mock.patch.object(
            torch.distributed, "all_gather_into_tensor", side_effect=exchange
        ) as gather, mock.patch.object(
            deepep, "_warn_a2a_oversize"
        ) as warn:
            first, second, fallback = [
                deepep.Sm120DecodeStrategy._prepare_sm120_all_to_all(
                    strategy, x, weights, indices
                )
                for _ in range(3)
            ]
        self.assertEqual(gather.call_count, 3)
        self.assertEqual(first["recv_counts"], [2, 3])
        self.assertEqual(second["recv_counts"], [2, 4])
        self.assertEqual(first["send_counts"], [2, 2])
        self.assertEqual(first["recv_payload"].shape, (5, 140))
        self.assertEqual(fallback["mode"], "fixed_ep")
        self.assertEqual(fallback["pad_floor"], BOUND)
        self.assertNotIn("recv_counts", fallback)
        warn.assert_called_once_with([2, BOUND])


if __name__ == "__main__":
    unittest.main()
