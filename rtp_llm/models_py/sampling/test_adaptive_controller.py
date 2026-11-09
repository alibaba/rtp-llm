import json
import math
import os
import pathlib
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from rtp_llm.models_py.sampling.adaptive_controller import AdaptiveDraftController
from rtp_llm.models_py.sampling.adaptive_runtime import create_controller
from rtp_llm.models_py.sampling.adaptive_session import AdaptiveVerificationSession
from rtp_llm.models_py.sampling.adaptive_verification import AdaptiveBudgetInputError


def ready_controller(request_ids: list[int]) -> AdaptiveDraftController:
    controller = AdaptiveDraftController(4, 3, torch.device("cuda"))
    for request_id in request_ids:
        controller.register(request_id)
    for step in range(3):
        logits = torch.tensor(
            [[float(request_id % 7 + step), 0.0] for request_id in request_ids],
            device="cuda",
        )
        controller.predict(request_ids, step, logits)
    return controller


class AdaptiveControllerTest(unittest.TestCase):
    def test_unfinished_round_is_invalidated_before_retry(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}):
            session = AdaptiveVerificationSession(2, 5, "cuda")
        session.begin([10])
        for step in range(5):
            session.predict([10], step, torch.ones((1, 8), device="cuda"))
        session.plan()
        session.begin([20])
        caps, _, total = session.plan()
        self.assertEqual((caps.item(), total), (3, 4))
        session.target_done()
        session.observe(torch.tensor([3], device="cuda"), [False])
        self.assertEqual(session.controller.estimator.counts.tolist(), [0] * 5)

    def test_five_proposals_fixed_three_verification_rows(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}):
            session = AdaptiveVerificationSession(4, 5, "cuda")
        session.begin([10, 20])
        for step in range(5):
            session.predict([10, 20], step, torch.ones((2, 8), device="cuda"))
        caps, rows, total = session.plan()
        self.assertEqual(caps.tolist(), [3, 3])
        self.assertEqual(rows.tolist(), [0, 1, 2, 3, 6, 7, 8, 9])
        self.assertEqual(total, 8)
        session.target_done()
        session.observe(torch.tensor([3, 1], device="cuda"), [False, False])
        self.assertEqual(session.controller.estimator.counts.tolist(), [0] * 5)
        self.assertIsNone(session.events)
        self.assertEqual(session.pending_costs, [])

    def test_fixed_cap_skips_prediction_and_training(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}):
            session = AdaptiveVerificationSession(2, 5, "cuda")
        with (
            patch.object(session.controller.estimator, "predict") as predict,
            patch.object(session.controller.estimator, "step") as train,
        ):
            session.begin([10])
            for step in range(5):
                session.predict([10], step, torch.ones((1, 8), device="cuda"))
            caps, _, _ = session.plan()
            session.target_done()
            session.observe(caps, [False])
            predict.assert_not_called()
            train.assert_not_called()

    def test_unfinished_costs_are_deferred_without_waiting(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 5, "cuda")
        events = [Mock(spec=torch.cuda.Event) for _ in range(4)]
        events[3].query.return_value = False
        session.pending_costs.append((1, 4, events))
        session._collect_costs()
        self.assertEqual(len(session.pending_costs), 1)
        events[0].elapsed_time.assert_not_called()
        events[2].elapsed_time.assert_not_called()
        events[3].synchronize.assert_not_called()
        events[3].query.return_value = True
        events[0].elapsed_time.return_value = 2.0
        events[2].elapsed_time.return_value = 3.0
        session._collect_costs()
        self.assertEqual(session.pending_costs, [])
        self.assertEqual(session.draft_costs[1], 2.0)
        self.assertEqual(session.target_costs[1][4], 3.0)
        self.assertEqual(session.event_pool, [events])

    def test_async_prediction_and_feedback_do_not_synchronize_host(self) -> None:
        controller = ready_controller([10, 20])
        accepted = torch.tensor([1, 2], device="cuda")
        logits = torch.ones((2, 8), device="cuda")
        costs = dict(draft_cost_ms=np.zeros(3), verify_cost_ms=np.ones(9))
        controller.plan([10, 20], [3, 3], fixed_cap=3, **costs)

        def run_round() -> None:
            controller.observe([10, 20], accepted, [False, False], asynchronous=True)
            for step in range(3):
                controller.predict([10, 20], step, logits)
            controller.plan([10, 20], [3, 3], fixed_cap=3, **costs)

        run_round()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.profiler.record_function("adaptive_async_round"):
                run_round()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        scope = next(e for e in events if e.get("name") == "adaptive_async_round")
        synchronizations = [
            e["name"]
            for e in events
            if e.get("ph") == "X"
            and e.get("tid") == scope["tid"]
            and scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
            and ("Synchronize" in e["name"] or e["name"] == "aten::item")
        ]
        self.assertEqual(synchronizations, [])
        self.assertTrue(torch.isfinite(controller.estimator.grad).all().item())

    def test_cached_indices_are_safe_across_streams_and_eviction(self) -> None:
        controller = AdaptiveDraftController(2, 3, torch.device("cuda"))
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            first = controller._device_indices([1, 0])
        with torch.cuda.stream(consumer):
            reused = controller._device_indices([1, 0])
            self.assertEqual(first.data_ptr(), reused.data_ptr())
            result = reused + 7
            for index in range(40):
                controller._device_indices([index])
        consumer.synchronize()
        self.assertEqual(result.tolist(), [8, 7])
        self.assertLessEqual(len(controller._index_cache), 32)

    def test_dynamic_budget_collects_real_costs_before_selecting(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        observed = []
        for iteration in range(4):
            session.begin([10])
            for step in range(2):
                session.predict([10], step, torch.ones((1, 8), device="cuda"))
            caps, _, total = session.plan()
            observed.append(caps.item())
            torch.ones((total, 128), device="cuda").square()
            session.target_done()
            session.observe(caps, [False])
        self.assertEqual(observed[:3], [2, 1, 0])
        self.assertIn(observed[3], (0, 1, 2))
        self.assertTrue(all(cost > 0 for cost in session.target_costs[1].values()))

    def test_opt_in_is_independent_and_can_be_disabled_for_next_start(self) -> None:
        for gumbel in ("0", "1"):
            for rejection in ("0", "1"):
                env = {
                    "RTP_LLM_ADAPTIVE_VERIFICATION": "1",
                    "RTP_LLM_USE_GUMBEL_SAMPLING": gumbel,
                    "RTP_LLM_USE_VLLM_REJECTION_SAMPLING": rejection,
                }
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    controller = create_controller(2, 3, "cuda")
                    self.assertIsInstance(controller, AdaptiveDraftController)
                    controller.register(10)
                    controller.predict(
                        [10], 0, torch.tensor([[1.0, 0.0]], device="cuda")
                    )
                    os.environ["RTP_LLM_ADAPTIVE_VERIFICATION"] = "0"
                    before = torch.cuda.memory_allocated()
                    self.assertIsNone(create_controller(2, 3, "cuda"))
                    self.assertEqual(torch.cuda.memory_allocated(), before)
                    self.assertEqual(os.environ["RTP_LLM_USE_GUMBEL_SAMPLING"], gumbel)
                    self.assertEqual(
                        os.environ["RTP_LLM_USE_VLLM_REJECTION_SAMPLING"], rejection
                    )

    def test_predictions_follow_slots_and_temperature_after_reorder(self) -> None:
        controller = AdaptiveDraftController(3, 3, torch.device("cuda"))
        first, second = controller.register(10), controller.register(20)
        controller.predict(
            [10, 20], 0, torch.tensor([[4.0, 0.0], [0.0, 0.0]], device="cuda")
        )
        controller.predict(
            [20, 10], 1, torch.tensor([[2.0, 0.0], [1.0, 0.0]], device="cuda")
        )
        controller.predict(
            [10, 20],
            2,
            torch.tensor([[4.0, 0.0], [6.0, 0.0]], device="cuda"),
            torch.tensor([2.0, 0.0], device="cuda"),
        )

        expected = torch.tensor([[4.0, 1.0, 2.0], [0.0, 2.0, 6.0]])
        torch.testing.assert_close(
            controller.estimator.features[[first, second]].cpu(),
            expected,
            atol=2e-4,
            rtol=2e-4,
        )
        torch.testing.assert_close(
            controller.confidence_probs[[first, second]].cpu(),
            torch.sigmoid(0.5 * expected - 0.2),
            atol=2e-4,
            rtol=2e-4,
        )
        verify_costs = np.ones(9)
        verify_costs[3:] = 1e6
        caps, _, total = controller.plan(
            [20, 10], [3, 3], draft_cost_ms=np.zeros(3), verify_cost_ms=verify_costs
        )
        self.assertEqual((caps.tolist(), total), ([0, 0], 2))

    def test_unknown_seed_keeps_full_capacity_and_is_not_trained(self) -> None:
        controller = AdaptiveDraftController(2, 3, torch.device("cuda"))
        controller.register(10)
        controller.predict([10], 1, torch.tensor([[5.0, 0.0]], device="cuda"))

        caps, indices, total = controller.plan(
            [10],
            [3],
            draft_cost_ms=np.zeros(2),
            verify_cost_ms=np.array([1.0, 1.0, 1e6, 1e6, 1e6]),
        )
        controller.observe([10], torch.tensor([1], device="cuda"), [False])

        self.assertEqual(caps.tolist(), [3])
        self.assertEqual(indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(total, 4)
        torch.testing.assert_close(controller.estimator.counts.cpu(), torch.zeros(3))
        torch.testing.assert_close(controller.estimator.grad.cpu(), torch.zeros(3, 2))

    def test_training_excludes_forced_requests_and_unobserved_suffix(self) -> None:
        controller = ready_controller([10, 20, 30])
        controller.plan(
            [10, 20, 30],
            [3, 3, 2],
            draft_cost_ms=np.zeros(4),
            verify_cost_ms=np.ones(12),
        )
        expected_info, expected_grad = torch.zeros(3, 3), torch.zeros(3, 2)
        for request_id, step, label in ((30, 0, 0), (10, 0, 1), (10, 1, 0)):
            feature = float(request_id % 7 + step)
            probability = 1 / (1 + math.exp(-(0.5 * feature - 0.2)))
            row = torch.tensor([feature, 1.0])
            hessian = torch.outer(row, row) * probability * (1 - probability)
            expected_info[step] += torch.stack(
                (hessian[0, 0], hessian[0, 1], hessian[1, 1])
            )
            expected_grad[step] += row * (label - probability)

        controller.observe(
            [30, 10, 20], torch.tensor([0, 1, 3], device="cuda"), [False, False, True]
        )

        torch.testing.assert_close(
            controller.estimator.counts.cpu(), torch.tensor([2.0, 1.0, 0.0])
        )
        torch.testing.assert_close(
            controller.estimator.info.cpu(), expected_info, atol=2e-5, rtol=2e-5
        )
        torch.testing.assert_close(
            controller.estimator.grad.cpu(), expected_grad, atol=2e-5, rtol=2e-5
        )

    def test_zero_partial_and_full_budgets_preserve_dense_row_order(self) -> None:
        request_ids, capacities = [30, 10, 20], [0, 2, 3]
        for budget in (0, 2, 5):
            with self.subTest(budget=budget):
                controller = ready_controller([10, 20, 30])
                verify_costs = np.ones(9)
                verify_costs[3 + budget + 1 :] = 1e6
                expected = [0, 0, 0]
                probabilities = [
                    [
                        1 / (1 + math.exp(-(0.5 * (request_id % 7 + step) - 0.2)))
                        for step in range(3)
                    ]
                    for request_id in request_ids
                ]
                for _ in range(budget):
                    candidates = [i for i in range(3) if expected[i] < capacities[i]]
                    selected = max(
                        candidates,
                        key=lambda i: (
                            math.prod(probabilities[i][: expected[i] + 1]),
                            -i,
                        ),
                    )
                    expected[selected] += 1
                expected_rows = [
                    row * 4 + step
                    for row in range(3)
                    for step in range(expected[row] + 1)
                ]

                caps, indices, total = controller.plan(
                    request_ids,
                    capacities,
                    draft_cost_ms=np.zeros(4),
                    verify_cost_ms=verify_costs,
                )

                self.assertEqual(caps.tolist(), expected)
                self.assertEqual(indices.tolist(), expected_rows)
                self.assertEqual(total, len(expected_rows))

    def test_pending_verification_prevents_seed_overwrite_and_slot_reuse(self) -> None:
        controller = ready_controller([10])
        slot = controller.register(10)
        controller.plan([10], [3], draft_cost_ms=np.zeros(2), verify_cost_ms=np.ones(5))
        with self.assertRaisesRegex(AdaptiveBudgetInputError, "observe"):
            controller.predict([10], 0, torch.tensor([[1.0, 0.0]], device="cuda"))
        with self.assertRaisesRegex(AdaptiveBudgetInputError, "observe"):
            controller.release(10)

        controller.observe([10], torch.tensor([0], device="cuda"), [False])
        controller.predict([10], 0, torch.tensor([[1.0, 0.0]], device="cuda"))
        controller.release(10)
        new_slot = controller.register(40)

        self.assertEqual(new_slot, slot)
        torch.testing.assert_close(
            controller.confidence_probs[slot].cpu(), torch.ones(3)
        )
        torch.testing.assert_close(
            controller.estimator.features[slot].cpu(), torch.zeros(3)
        )
        caps, rows, total = controller.plan(
            [40],
            [2],
            draft_cost_ms=np.zeros(2),
            verify_cost_ms=np.array([1.0, 1.0, 1e6, 1e6]),
        )
        self.assertEqual((caps.tolist(), rows.tolist(), total), ([2], [0, 1, 2], 3))

    def test_observe_rejects_accepted_count_above_admitted_count(self) -> None:
        controller = ready_controller([10])
        controller.plan([10], [1], draft_cost_ms=np.zeros(2), verify_cost_ms=np.ones(3))

        with self.assertRaisesRegex(AdaptiveBudgetInputError, "outside admitted"):
            controller.observe([10], torch.tensor([2], device="cuda"), [False])
        controller.observe([10], torch.tensor([0], device="cuda"), [False])

        torch.testing.assert_close(
            controller.estimator.counts.cpu(), torch.tensor([1.0, 0.0, 0.0])
        )

    def test_empty_batch_and_zero_proposals_keep_anchor_only(self) -> None:
        controller = AdaptiveDraftController(2, 3, torch.device("cuda"))
        controller.predict([], 0, torch.empty(0, 2, device="cuda"))
        caps, rows, total = controller.plan(
            [], [], draft_cost_ms=np.array([]), verify_cost_ms=np.array([])
        )
        controller.observe([], torch.empty(0, dtype=torch.int32, device="cuda"), [])
        self.assertEqual((caps.numel(), rows.numel(), total), (0, 0, 0))
        controller.register(10)

        caps, rows, total = controller.plan(
            [10], [0], draft_cost_ms=np.zeros(2), verify_cost_ms=np.ones(2)
        )
        controller.observe([10], torch.tensor([0], device="cuda"), [False])

        self.assertEqual((caps.tolist(), rows.tolist(), total), ([0], [0], 1))
        torch.testing.assert_close(controller.estimator.counts.cpu(), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
