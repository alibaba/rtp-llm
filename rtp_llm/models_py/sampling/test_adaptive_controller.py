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
    def test_external_caps_do_not_train_unverified_rejections(self) -> None:
        controller = ready_controller([10, 20])
        caps, _, _ = controller.prepare_from_host(
            [10, 20],
            confidence=None,
            fixed_cap=3,
            draft_cost_ms=np.zeros(3),
            verify_cost_ms=np.ones(9),
        )
        controller.commit_prepared([10, 20], caps)
        # Reordered rows: request20 has cap0, request10 has cap1 and accepted it.
        effective = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        with patch.object(
            controller.estimator, "step", wraps=controller.estimator.step
        ) as step:
            controller.observe(
                [20, 10],
                effective.clone(),
                [False, False],
                effective_capacities=effective,
            )
        torch.testing.assert_close(
            step.call_args.args[1],
            torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(
            step.call_args.args[2], torch.zeros(2, device="cuda", dtype=torch.int32)
        )
        torch.testing.assert_close(
            controller.estimator.counts,
            torch.tensor([1.0, 0.0, 0.0], device="cuda"),
        )
        self.assertIsNone(controller._pending)

    def test_session_forwards_external_caps(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}):
            session = AdaptiveVerificationSession(4, 3, "cuda")
        session.begin([10, 20])
        accepted = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        effective = torch.tensor([1, 1], device="cuda", dtype=torch.int32)
        with patch.object(session.controller, "observe") as observe:
            session.observe(accepted, [False, True], effective)
        args, kwargs = observe.call_args
        self.assertEqual(args[0], [10, 20])
        self.assertIs(args[1], accepted)
        self.assertEqual(args[2], [False, True])
        self.assertIs(kwargs["effective_capacities"], effective)
        self.assertTrue(kwargs["asynchronous"])

    def test_external_cap_validation_preserves_pending_plan(self) -> None:
        controller = ready_controller([10, 20])
        caps, _, _ = controller.prepare_from_host(
            [10, 20],
            confidence=None,
            fixed_cap=1,
            draft_cost_ms=np.zeros(3),
            verify_cost_ms=np.ones(9),
        )
        controller.commit_prepared([10, 20], caps)
        accepted = torch.zeros(2, dtype=torch.int32, device="cuda")
        for effective in (
            torch.zeros(1, device="cuda", dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, device="cuda"),
            torch.tensor([-1, 1], device="cuda", dtype=torch.int32),
            torch.tensor([2, 1], device="cuda", dtype=torch.int32),
        ):
            with self.subTest(effective=effective):
                with self.assertRaises(AdaptiveBudgetInputError):
                    controller.observe(
                        [10, 20],
                        accepted,
                        [False, False],
                        effective_capacities=effective,
                    )
                self.assertIsNotNone(controller._pending)
        controller.observe(
            [10, 20],
            accepted,
            [False, False],
            effective_capacities=caps,
            asynchronous=True,
        )
        self.assertIsNone(controller._pending)

    def test_metrics_host_counts_and_decision_modes(self) -> None:
        for cap in (0, 1, 2):
            with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": str(cap)}):
                session = AdaptiveVerificationSession(2, 2, "cuda")
            session.begin([10, 20])
            with (
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("GPU read")
                ),
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("GPU read")
                ),
            ):
                mode, zero, full, latency = session.metrics_snapshot()
            self.assertEqual(
                (mode, zero, full), (0, 2 if cap == 0 else 0, 2 if cap == 2 else 0)
            )
            self.assertGreaterEqual(latency, 0)
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        session.begin([10, 20])
        self.assertEqual(session.metrics_snapshot()[:3], (1, 0, 2))
        session.round_costs[2] = {2: 1.0, 4: 2.0, 6: 3.0}
        session._prepare_plan()
        self.assertEqual(session.metrics_snapshot()[:3], (3, 0, 2))
        session._host_predictions = {
            i: (session.round, np.array([0.9, 0.8])) for i in (10, 20)
        }
        session._prepare_plan()
        self.assertEqual(session.metrics_snapshot()[0], 2)
        session.begin([])
        self.assertEqual(session.metrics_snapshot()[1:3], (0, 0))

    def test_dynamic_session_plan_does_not_wait_for_gpu(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        logits = torch.tensor([[4.0, 0.0], [0.0, 1.0]], device="cuda")

        def run_round() -> None:
            session.begin([10, 20])
            for step in range(2):
                session.predict([10, 20], step, logits)
            caps, _, _ = session.plan()
            session.observe(caps, [False, False])
            session.finish()

        for _ in range(4):
            run_round()
            torch.cuda.synchronize()
        # Force the learned-budget branch, independently of noisy GPU timing.
        session.round_costs[2] = {2: 1.0, 4: 1.5, 6: 2.0}
        session.pending_costs = []
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            with torch.profiler.record_function("adaptive_dynamic_round"):
                run_round()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        scope = next(e for e in events if e.get("name") == "adaptive_dynamic_round")
        waits = [
            e["name"]
            for e in events
            if e.get("ph") == "X"
            and e.get("tid") == scope["tid"]
            and scope["ts"] <= e["ts"] < scope["ts"] + scope["dur"]
            and ("Synchronize" in e["name"] or e["name"] == "aten::item")
        ]
        self.assertEqual(waits, [])
        self.assertFalse(session._calibration)

    def test_unmeasured_rows_do_not_invent_fractional_graph_savings(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        session.begin([10, 20])
        session.round_costs[2] = {2: 1.0, 4: 3.0, 6: 5.0}
        with patch.object(
            session.controller,
            "prepare_from_host",
            wraps=session.controller.prepare_from_host,
        ) as prepare:
            session._prepare_plan()
        np.testing.assert_array_equal(
            prepare.call_args.kwargs["verify_cost_ms"][2:], [1.0, 3.0, 3.0, 5.0, 5.0]
        )
        # Preserve real nonmonotonic measurements; do not inflate full-cap costs.
        session.round_costs[2][4] = 0.5
        with patch.object(
            session.controller,
            "prepare_from_host",
            wraps=session.controller.prepare_from_host,
        ) as prepare:
            session._prepare_plan()
        np.testing.assert_array_equal(
            prepare.call_args.kwargs["verify_cost_ms"][2:], [1.0, 1.0, 0.5, 5.0, 5.0]
        )

    def test_host_plan_keeps_prefixes_and_matches_budget(self) -> None:
        controller = ready_controller([10, 20])
        costs = dict(draft_cost_ms=np.zeros(3), verify_cost_ms=np.ones(9))
        for cap in (0, 1, 3):
            caps, rows, total = controller.prepare_from_host(
                [20, 10], confidence=None, fixed_cap=cap, **costs
            )
            self.assertEqual(caps.tolist(), [cap, cap])
            self.assertEqual(
                rows.tolist(), list(range(cap + 1)) + list(range(4, 5 + cap))
            )
            self.assertEqual(total, 2 * (cap + 1))
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("GPU read")):
            caps, rows, total = controller.prepare_from_host(
                [10, 20],
                confidence=np.array([[0.8, 0.6, 0.3], [0.9, 0.4, 0.1]]),
                **costs,
            )
        expected = [
            r * 4 + j for r, cap in enumerate(caps.tolist()) for j in range(cap + 1)
        ]
        self.assertEqual(rows.tolist(), expected)
        self.assertEqual(total, len(expected))
        controller.commit_prepared([10, 20], caps)
        controller.observe([20, 10], caps.flip(0), [False, False])
        self.assertIsNone(controller._pending)

    def test_snapshot_wait_is_nonblocking_and_reused_id_is_isolated(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        session.begin([10])
        epoch = session._request_epochs[10]
        ready = Mock(spec=torch.cuda.Event)
        ready.query.return_value = False
        host = torch.tensor([[0.2, 0.1]])
        session._pending_predictions.append(((10,), (epoch,), 0, host, ready))
        session._collect_predictions()
        self.assertEqual(session._host_predictions, {})
        ready.synchronize.assert_not_called()
        session.begin([20])
        session.begin([10])
        self.assertNotEqual(epoch, session._request_epochs[10])
        ready.query.return_value = True
        session._collect_predictions()
        self.assertEqual(session._host_predictions, {})
        self.assertEqual(session._pending_predictions, [])

    def test_real_async_snapshot_reorders_without_changing_prepared_plan(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        session.begin([10, 20])
        first_caps, first_rows, first_total = session.prepared_plan()
        for step in range(2):
            session.predict(
                [10, 20], step, torch.tensor([[4.0, 0.0], [0.0, 1.0]], device="cuda")
            )
        caps, rows, total = session.plan()
        self.assertEqual(caps.data_ptr(), first_caps.data_ptr())
        self.assertEqual(rows.data_ptr(), first_rows.data_ptr())
        self.assertEqual(total, first_total)
        session.observe(caps, [False, False])
        session.finish()
        session._feedback_stream.synchronize()
        expected = session.controller.confidence_probs.cpu().numpy().copy()
        session.begin([20, 10])
        np.testing.assert_array_equal(session._host_predictions[10][1], expected[0])
        np.testing.assert_array_equal(session._host_predictions[20][1], expected[1])
        self.assertEqual(session.ids, [20, 10])

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
        session.observe(torch.tensor([3], device="cuda"), [False])
        session.finish()
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
        session.observe(torch.tensor([3, 1], device="cuda"), [False, False])
        session.finish()
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
            session.observe(caps, [False])
            session.finish()
            predict.assert_not_called()
            train.assert_not_called()

    def test_round_cost_is_finished_after_sampling_and_includes_host_tail(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(1, 2, "cuda")
        session.begin([10])
        caps, _, total = session.plan()
        session.observe(caps, [False])
        self.assertEqual(session.pending_costs, [])
        self.assertIsNotNone(session.events)
        events = [Mock(spec=torch.cuda.Event) for _ in range(2)]
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 2.0
        session.events = events
        session.finish(7.0)
        events[1].record.assert_called_once()
        session.finish(99.0)  # Finishing twice cannot add duplicate observations.
        self.assertEqual(len(session.pending_costs), 1)
        session._collect_costs()
        self.assertEqual(session.round_costs[1][total], 7.0)  # max, not 2+7.
        events[1].synchronize.assert_not_called()

    def test_bad_wall_clock_telemetry_cannot_fail_forward(self) -> None:
        for invalid in (-1.0, float("nan"), float("inf")):
            with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
                session = AdaptiveVerificationSession(1, 2, "cuda")
            session.begin([10])
            events = [Mock(spec=torch.cuda.Event) for _ in range(2)]
            events[1].query.return_value = True
            events[0].elapsed_time.return_value = 2.0
            session.events = events
            with patch(
                "rtp_llm.models_py.sampling.adaptive_session.time.perf_counter",
                return_value=session._cpu_round_start + 0.004,
            ):
                session.finish(invalid)
            session._collect_costs()
            self.assertAlmostEqual(session.round_costs[1][session.total], 4.0, places=4)

    def test_invalid_prediction_snapshot_falls_back_without_gpu_read(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(1, 2, "cuda")
        session.begin([10])
        session.round_costs[1] = {1: 1.0, 2: 2.0, 3: 3.0}
        ready = Mock(spec=torch.cuda.Event)
        ready.query.return_value = True
        for bad in (float("nan"), float("inf"), -0.1, 1.1):
            session._host_predictions[10] = (0, np.array([0.2, 0.1]))
            session._pending_predictions.append(
                (
                    (10,),
                    (session._request_epochs[10],),
                    0,
                    torch.tensor([[bad, 0.1]]),
                    ready,
                )
            )
            session._collect_predictions()
            with patch.object(
                torch.Tensor, "cpu", side_effect=AssertionError("GPU read")
            ):
                session._prepare_plan()
            self.assertEqual(session.prepared_plan()[0].tolist(), [2])
        ready.synchronize.assert_not_called()

    def test_unfinished_costs_are_deferred_without_waiting(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 5, "cuda")
        events = [Mock(spec=torch.cuda.Event) for _ in range(2)]
        events[1].query.return_value = False
        session.pending_costs.append((1, 4, events, 3.0))
        session._collect_costs()
        self.assertEqual(len(session.pending_costs), 1)
        events[0].elapsed_time.assert_not_called()
        events[1].synchronize.assert_not_called()
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 2.0
        session._collect_costs()
        self.assertEqual(session.pending_costs, [])
        self.assertEqual(session.round_costs[1][4], 3.0)
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
            session.observe(caps, [False])
            session.finish()
        self.assertEqual(observed[:3], [2, 1, 0])
        self.assertIn(observed[3], (0, 1, 2))
        self.assertTrue(all(cost > 0 for cost in session.round_costs[1].values()))

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
        self.assertIsNone(controller._pending)
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
