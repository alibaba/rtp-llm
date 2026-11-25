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
from rtp_llm.models_py.sampling.adaptive_session import (
    AdaptiveVerificationSession,
    _MeasuredEpPolicy,
)
from rtp_llm.models_py.sampling.adaptive_tail_budget import (
    RollingAcceptance,
    TailRiskEpPolicy,
)
from rtp_llm.models_py.sampling.adaptive_verification import (
    AdaptiveBudgetInputError,
    pack_ep_tail_rows,
)


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


class MeasuredEpPolicyTest(unittest.TestCase):
    def make_policy(self) -> _MeasuredEpPolicy:
        buckets = [1, 2, 4, 8, 14, 32, 48, 64, 72, 76, 80, 84]
        execution = np.arange(85)
        execution[1:] = np.asarray(buckets)[np.searchsorted(buckets, execution[1:])]
        return _MeasuredEpPolicy(5, execution)

    def run_initial_probes(self, costs, accepted):
        policy = self.make_policy()
        ids = list(range(14))
        observed = []
        for _ in range(9):
            rows, mode = policy.begin(ids)
            self.assertEqual(mode, 1)
            observed.append(rows)
            policy.record(
                policy.epoch, tuple(ids), rows, costs[rows], np.full(14, accepted[rows])
            )
        self.assertEqual(observed, [84] * 3 + [76] * 3 + [72] * 3)
        return policy, ids

    def test_faster_verify_loses_when_it_emits_fewer_tokens(self):
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 33.0, 72: 31.0}, {84: 4, 76: 3, 72: 3}
        )
        self.assertEqual(policy.begin(ids), (84, 3))

    def test_rejected_buckets_are_not_probed_again_in_same_cohort(self):
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 33.0, 72: 31.0}, {84: 4, 76: 3, 72: 3}
        )
        for _ in range(48):
            self.assertEqual(policy.begin(ids[::-1]), (84, 3))
        self.assertFalse(policy.needs_measurement())
        self.assertEqual(policy.begin([*ids[1:], 14]), (84, 1))
        self.assertTrue(policy.needs_measurement())

    def test_selects_proven_bucket_and_refreshes_full_width(self):
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 32.0, 72: 31.0}, {84: 4, 76: 4, 72: 3}
        )
        for _ in range(7):
            self.assertEqual(policy.begin(ids[::-1]), (76, 2))
        self.assertEqual(policy.begin(ids), (84, 1))
        for _ in range(3):
            self.assertEqual(policy.begin(ids), (84, 3))
        policy.record(policy.epoch, tuple(ids), 84, 35.0, np.full(14, 4))
        self.assertEqual(policy.begin(ids), (76, 2))

    def test_only_settled_ep_full_width_can_skip_rounds(self):
        session = AdaptiveVerificationSession.__new__(AdaptiveVerificationSession)
        session._ep_policy = None
        self.assertFalse(session.can_reuse_dense_fallback())
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 32.0, 72: 31.0}, {84: 4, 76: 4, 72: 3}
        )
        session._ep_policy = policy
        self.assertEqual(policy.begin(ids), (76, 2))
        self.assertFalse(session.can_reuse_dense_fallback())
        policy, _ = self.run_initial_probes(
            {84: 35.0, 76: 33.0, 72: 31.0}, {84: 4, 76: 3, 72: 3}
        )
        session._ep_policy = policy
        self.assertEqual(policy.begin(ids), (84, 3))
        self.assertTrue(session.can_reuse_dense_fallback())
        self.assertEqual(policy.begin([*ids[1:], 14]), (84, 1))
        self.assertFalse(session.can_reuse_dense_fallback())

    def test_refresh_retires_stale_winner(self):
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 32.0, 72: 31.0}, {84: 4, 76: 4, 72: 3}
        )
        for _ in range(7):
            self.assertEqual(policy.begin(ids), (76, 2))
        self.assertEqual(policy.begin(ids), (84, 1))
        policy.record(policy.epoch, tuple(ids), 84, 20.0, np.full(14, 4))
        self.assertEqual(policy.begin(ids), (84, 3))
        self.assertFalse(policy.needs_measurement())

    def test_missing_refresh_retires_winner_and_measurement(self):
        policy, ids = self.run_initial_probes(
            {84: 35.0, 76: 32.0, 72: 31.0}, {84: 4, 76: 4, 72: 3}
        )
        for _ in range(7):
            self.assertEqual(policy.begin(ids), (76, 2))
        self.assertEqual(policy.begin(ids), (84, 1))
        for _ in range(policy._SAMPLE_GRACE_ROUNDS):
            self.assertTrue(policy.needs_measurement())
            self.assertEqual(policy.begin(ids), (84, 3))
        self.assertFalse(policy.needs_measurement())
        policy.record(policy.epoch, tuple(ids), 84, 35.0, np.full(14, 4))
        self.assertEqual(policy.begin(ids), (84, 3))
        self.assertFalse(policy.needs_measurement())

    def test_dense_only_trains_and_changed_cohort_rejects_late_samples(self):
        policy = self.make_policy()
        ids = list(range(14))
        policy.begin(ids)
        epoch = policy.epoch
        accepted = np.arange(14) % 6
        policy.record(epoch, tuple(ids), 84, 35.0, accepted)
        before = policy.confidence(ids)
        policy.record(epoch, tuple(ids), 76, 32.0, np.zeros(14, dtype=np.int32))
        np.testing.assert_array_equal(policy.confidence(ids), before)
        self.assertGreater(np.prod(before[5]), np.prod(before[0]))
        replacement = [*ids[1:], 14]
        self.assertEqual(policy.begin(replacement), (84, 1))
        policy.record(epoch, tuple(ids), 84, 35.0, accepted)
        self.assertIsNone(policy.confidence(replacement))

    def test_probe_waits_for_completed_dense_reference(self):
        policy = self.make_policy()
        ids = list(range(14))
        for _ in range(8):
            self.assertEqual(policy.begin(ids), (84, 1))
        for _ in range(3):
            self.assertEqual(policy.begin(ids), (84, 1))
            policy.record(policy.epoch, tuple(ids), 84, 35.0, np.full(14, 4))
        self.assertEqual(policy.begin(ids), (76, 1))

    def test_no_valid_feedback_retires_calibration(self):
        ids = list(range(14))
        no_dense = self.make_policy()
        for _ in range(32):
            self.assertEqual(no_dense.begin(ids)[0], 84)
        self.assertFalse(no_dense.needs_measurement())

        missing_probe = self.make_policy()
        for _ in range(3):
            missing_probe.begin(ids)
            missing_probe.record(
                missing_probe.epoch, tuple(ids), 84, 35.0, np.full(14, 4)
            )
        for _ in range(48):
            missing_probe.begin(ids)
        self.assertFalse(missing_probe.needs_measurement())

    def test_grammar_caps_are_charged_to_throughput_and_omitted_from_tail(self):
        policy = self.make_policy()
        ids = list(range(14))
        policy.begin(ids)
        effective_caps = np.full(14, 5)
        effective_caps[0] = 3
        policy.record(
            policy.epoch,
            tuple(ids),
            84,
            35.0,
            np.full(14, 3),
            effective_caps,
        )
        self.assertEqual(list(policy.samples[84]), [(35.0, 56)])
        self.assertNotIn(0, policy.accepted)
        self.assertEqual(policy.accepted[1][3], 1)

    def test_rejoining_same_cohort_still_rejects_prior_epoch(self):
        policy = self.make_policy()
        ids = list(range(14))
        policy.begin(ids)
        old_epoch = policy.epoch
        policy.begin(ids[1:])
        policy.begin(ids)
        policy.record(old_epoch, tuple(ids), 84, 35.0, np.full(14, 4))
        self.assertEqual(policy.samples, {})
        self.assertIsNone(policy.confidence(ids))

    def test_no_probe_if_all_near_full_rows_replay_dense_graph(self):
        policy = _MeasuredEpPolicy(5, np.array([0] + [12] * 12))
        self.assertEqual(policy.begin([10, 20]), (12, 3))
        self.assertEqual(policy.begin([]), (0, 3))

    def test_no_cheaper_bucket_skips_timing_and_device_validation(self):
        with (
            patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}),
            patch.object(torch.cuda, "Stream"),
            patch.object(torch.cuda, "Event") as event,
        ):
            session = AdaptiveVerificationSession(2, 5, "cpu", [1, 2, 4, 12], ep_size=4)
            with (
                patch.object(
                    session.controller,
                    "_device_indices",
                    side_effect=lambda values, dtype=torch.int64: torch.tensor(
                        list(values), dtype=dtype
                    ),
                ),
                patch.object(session.controller, "observe") as legacy_observe,
                patch.object(session, "_queue_ep_feedback") as queue,
            ):
                session.begin([10, 20])
                caps, _, total = session.plan()
                self.assertEqual((caps.tolist(), total), ([5, 5], 12))
                session.observe(torch.tensor([3, 1]), [False, False])
                session.finish()
            self.assertIsNone(session.controller._pending)
            event.assert_not_called()
            legacy_observe.assert_not_called()
            queue.assert_not_called()

    def test_feedback_bounds_are_checked_only_after_event_completion(self):
        session = object.__new__(AdaptiveVerificationSession)
        session._ep_policy = Mock()
        session.event_pool = []
        session._ep_buffers = []
        events = [Mock(), Mock()]
        events[1].query.side_effect = [False, True]
        planned = np.array([3, 5], dtype=np.int32)
        # Request 10 accepts more than its planned cap. Request 20's grammar
        # cap exceeds its planned cap even though its accepted count is valid.
        host = torch.tensor([[4, 2], [3, 6]])
        session._ep_pending = [(1, (10, 20), 10, events, 35.0, host, planned, False, 0)]
        with patch("logging.error") as error:
            session._collect_ep_samples()
            error.assert_not_called()
            self.assertEqual(len(session._ep_pending), 1)
            session._collect_ep_samples()
            error.assert_called_once()
        session._ep_policy.record.assert_not_called()
        self.assertEqual(session._ep_pending, [])
        self.assertEqual(session._ep_buffers, [host])


class TailRiskRowPriceTest(unittest.TestCase):
    def test_decayed_censored_history_matches_full_recompute(self):
        decay = 0.5 ** (1 / 8)
        prior = (0.7, 0.8, 0.9, 0.9, 0.9)
        history = RollingAcceptance(prior, observations=32, prior_weight=1, decay=decay)
        outcomes = [(0, 5), (1, 1), (4, 5), (3, 3), (5, 5), (0, 0)] * 20
        for end, entry in enumerate(outcomes, 1):
            history.observe(*entry)
            window = outcomes[max(0, end - 32) : end]
            expected = 1.0
            actual_probs = history.survival()
            for hop, probability in enumerate(prior):
                chances = successes = 0.0
                for age, (accepted, cap) in enumerate(reversed(window)):
                    if hop < min(cap, accepted + 1):
                        weight = decay**age
                        chances += weight
                        if hop < accepted:
                            successes += weight
                expected *= (successes + probability) / (chances + 1)
                self.assertAlmostEqual(expected, actual_probs[hop], places=12)

    def test_price_keeps_long_prefixes_and_only_uses_cheaper_graphs(self):
        execution = np.array([0] + [9] * 9 + [12] * 3)
        policy = TailRiskEpPolicy(5, execution, row_price=0.4)
        ids = [12, 9]
        self.assertEqual(policy.begin(ids), (12, 1))
        for _ in range(16):
            policy.record_acceptance(policy.epoch, ids, 12, [0, 5], [5, 5])
        self.assertEqual(policy.begin(ids[::-1]), (8, 2))
        self.assertEqual(policy.capacities, (5, 1))
        # An unseen request stays full. Do not prune just to pad back to twelve.
        self.assertEqual(policy.begin([9, 11]), (12, 3))
        self.assertEqual(policy.capacities, (5, 5))
        policy.record_acceptance(policy.epoch, [9, 11], 12, [5, 2], [5, 5])
        self.assertEqual(policy.begin([9, 11]), (12, 3))
        self.assertEqual(policy.capacities, (5, 5))

    @staticmethod
    def execution_rows(*buckets):
        return [0] + [next(b for b in buckets if b >= n) for n in range(1, 85)]

    def test_bucket_bid_fills_selected_graph_with_actual_prefixes(self):
        execution = self.execution_rows(48, 56, 64, 72, 84)
        ids = list(range(14))
        for accepted, width in ((0, 56), (5, 64)):
            with self.subTest(accepted=accepted):
                policy = TailRiskEpPolicy(
                    5, execution, row_price=0.345, bucket_bid_price=0.2
                )
                for _ in range(16):
                    self.assertEqual(policy.begin(ids), (84, 1))
                    policy.record_acceptance(policy.epoch, ids, 84, [accepted] * 14)
                self.assertEqual(policy.begin(ids), (width, 2))
                self.assertEqual(len(ids) + sum(policy.capacities), width)
                self.assertTrue(all(1 <= cap <= 5 for cap in policy.capacities))
                self.assertEqual(execution[width], width)
                self.assertEqual(len(policy.margin_prior), 14)
                for _ in range(18, 32):
                    self.assertEqual(policy.begin(ids)[0], width)
                self.assertEqual(policy.begin(ids), (84, 1))
                self.assertIsNone(policy.margin_prior)

    def test_bucket_bid_waits_for_new_request_history_and_keeps_ids_in_order(self):
        policy = TailRiskEpPolicy(
            5,
            self.execution_rows(56, 64, 84),
            row_price=0.345,
            bucket_bid_price=0.2,
        )
        ids = list(range(14))
        self.assertEqual(policy.begin(ids), (84, 1))
        for _ in range(16):
            policy.record_acceptance(policy.epoch, ids, 84, [0] * 7 + [5] * 7)
        self.assertEqual(policy.begin(ids), (56, 2))
        ordered = dict(zip(ids, policy.capacities))
        self.assertGreater(ordered[13], ordered[0])
        # Different request order is valid. Distinct probabilities keep the two
        # groups separate while tied bids may distribute caps differently.
        self.assertEqual(policy.begin(ids[::-1]), (56, 2))
        self.assertGreater(policy.capacities[0], policy.capacities[-1])
        stale = policy.epoch
        changed = ids[1:] + [99]
        self.assertEqual(policy.begin(changed), (84, 1))
        for _ in range(16):
            policy.record_acceptance(stale, changed, 84, [0] * 14)
        self.assertEqual(policy.begin(changed), (84, 1))
        self.assertIsNone(policy.margin_prior)

    def test_bucket_bid_falls_back_to_existing_price_without_matching_graphs(self):
        for buckets, batch in (
            ((48, 64, 72, 84), 14),
            ((48, 56, 72, 84), 14),
            ((48, 56, 64, 72, 80, 84), 13),
        ):
            with self.subTest(buckets=buckets, batch=batch):
                execution = self.execution_rows(*buckets)
                plain = TailRiskEpPolicy(5, execution, row_price=0.345)
                opted_in = TailRiskEpPolicy(
                    5, execution, row_price=0.345, bucket_bid_price=0.2
                )
                ids = list(range(batch))
                dense = batch * 6
                for _ in range(17):
                    self.assertEqual(opted_in.begin(ids), plain.begin(ids))
                    self.assertEqual(opted_in.capacities, plain.capacities)
                    self.assertEqual(opted_in.margin_prior, plain.margin_prior)
                    for policy in (plain, opted_in):
                        policy.record_acceptance(policy.epoch, ids, dense, [0] * batch)

    def test_bucket_bid_requires_valid_row_price(self):
        execution = self.execution_rows(56, 64, 84)
        for bid in (0, 1, float("nan"), float("inf")):
            with (
                self.subTest(bid=bid),
                self.assertRaisesRegex(ValueError, "bucket bid"),
            ):
                TailRiskEpPolicy(5, execution, row_price=0.345, bucket_bid_price=bid)
        with self.assertRaisesRegex(ValueError, "bucket bid"):
            TailRiskEpPolicy(5, execution, bucket_bid_price=0.2)

    def test_bucket_row_quota_charges_dense_references_after_initial_history(self):
        ids = list(range(14))
        execution = self.execution_rows(48, 56, 64, 72, 84)
        policy = TailRiskEpPolicy(
            5, execution, row_price=0.345, bucket_row_quota=49.125
        )
        for _ in range(16):
            self.assertEqual(policy.begin(ids), (84, 1))
            self.assertIsNone(policy._bucket_row_debt)
            policy.record_acceptance(policy.epoch, ids, 84, [0] * 14)
        completed = 0
        for phase in range(17, 66):
            rows, mode = policy.begin(ids)
            self.assertEqual(rows, len(ids) + sum(policy.capacities))
            self.assertEqual(execution[rows], rows)
            self.assertTrue(all(1 <= cap <= 5 for cap in policy.capacities))
            completed += rows
            self.assertEqual(policy._bucket_row_debt, completed - (phase - 16) * 49.125)
            if phase % 32 == 0:
                self.assertEqual((rows, mode), (84, 1))
                self.assertIsNone(policy.margin_prior)
            else:
                self.assertIn(rows, (48, 56, 64, 72))
                self.assertEqual(mode, 2)
                self.assertIsNotNone(policy.margin_prior)
            policy.record_acceptance(
                policy.epoch, ids, rows, [0] * 14, policy.capacities
            )
        changed_ids = ids[1:] + [99]
        self.assertEqual(policy.begin(changed_ids), (84, 1))
        self.assertIsNone(policy._bucket_row_debt)

    def test_bucket_row_quota_falls_back_if_exact_paid_shape_is_missing(self):
        execution = self.execution_rows(48, 56, 64, 76, 84)
        ids = list(range(14))
        plain = TailRiskEpPolicy(5, execution, row_price=0.345)
        quota = TailRiskEpPolicy(5, execution, row_price=0.345, bucket_row_quota=67.625)
        for _ in range(17):
            self.assertEqual(quota.begin(ids), plain.begin(ids))
            self.assertEqual(quota.capacities, plain.capacities)
            for policy in (plain, quota):
                policy.record_acceptance(policy.epoch, ids, 84, [0] * 14)
        self.assertIsNone(quota._bucket_row_debt)

    def test_bucket_row_quota_requires_valid_price_mode_and_feasible_rows(self):
        execution = self.execution_rows(48, 56, 64, 72, 84)
        for options in (
            {"bucket_row_quota": 67.625},
            {"row_price": 0.345, "bucket_row_quota": 67.625, "bucket_bid_price": 0.2},
            {"row_price": 0.345, "bucket_row_quota": 48},
            {"row_price": 0.345, "bucket_row_quota": 73},
            {"row_price": 0.345, "bucket_row_quota": float("nan")},
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, "bucket row quota"),
            ):
                TailRiskEpPolicy(5, execution, **options)


class TailRiskHostPlanTest(unittest.TestCase):
    def setup_session(self, *, tail_risk=True):
        with (
            patch.dict(
                os.environ,
                {
                    "RTP_LLM_ADAPTIVE_EP_TAIL_RISK": str(int(tail_risk)),
                    "RTP_LLM_ADAPTIVE_EP_TAIL_RISK_NEAR_FULL": "1",
                    "RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES": "0",
                    "RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1",
                },
            ),
            patch.object(torch.cuda, "Stream"),
        ):
            session = AdaptiveVerificationSession(
                14,
                5,
                "cpu",
                [1, 14, 82, 84],
                ep_size=4,
                allow_ep_current_scores=True,
            )
        indices = patch.object(
            session.controller,
            "_device_indices",
            side_effect=lambda values, dtype=torch.int64: torch.tensor(
                list(values), dtype=dtype
            ),
        )
        indices.start()
        self.addCleanup(indices.stop)
        return session

    def test_early_commit_keeps_caps_order_and_feedback(self):
        old, new = self.setup_session(), self.setup_session()
        count_narrow = 0
        with patch.object(torch.cuda, "Event"):
            for index in range(45):
                ids = list(range(14))
                if index % 2:
                    ids.reverse()
                accepted = torch.tensor([0 if i < 2 else 5 for i in ids])
                old.begin(ids)
                expected = old.plan()
                plan, _, realloc, scores, _, committed, margin = new.begin_for_executor(
                    ids
                )
                self.assertFalse(margin)
                self.assertTrue(committed)
                self.assertFalse(realloc or scores)
                self.assertEqual(expected[2], plan[2])
                self.assertTrue(torch.equal(expected[0], plan[0]))
                self.assertTrue(torch.equal(expected[1], plan[1]))
                count_narrow += plan[2] == 82
                for session in (old, new):
                    # Exercise commit/observe and inject already-completed
                    # feedback on CPU. Event ordering is covered separately.
                    session.events = None
                    if session is old:
                        session.observe(accepted, [False] * 14)
                    else:
                        session.observe_emitted_tokens(accepted + 1, [False] * 14)
                    session._ep_policy.record_acceptance(
                        session._ep_policy.epoch,
                        ids,
                        plan[2],
                        accepted.tolist(),
                        plan[0].tolist(),
                    )
        self.assertGreater(count_narrow, 0)
        self.assertLessEqual(len(new.controller._explicit_host_plans), 3)

    def test_current_prediction_policy_still_commits_at_plan(self):
        session = self.setup_session(tail_risk=False)
        with patch.object(torch.cuda, "Event"):
            plan, _, _, _, _, committed, margin = session.begin_for_executor(
                list(range(14))
            )
            self.assertFalse(margin)
            self.assertFalse(committed)
            self.assertIsNone(session.controller._pending)
            self.assertIs(session.plan()[0], plan[0])
            self.assertIsNotNone(session.controller._pending)

    def test_unfinished_early_plan_is_discarded(self):
        session = self.setup_session()
        with patch.object(torch.cuda, "Event"), patch("logging.warning") as warning:
            session.begin_for_executor(list(range(14)))
            session.begin_for_executor(list(range(1, 15)))
        warning.assert_called_once()
        self.assertEqual(session.controller._pending.request_ids, tuple(range(1, 15)))

    def test_bucket_bid_env_uses_completed_history_and_preserves_draft_rerank(self):
        env = {
            "RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1",
            "RTP_LLM_ADAPTIVE_EP_TAIL_RISK_NEAR_FULL": "0",
            "RTP_LLM_ADAPTIVE_EP_ROW_PRICE": "0.345",
            "RTP_LLM_ADAPTIVE_EP_ROW_PRICE_PRIOR": "0.7,0.8,0.9,0.9,0.9",
            "RTP_LLM_ADAPTIVE_EP_BUCKET_BID_PRICE": "0.2",
            "RTP_LLM_ADAPTIVE_EP_MARGIN_RERANK": "1",
            "RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(torch.cuda, "Stream"),
        ):
            session = AdaptiveVerificationSession(
                14,
                5,
                "cpu",
                [1, 14, 48, 56, 64, 72, 84],
                ep_size=4,
                allow_ep_current_scores=True,
            )
        self.assertTrue(session._ep_margin_rerank)
        policy = session._ep_policy
        ids = list(range(14))
        self.assertEqual(policy.begin(ids), (84, 1))
        for _ in range(16):
            policy.record_acceptance(policy.epoch, ids, 84, [0] * 14)
        self.assertEqual(policy.begin(ids), (56, 2))
        self.assertIsNotNone(policy.margin_prior)

    def test_bucket_row_quota_env_uses_actual_distributed_rank(self):
        env = {
            "RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1",
            "RTP_LLM_ADAPTIVE_EP_ROW_PRICE": "0.345",
            "RTP_LLM_ADAPTIVE_EP_BUCKET_ROW_QUOTAS": "67.625,67.625,49.125,67.625",
            "RTP_LLM_ADAPTIVE_EP_MARGIN_RERANK": "1",
            "WORLD_RANK": "0",  # Launcher metadata is not the source of rank.
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(torch.cuda, "Stream"),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=4),
            patch.object(torch.distributed, "get_rank", return_value=2),
        ):
            session = AdaptiveVerificationSession(
                14,
                5,
                "cpu",
                [1, 14, 48, 56, 64, 72, 84],
                ep_size=4,
                allow_ep_current_scores=True,
            )
            self.assertEqual(session._ep_policy._bucket_row_quota, 49.125)
            self.assertTrue(session._ep_margin_rerank)
            for bad in ("49,67,68,68", "67.625,49.125", "63,63,63,62", "nan,63,63,126"):
                with (
                    self.subTest(bad=bad),
                    patch.dict(
                        os.environ, {"RTP_LLM_ADAPTIVE_EP_BUCKET_ROW_QUOTAS": bad}
                    ),
                    self.assertRaisesRegex(ValueError, "four feasible EP row quotas"),
                ):
                    AdaptiveVerificationSession(
                        14,
                        5,
                        "cpu",
                        [1, 14, 48, 56, 64, 72, 84],
                        ep_size=4,
                        allow_ep_current_scores=True,
                    )

    def test_bucket_row_quota_requires_the_four_rank_group(self):
        env = {
            "RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1",
            "RTP_LLM_ADAPTIVE_EP_ROW_PRICE": "0.345",
            "RTP_LLM_ADAPTIVE_EP_BUCKET_ROW_QUOTAS": "49.125,67.625,67.625,67.625",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(torch.cuda, "Stream"),
            patch.object(torch.distributed, "is_initialized", return_value=True),
            patch.object(torch.distributed, "get_world_size", return_value=8),
        ):
            with self.assertRaisesRegex(ValueError, "four-rank TP1 EP group"):
                AdaptiveVerificationSession(
                    14,
                    5,
                    "cpu",
                    [1, 14, 48, 56, 64, 72, 84],
                    ep_size=4,
                    allow_ep_current_scores=True,
                )
            # Fixed verify caps still bypass experimental EP configuration.
            with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "5"}):
                session = AdaptiveVerificationSession(
                    14,
                    5,
                    "cpu",
                    [1, 14, 48, 56, 64, 72, 84],
                    ep_size=4,
                    allow_ep_current_scores=True,
                )
            self.assertIsNone(session._ep_policy)

    def test_four_gap_flag_selects_model_without_changing_host_paid_shape(self):
        common = {
            "RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1",
            "RTP_LLM_ADAPTIVE_EP_ROW_PRICE": "0.345",
            "RTP_LLM_ADAPTIVE_EP_MARGIN_RERANK": "1",
        }
        for env, use_four in (
            (common, False),
            ({**common, "RTP_LLM_ADAPTIVE_EP_MARGIN_FOUR_GAPS": "1"}, True),
            (
                {
                    **common,
                    "RTP_LLM_ADAPTIVE_EP_MARGIN_FOUR_GAPS": "1",
                    "RTP_LLM_ADAPTIVE_EP_MARGIN_RERANK": "0",
                },
                False,
            ),
        ):
            with (
                self.subTest(env=env),
                patch.dict(os.environ, env, clear=True),
                patch.object(torch.cuda, "Stream"),
            ):
                session = AdaptiveVerificationSession(
                    14,
                    5,
                    "cpu",
                    [1, 14, 48, 56, 64, 72, 84],
                    ep_size=4,
                    allow_ep_current_scores=True,
                )
                self.assertEqual(session._ep_margin_four_gaps, use_four)
                if not session._ep_margin_rerank:
                    continue
                ids = list(range(14))
                session.ids = ids
                previous_caps = torch.tensor([3] * 14, dtype=torch.int32)
                previous_rows = torch.arange(56)
                session._prepared = previous_caps, previous_rows, 56
                session.total = 56
                session._calibration = False
                session._ep_margin_prior = torch.full((14, 5), 0.5)
                gaps = torch.rand(4, 14)
                with (
                    patch.object(session.controller, "commit_prepared") as commit,
                    patch(
                        "rtp_llm.models_py.sampling.adaptive_session.pack_ep_margin_rows",
                        return_value=(previous_caps, previous_rows),
                    ) as pack,
                ):
                    caps, rows, total = session.plan(gaps)
                pack.assert_called_once_with(
                    gaps,
                    session._ep_margin_prior,
                    previous_caps,
                    56,
                    four_gaps=use_four,
                )
                commit.assert_called_once_with(ids, previous_caps)
                self.assertIs(caps, previous_caps)
                self.assertIs(rows, previous_rows)
                self.assertEqual(total, 56)

    def test_host_bonus_conversion_waits_for_feedback_event(self):
        session = self.setup_session()
        policy = Mock()
        session._ep_policy = policy
        events = [None, Mock()]
        events[1].query.side_effect = [False, True]
        planned = np.array([3, 5], dtype=np.int32)
        host = torch.tensor([[4, 2], [3, 5]])
        session._ep_pending = [(1, (10, 20), 10, events, 0.0, host, planned, False, 1)]
        session._collect_ep_samples()
        policy.record_acceptance.assert_not_called()
        session._collect_ep_samples()
        np.testing.assert_array_equal(
            policy.record_acceptance.call_args.args[3], [3, 1]
        )
        np.testing.assert_array_equal(
            policy.record_acceptance.call_args.args[4], [3, 5]
        )
        self.assertEqual(host[0].tolist(), [4, 2])


class EpVerifyRowsPerRequestTest(unittest.TestCase):
    ENV = "RTP_LLM_ADAPTIVE_EP_VERIFY_ROWS_PER_REQUEST"
    GRID = [
        1,
        2,
        4,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
    ]

    def create_session(
        self, *, value="4.5", env=None, steps=5, rows=None, ep_size=8, allowed=True
    ):
        flags = {} if value is None else {self.ENV: value}
        if rows is None:
            maximum = 64 * (steps + 1)
            rows = [bucket for bucket in self.GRID if bucket < maximum] + [maximum]
        with (
            patch.dict(os.environ, {**flags, **(env or {})}, clear=True),
            patch.object(torch.cuda, "Stream"),
        ):
            session = AdaptiveVerificationSession(
                64,
                steps,
                "cpu",
                rows,
                ep_size=ep_size,
                allow_ep_current_scores=allowed,
            )
        indices = patch.object(
            session.controller,
            "_device_indices",
            side_effect=lambda values, dtype=torch.int64: torch.tensor(
                list(values), dtype=dtype
            ),
        )
        indices.start()
        self.addCleanup(indices.stop)
        return session

    def prepare(self, session, ids, *, age=0, missing=None):
        for request_id in ids:
            if request_id not in session.registered:
                session.controller.register(request_id)
                session.registered.add(request_id)
            if request_id != missing:
                session._host_predictions[request_id] = (
                    session.round - age,
                    np.full(session.steps, 0.8),
                )
        return session.begin_for_executor(ids)

    def test_unset_uses_dynamic_cap_and_default_budget_on_supported_layout(self):
        session = self.create_session(value=None)
        self.assertEqual(session.fixed_cap, -1)
        self.assertEqual(session._ep_verify_rows_per_request, 4.5)
        self.assertIsNone(session._ep_policy)
        self.assertTrue(session.needs_prediction())
        plan, _, reallocate, _, _, _, _ = self.prepare(session, list(range(14)))
        self.assertTrue(reallocate)
        self.assertEqual(plan[2], 64)

    def test_unset_does_not_override_other_layouts_or_explicit_policies(self):
        alternatives = (
            {"ep_size": 1},
            {"steps": 4, "rows": [1, 320]},
            {"rows": []},
            {"allowed": False},
            {"env": {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}},
            {"env": {"RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1"}},
            {"env": {"RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES": "1"}},
        )
        for options in alternatives:
            with self.subTest(options=options):
                session = self.create_session(value=None, **options)
                self.assertIsNone(session._ep_verify_rows_per_request)
        fixed = self.create_session(
            value=None, env={"RTP_LLM_ADAPTIVE_VERIFY_CAP": "3"}
        )
        self.assertEqual(fixed.fixed_cap, 3)
        self.assertFalse(fixed.needs_prediction())

    def test_unset_preserves_measured_policy_for_other_draft_widths(self):
        for steps in (7, 9):
            with self.subTest(steps=steps):
                session = self.create_session(value=None, steps=steps)
                self.assertIsNone(session._ep_verify_rows_per_request)
                self.assertIsInstance(session._ep_policy, _MeasuredEpPolicy)
                self.assertFalse(session.needs_prediction())

    def test_validates_number_topology_and_conflicts(self):
        for bad in ("", "bad", "nan", "inf", "0.75", "6.25"):
            with self.subTest(value=bad), self.assertRaisesRegex(ValueError, self.ENV):
                self.create_session(value=bad)
        for steps in (7, 9):
            with (
                self.subTest(steps=steps),
                self.assertRaisesRegex(ValueError, self.ENV),
            ):
                self.create_session(value=str(steps + 1.25), steps=steps)
        unsupported = (
            {"ep_size": 1},
            {"value": "1", "steps": 0},
            {"rows": []},
            {"allowed": False},
            {"env": {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "5"}},
            {"env": {"RTP_LLM_ADAPTIVE_EP_TAIL_RISK": "1"}},
            {"env": {"RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES": "1"}},
        )
        for options in unsupported:
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, self.ENV),
            ):
                self.create_session(**options)

    def test_configured_widths_use_current_prefix_allocator_at_same_budget(self):
        ids = list(range(14))
        for steps, value, expected in ((5, "4.5", 64), (7, "6.4", 96), (9, "8", 112)):
            with self.subTest(steps=steps, value=value):
                session = self.create_session(value=value, steps=steps)
                width = steps + 1
                self.assertIsNone(session._ep_policy)
                self.assertTrue(session.needs_prediction())
                cold, _, current, _, metrics, _, _ = session.begin_for_executor(ids)
                self.assertEqual(cold[2], len(ids) * width)
                self.assertEqual(metrics[0], 3)
                self.assertFalse(current)

                plan, cached, current, _, metrics, committed, _ = self.prepare(
                    session, ids, age=16
                )
                caps, rows, total = plan
                self.assertEqual(total, expected)
                self.assertEqual(session._cost_key(total), total)
                self.assertEqual(int(caps.sum()) + len(ids), total)
                self.assertTrue(torch.all((caps >= 0) & (caps <= steps)))
                positions = torch.arange(width)
                dense = torch.arange(len(ids) * width).reshape(len(ids), width)
                torch.testing.assert_close(rows, dense[positions <= caps[:, None]])
                self.assertTrue(current)
                self.assertFalse(cached or committed)
                self.assertEqual(metrics[0], 2)

                moved = caps.clone()
                moved[0] += 1
                moved[-1] -= 1
                reranked = moved, dense[positions <= moved[:, None]], total
                with patch.object(
                    session.controller, "reallocate_current", return_value=reranked
                ) as allocate:
                    current_caps, current_rows, current_total = session.plan()
                    self.assertIs(current_caps, reranked[0])
                    self.assertIs(current_rows, reranked[1])
                    self.assertEqual(current_total, expected)
                    allocate.assert_called_once_with(ids, expected)
                self.assertEqual(session.controller._pending.request_ids, tuple(ids))
                self.assertIs(session.controller._pending.capacities, moved)

    def test_ep_steady_shape_reuses_host_plan_across_prediction_order(self):
        session = self.create_session()
        ids = list(range(14))
        first, _, reallocate, _, _, _, _ = self.prepare(session, ids)
        self.assertTrue(reallocate)
        cached = session.controller.host_capacities
        for request_id in ids:
            session._host_predictions[request_id] = (
                session.round,
                np.full(session.steps, 0.45 + request_id / 50),
            )
        with patch.object(
            session.controller, "_assign_host_budget", side_effect=AssertionError
        ):
            second, _, reallocate, _, _, _, _ = session.begin_for_executor(ids)
        self.assertTrue(reallocate)
        self.assertIs(session.controller.host_capacities, cached)
        self.assertEqual(first[2], second[2])
        self.assertEqual(second[2], 64)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])

    def test_ep_one_row_budget_commits_and_observes_zero_caps(self):
        session = self.create_session(value="1")
        ids = list(range(14))
        _, _, reallocate, _, metrics, _, _ = self.prepare(session, ids)
        self.assertFalse(reallocate)
        self.assertEqual(metrics[:3], (2, 14, 0))
        caps, rows, total = session.plan()
        self.assertEqual(total, 14)
        torch.testing.assert_close(caps, torch.zeros(14, dtype=torch.int32))
        torch.testing.assert_close(rows, torch.arange(0, 84, 6))
        session.observe(torch.zeros_like(caps), [False] * len(ids))
        self.assertIsNone(session.controller._pending)

    def test_ep_empty_batch_keeps_empty_plan(self):
        session = self.create_session()
        prepared = session.begin_for_executor([])
        (caps, rows, total), _, reallocate, _, metrics, _, _ = prepared
        self.assertEqual((caps.numel(), rows.numel(), total), (0, 0, 0))
        self.assertFalse(reallocate)
        self.assertEqual(metrics[0], 3)

    def test_bucket_uses_actual_batch_and_favors_high_ties_without_padding(self):
        for batch, expected in ((10, 48), (14, 64), (16, 80)):
            with self.subTest(batch=batch):
                session = self.create_session()
                plan, _, _, _, metrics, _, _ = self.prepare(session, list(range(batch)))
                self.assertEqual(plan[2], expected)
                self.assertEqual(session._cost_key(expected), expected)
                self.assertLess(expected, session._cost_key(batch * 6))
                self.assertEqual(metrics[0], 2)

    def test_full_width_competes_with_reduced_buckets(self):
        for steps, batch, requested, expected in (
            (5, 14, "6", 84),
            (5, 14, "6.0", 84),
            (5, 14, "5.9", 84),
            (5, 10, "5.8", 60),
            (5, 14, "5.7", 80),
            (7, 14, "8", 112),
            (9, 14, "10", 140),
        ):
            with self.subTest(steps=steps, batch=batch, requested=requested):
                session = self.create_session(value=requested, steps=steps)
                plan, cached, current, _, metrics, _, _ = self.prepare(
                    session, list(range(batch))
                )
                cutting = expected < batch * (steps + 1)
                self.assertEqual(plan[2], expected)
                self.assertEqual(metrics[0], 2 if cutting else 3)
                self.assertEqual(current, cutting)
                self.assertFalse(cached)

    def test_cold_stale_future_and_no_smaller_bucket_stay_full(self):
        cases = (
            (None, list(range(14)), 0, 5),
            (None, list(range(14)), 17, None),
            (None, list(range(14)), -1, None),
            ([1, 384], list(range(14)), 0, None),
            (None, [], 0, None),
        )
        for grid, ids, age, missing in cases:
            with self.subTest(grid=grid, batch=len(ids), age=age, missing=missing):
                session = self.create_session(rows=grid)
                plan, cached, current, _, metrics, _, _ = self.prepare(
                    session, ids, age=age, missing=missing
                )
                self.assertEqual(plan[2], len(ids) * 6)
                self.assertEqual(metrics[0], 3)
                self.assertFalse(cached or current)
                self.assertTrue(session.needs_prediction())

    def test_only_complete_prior_proposals_are_queued_after_observe(self):
        session = self.create_session()
        ids = list(range(14))
        self.prepare(session, ids, missing=5)
        session.controller._ready[: len(ids), :] = True
        session.controller._ready[5, 0] = False
        session.plan()
        with (
            patch.object(session, "_queue_predictions") as enqueue,
            patch.object(session.controller.estimator, "step"),
        ):
            session.observe(torch.zeros(14, dtype=torch.int32), [False] * 14)
        enqueue.assert_called_once_with(tuple(i for i in ids if i != 5))
        self.assertFalse(session.controller._ready.any())
        self.assertEqual(session.round, 1)

    def test_late_copy_from_released_request_epoch_cannot_enable_cut(self):
        session = self.create_session()
        ids = list(range(14))
        session.begin_for_executor(ids)
        self.prepare(session, ids)
        prior_epoch = session._request_epochs[5]
        session.begin_for_executor([i for i in ids if i != 5])
        session._pending_predictions.append(
            (
                (5,),
                (prior_epoch,),
                session.round,
                torch.ones((1, 5)),
                Mock(query=Mock(return_value=True)),
            )
        )
        plan, _, current, _, metrics, _, _ = session.begin_for_executor(ids)
        self.assertNotEqual(session._request_epochs[5], prior_epoch)
        self.assertNotIn(5, session._host_predictions)
        self.assertEqual(plan[2], 84)
        self.assertEqual(metrics[0], 3)
        self.assertFalse(current)


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
        session.round_costs[2] = {2: 1.0, 3: 1.5, 4: 2.0, 5: 2.5, 6: 3.0}
        session._prepare_plan()
        self.assertEqual(session.metrics_snapshot()[:3], (3, 0, 2))
        session._host_predictions = {
            i: (session.round, np.array([0.9, 0.8])) for i in (10, 20)
        }
        session._prepare_plan()
        self.assertEqual(session.metrics_snapshot()[0], 2)
        session.begin([])
        self.assertEqual(session.metrics_snapshot()[1:3], (0, 0))

    def test_partial_predictions_do_not_force_whole_batch_full_width(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda")
        session.begin([10, 20])
        session.round = 20
        session.round_costs[2] = {2: 2.0, 3: 2.2, 4: 2.4, 5: 3.0, 6: 4.0}
        for missing in (None, (3, np.array([0.1, 0.1])), (21, np.array([0.1, 0.1]))):
            with self.subTest(missing=missing):
                session._host_predictions = {10: (20, np.array([0.01, 0.01]))}
                if missing is not None:
                    session._host_predictions[20] = missing
                session._prepare_plan()
                self.assertEqual(session.metrics_snapshot()[0], 2)
                self.assertEqual(session.prepared_plan()[0].tolist(), [0, 2])
                self.assertEqual(session.prepared_plan()[2], 4)

    def test_five_step_request_churn_does_not_trigger_batch_wide_fallback(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(
                8, 5, "cuda", [1, 2, 4, 8, 12, 16, 24, 32, 48]
            )
        session.seed_cost_curves(
            [(n, float(n)) for n in [1, 2, 4, 8, 12, 16, 24, 32, 48]],
            [(n, 0.1) for n in [1, 2, 4, 8]],
        )
        modes = []
        for turn in range(192):
            # Keep established requests alongside one newly admitted request;
            # alternate batch sizes and order, beyond the cost-aging horizon.
            batch = (3, 8)[turn % 2]
            ids = list(range(batch - 1)) + [1000 + turn]
            if turn % 3 == 0:
                ids.reverse()
            session.begin(ids)
            session.round = turn
            session._host_predictions = {
                request_id: (turn, np.full(5, 0.05))
                for request_id in ids
                if request_id < 1000
            }
            session._prepare_plan()
            caps, rows, total = session.prepared_plan()
            modes.append(session.metrics_snapshot()[0])
            self.assertFalse(session._calibration)
            self.assertLess(total, batch * 6)
            counts = caps.tolist()
            self.assertEqual(total, batch + sum(counts))
            self.assertEqual(
                rows.tolist(),
                [i * 6 + j for i, count in enumerate(counts) for j in range(count + 1)],
            )
        self.assertEqual(set(modes), {2})
        # Genuine all-missing predictions retain the explicit safety fallback.
        session._host_predictions.clear()
        session._prepare_plan()
        self.assertEqual(session.metrics_snapshot()[0], 3)
        self.assertEqual(session.total, len(session.ids) * 6)

    def test_current_allocation_reverses_stale_ranking_at_online_batch_sizes(
        self,
    ) -> None:
        for batch in (16, 24, 30):
            with self.subTest(batch=batch):
                controller = AdaptiveDraftController(batch, 5, torch.device("cuda"))
                ids = list(range(100, 100 + batch))[::-1]
                for request_id in ids:
                    controller.register(request_id)
                slots = controller._slots(ids)
                stale = np.array(
                    [[0.05] * 5] * (batch // 2) + [[0.99] * 5] * (batch // 2)
                )
                current = torch.tensor(
                    stale[::-1].copy(), dtype=torch.float32, device="cuda"
                )
                controller.confidence_probs[list(slots)] = current
                controller._ready[list(slots)] = True
                with (
                    patch.object(
                        torch.Tensor, "cpu", side_effect=AssertionError("GPU read")
                    ),
                    patch.object(
                        torch.Tensor, "item", side_effect=AssertionError("GPU read")
                    ),
                ):
                    caps, rows, total = controller.reallocate_current(ids, batch * 4)
                expected_caps = [5] * (batch // 2) + [1] * (batch // 2)
                self.assertEqual(caps.tolist(), expected_caps)
                self.assertEqual(total, batch * 4)
                self.assertEqual(
                    rows.tolist(),
                    [
                        i * 6 + j
                        for i, c in enumerate(expected_caps)
                        for j in range(c + 1)
                    ],
                )
                # The compacted target writes exactly these valid KV positions;
                # no suffix/padding position may leak into output or feedback.
                dense = torch.arange(batch * 6, device="cuda")
                compact = dense.index_select(0, rows)
                restored = torch.full_like(dense, -1).index_copy_(0, rows, compact)
                self.assertEqual(int((restored != -1).sum()), total)
                controller.commit_prepared(ids, caps)
                self.assertEqual(controller._pending.capacities.tolist(), expected_caps)
                controller.observe(ids, caps, [False] * batch)

    def test_current_allocation_incomplete_nonfinite_ties_and_boundary_budgets(
        self,
    ) -> None:
        controller = AdaptiveDraftController(4, 5, torch.device("cuda"))
        ids = [40, 10, 30, 20]
        for request_id in ids:
            controller.register(request_id)
        slots = controller._slots(ids)
        controller.confidence_probs[list(slots)] = 0.5
        controller._ready[list(slots)] = True
        # Stable ties must take request prefixes, including CAP=0/full edges.
        for total in range(4, 25):
            caps, rows, count = controller.reallocate_current(ids, total)
            counts = caps.tolist()
            self.assertEqual(sum(counts) + 4, total)
            self.assertTrue(all(0 <= n <= 5 for n in counts))
            self.assertEqual(
                rows.tolist(),
                [i * 6 + j for i, n in enumerate(counts) for j in range(n + 1)],
            )
        controller.confidence_probs[slots[0]] = float("nan")
        controller._ready[slots[1], 0] = False
        caps, rows, _ = controller.reallocate_current(ids, 14)
        self.assertEqual(caps.tolist(), [5, 5, 0, 0])
        for total in (3, 25):
            with self.assertRaises(AdaptiveBudgetInputError):
                controller.reallocate_current(ids, total)

    def test_session_reallocates_only_learned_partial_budget(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 5, "cuda")
        session.begin([10, 20])
        slots = session.controller._slots([10, 20])
        session.controller._ready[list(slots)] = True
        session.controller.confidence_probs[list(slots)] = torch.tensor(
            [[0.99] * 5, [0.01] * 5], device="cuda"
        )
        caps = torch.tensor([1, 5], dtype=torch.int32, device="cuda")
        rows = torch.tensor([0, 1, 6, 7, 8, 9, 10, 11], device="cuda")
        session._prepared = caps, rows, 8
        session.total = 8
        session._metrics = (2, 0, 1, 0.0)
        self.assertTrue(session.needs_current_allocation())
        actual, _, total = session.plan()
        self.assertEqual(actual.tolist(), [5, 1])
        self.assertEqual(total, 8)
        self.assertEqual(session.controller._pending.capacities.tolist(), [5, 1])
        for mode in (0, 1, 3):
            session._metrics = (mode, 0, 1, 0.0)
            self.assertFalse(session.needs_current_allocation())
        session._metrics = (2, 0, 0, 0.0)
        for total in (2, 12):
            session.total = total
            self.assertFalse(session.needs_current_allocation())

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
        session.round_costs[2] = {2: 1.0, 3: 1.25, 4: 1.5, 5: 1.75, 6: 2.0}
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

    def test_calibration_visits_buckets_between_uniform_request_depths(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda", [1, 2, 3, 4, 5, 6])
        session.begin([10, 20])
        session.round_costs[2] = {2: 12.0, 4: 15.0, 6: 20.0}
        session._host_predictions = {i: (0, np.array([0.9, 0.9])) for i in (10, 20)}
        session._prepare_plan()
        self.assertTrue(session._calibration)
        self.assertEqual(session.prepared_plan()[2], 5)
        self.assertEqual(session.prepared_plan()[0].tolist(), [2, 1])
        session.round_costs[2][5] = 16.0
        session._prepare_plan()
        self.assertEqual(session.prepared_plan()[2], 3)
        session.round_costs[2][3] = 14.0
        session._prepare_plan()
        self.assertFalse(session._calibration)
        # The measured intermediate bucket wins; the old uniform-only samples
        # priced five rows at 20ms and incorrectly selected all six rows.
        self.assertEqual(session.prepared_plan()[2], 5)

    def test_profile_seeding_covers_new_batches_and_preserves_measured_costs(
        self,
    ) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda", [1, 2, 4, 6])
        session.seed_cost_curves(
            [(1, 1.0), (2, 2.0), (4, 4.0), (6, 6.0)], [(1, 0.5), (2, 1.0)]
        )
        session.begin([10, 20])
        self.assertFalse(session._calibration)
        self.assertEqual(session._decision_costs(2), {2: 4.0, 4: 6.0, 6: 8.0})
        events = [Mock(spec=torch.cuda.Event), Mock(spec=torch.cuda.Event)]
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 10.0
        session.pending_costs = [(2, 4, events, 8.0)]
        session._collect_costs()
        for key, expected in {2: 20 / 3, 4: 10.0, 6: 40 / 3}.items():
            self.assertAlmostEqual(session._decision_costs(2)[key], expected)
        session.begin([10])
        self.assertFalse(session._calibration)
        self.assertEqual(session._decision_costs(1), {1: 2.0, 2: 3.0, 4: 5.0})
        events[1].synchronize.assert_not_called()

    def test_partial_draft_profile_only_seeds_covered_batches(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda", [1, 2, 4, 6])
        session.seed_cost_curves([(1, 1.0), (2, 2.0), (4, 4.0), (6, 6.0)], [(1, 0.5)])
        session.begin([10])
        self.assertFalse(session._calibration)
        session.begin([10, 20])
        self.assertTrue(session._calibration)
        self.assertEqual(session._decision_costs(2), {})
        events = [Mock(spec=torch.cuda.Event), Mock(spec=torch.cuda.Event)]
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 3.0
        session.pending_costs = [(2, 6, events, 2.0)]
        session._collect_costs()
        self.assertEqual(session._decision_costs(2), {6: 3.0})

    def test_shorter_live_context_does_not_lock_seeded_curve_at_full_width(
        self,
    ) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(1, 2, "cuda", [1, 2, 3])
        session.seed_cost_curves([(1, 10.0), (2, 20.0), (3, 30.0)], [(1, 0.5)])
        session.begin([10])
        events = [Mock(spec=torch.cuda.Event), Mock(spec=torch.cuda.Event)]
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 3.0
        session.pending_costs = [(1, 3, events, 2.0)]
        session._collect_costs()
        session._host_predictions = {10: (session.round, np.array([0.1, 0.1]))}
        session._prepare_plan()
        self.assertEqual(session.prepared_plan()[0].tolist(), [0])
        self.assertLess(session._decision_costs(1)[1], 2.0)

    def test_stale_expensive_bucket_recovers_without_forcing_small_budget(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(1, 2, "cuda", [1, 2, 3])
        session.seed_cost_curves([(1, 1.0), (2, 2.0), (3, 3.0)], [(1, 0.5)])
        session.begin([10])
        session.round_costs[1] = {1: 100.0, 2: 100.0, 3: 4.0}
        session._cost_sample_round = {(1, n): session.round for n in (1, 2, 3)}
        session._host_predictions = {10: (session.round, np.array([0.1, 0.1]))}
        session._prepare_plan()
        self.assertEqual(session.total, 3)
        session.round += session._COST_MAX_AGE // 2
        self.assertAlmostEqual(session._decision_costs(1)[1], 51.0)
        session.round += session._COST_MAX_AGE // 2
        session._host_predictions = {10: (session.round, np.array([0.1, 0.1]))}
        session._prepare_plan()
        self.assertEqual(session.total, 1)
        self.assertFalse(session._calibration)
        # High confidence still selects full width; aging is not a cap heuristic.
        session._host_predictions = {10: (session.round, np.ones(2))}
        session._prepare_plan()
        self.assertEqual(session.total, 3)
        # A new observation immediately restores the measured cost, without waiting.
        events = [Mock(spec=torch.cuda.Event), Mock(spec=torch.cuda.Event)]
        events[1].query.return_value = True
        events[0].elapsed_time.return_value = 8.0
        session.pending_costs = [(1, 3, events, 7.0)]
        session._collect_costs()
        self.assertEqual(session._cost_sample_round[(1, 3)], session.round)
        self.assertAlmostEqual(session._decision_costs(1)[3], 8.0)
        events[1].synchronize.assert_not_called()

    def test_cost_aging_does_not_invent_unprofiled_costs(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda", [1, 2, 4, 6])
        session.seed_cost_curves([(1, 1.0), (2, 2.0), (4, 4.0), (6, 6.0)], [(1, 0.5)])
        session.round_costs[2] = {6: 12.0}
        session._cost_sample_round[(2, 6)] = 0
        session.round = 10000
        self.assertEqual(session._decision_costs(2), {6: 12.0})
        self.assertEqual(session._decision_costs(1), {1: 2.0, 2: 3.0, 4: 5.0})

    def test_invalid_profile_curves_fail_before_replacing_valid_state(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 2, "cuda", [1, 2, 4, 6])
        good = [(1, 1.0), (2, 2.0), (4, 4.0), (6, 6.0)]
        session.seed_cost_curves(good, [(2, 1.0)])
        for target, draft in [
            (good[:-1], [(2, 1.0)]),
            (good + [(6, 7.0)], [(2, 1.0)]),
            (good, []),
            (good, [(2, float("nan"))]),
            (good, [(2, 0.0)]),
            (good, [(True, 1.0)]),
        ]:
            with (
                self.subTest(target=target, draft=draft),
                self.assertRaises(ValueError),
            ):
                session.seed_cost_curves(target, draft)
        self.assertEqual(session._profile_target, dict(good))
        self.assertEqual(session._profile_draft, {2: 1.0})

    def test_eager_calibration_stays_bounded_by_draft_steps(self) -> None:
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(64, 5, "cuda")
        session.begin(list(range(64)))
        visited = []
        for _ in range(6):
            session._prepare_plan()
            self.assertTrue(session._calibration)
            total = session.prepared_plan()[2]
            visited.append(total)
            session.round_costs[64][total] = float(total)
        session._prepare_plan()
        self.assertFalse(session._calibration)
        self.assertEqual(visited, [384, 320, 256, 192, 128, 64])

    def test_calibration_budget_rejects_invalid_inputs(self) -> None:
        controller = ready_controller([10, 20])
        for budget in (-1, 7, True, 1.5):
            with (
                self.subTest(budget=budget),
                self.assertRaises(AdaptiveBudgetInputError),
            ):
                controller.prepare_from_host(
                    [10, 20],
                    confidence=None,
                    draft_cost_ms=np.zeros(3),
                    verify_cost_ms=np.ones(9),
                    fixed_budget=budget,
                )
        with self.assertRaises(AdaptiveBudgetInputError):
            controller.prepare_from_host(
                [10, 20],
                confidence=None,
                draft_cost_ms=np.zeros(3),
                verify_cost_ms=np.ones(9),
                fixed_budget=2,
                fixed_cap=1,
            )
        for budget in range(7):
            caps, rows, total = controller.prepare_from_host(
                [10, 20],
                confidence=None,
                draft_cost_ms=np.zeros(3),
                verify_cost_ms=np.ones(9),
                fixed_budget=budget,
            )
            counts = caps.tolist()
            self.assertEqual(sum(counts), budget)
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertEqual(total, budget + 2)
            self.assertEqual(
                rows.tolist(),
                [i * 4 + j for i, count in enumerate(counts) for j in range(count + 1)],
            )

    def test_sparse_bucket_costs_calibration_and_eager_fallback(self) -> None:
        grid = [1, 2, 4, 8, 10, 12]
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 5, "cuda", grid)
            eager = AdaptiveVerificationSession(2, 5, "cuda")
        self.assertEqual(
            [session._cost_key(n) for n in range(1, 13)],
            [1, 2, 4, 4, 8, 8, 8, 8, 10, 10, 12, 12],
        )
        self.assertEqual([eager._cost_key(n) for n in range(1, 13)], list(range(1, 13)))
        session.begin([10, 20])
        # All fixed caps are calibrated in execution buckets: logical6 uses8.
        session.round_costs[2] = {2: 1.0, 4: 2.0, 8: 4.0, 10: 5.0, 12: 6.0}
        session.pending_costs = []
        with patch.object(
            session.controller,
            "prepare_from_host",
            wraps=session.controller.prepare_from_host,
        ) as prepare:
            session._prepare_plan()
        self.assertFalse(session._calibration)
        np.testing.assert_array_equal(
            prepare.call_args.kwargs["verify_cost_ms"][2:],
            [1, 2, 2, 4, 4, 4, 4, 5, 5, 6, 6],
        )
        # Full-cap count can itself be between buckets (one request, 6 rows).
        session.begin([10])
        session.round_costs[1] = {1: 1.0, 2: 2.0, 4: 3.0, 8: 4.0}
        with patch.object(
            session.controller,
            "prepare_from_host",
            wraps=session.controller.prepare_from_host,
        ) as prepare:
            session._prepare_plan()
        np.testing.assert_array_equal(
            prepare.call_args.kwargs["verify_cost_ms"][1:], [1, 2, 3, 3, 4, 4]
        )
        session.total = 5
        session.events = [Mock(), Mock()]
        session.finish()
        self.assertEqual(session.pending_costs[-1][1], 8)
        # A pending measurement in the same execution bucket prevents duplicate
        # calibration, while the plan waits conservatively at full cap.
        session.round_costs[1] = {1: 1.0, 2: 2.0, 4: 3.0}
        session._prepare_plan()
        self.assertTrue(session._calibration)
        self.assertEqual(session.total, 6)

    def test_budget_uses_free_rows_within_bucket_without_crossing_expensive_bucket(
        self,
    ):
        with patch.dict(os.environ, {"RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1"}):
            session = AdaptiveVerificationSession(2, 5, "cuda", [1, 2, 4, 8, 10, 12])
        session.begin([10, 20])
        session.round_costs[2] = {2: 1.0, 4: 2.0, 8: 3.0, 10: 100.0, 12: 100.0}
        session._host_predictions = {
            i: (session.round, np.full(5, 0.9)) for i in session.ids
        }
        session._prepare_plan()
        # 8-row bucket is optimal; no fictional savings for 5,6,7 live rows.
        self.assertEqual(session.total, 8)
        self.assertFalse(session._calibration)

    def test_invalid_capture_grids_are_rejected(self) -> None:
        for grid in ([1, 2, 2, 12], [2, 1, 12], [0, 12], [1, 8], [True, 12], [1, 12.0]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                AdaptiveVerificationSession(2, 5, "cuda", grid)

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
            [10, 20],
            2,
            torch.tensor([[4.0, 0.0], [6.0, 0.0]], device="cuda"),
            torch.tensor([2.0, 0.0], device="cuda"),
        )
        # Defaults after a custom temperature must still be one for both slots.
        controller.predict(
            [20, 10], 1, torch.tensor([[2.0, 0.0], [1.0, 0.0]], device="cuda")
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

    def test_ordered_feedback_preserves_effective_caps(self) -> None:
        ids = [10, 20, 30]
        controller = ready_controller(ids)
        caps, _, _ = controller.prepare_from_host(
            ids,
            confidence=None,
            fixed_cap=3,
            draft_cost_ms=np.zeros(4),
            verify_cost_ms=np.ones(13),
        )
        controller.commit_prepared(ids, caps)
        with patch.object(
            controller.estimator, "step", wraps=controller.estimator.step
        ) as step:
            controller.observe(
                ids,
                torch.tensor([2, 0, 1], device="cuda"),
                [False] * 3,
                asynchronous=True,
                effective_capacities=torch.tensor([2, 1, 1], device="cuda"),
            )
        mapping, sampled, rejected = step.call_args.args
        self.assertEqual(mapping.tolist(), [0, 1, 2])
        self.assertEqual(sampled.tolist(), [3, 1, 2])
        self.assertEqual(rejected.tolist(), [0, 1, 0])

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


class EpTailRowsTest(unittest.TestCase):
    def make_policy(self, upper_bucket: int) -> _MeasuredEpPolicy:
        execution = np.arange(85)
        buckets = np.asarray([1, 14, 56, 72, 76, upper_bucket, 84])
        execution[1:] = buckets[np.searchsorted(buckets, execution[1:])]
        return _MeasuredEpPolicy(5, execution, grouped_tail_cut=True)

    def test_nearest_smaller_physical_bucket(self) -> None:
        for upper_bucket in (80, 81):
            with self.subTest(upper_bucket=upper_bucket):
                policy = self.make_policy(upper_bucket)
                policy.begin(list(range(14)))
                self.assertEqual(policy.candidates, [80])
                self.assertEqual(policy.execution_rows[80], upper_bucket)

    def test_cost_comparison_excludes_unwarmed_shapes(self) -> None:
        policy = self.make_policy(81)
        ids = list(range(14))
        accepted = np.full(14, 3)
        dense_costs = [230.0, 68.0, 38.0, 38.0, 38.0]
        partial_costs = [205.0, 63.0, 36.0, 36.0, 36.0]
        for rows, costs in ((84, dense_costs), (80, partial_costs)):
            for ms in costs:
                self.assertEqual(policy.begin(ids), (rows, 1))
                policy.record(policy.epoch, tuple(ids), rows, ms, accepted)
        self.assertEqual(policy.begin(ids), (80, 2))
        self.assertEqual([ms for ms, _ in policy.samples[84]], dense_costs[-3:])
        self.assertEqual([ms for ms, _ in policy.samples[80]], partial_costs[-3:])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_effective_caps_are_checked_against_current_device_plan(self) -> None:
        ids = list(range(14))
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_ADAPTIVE_EP_CURRENT_SCORES": "1",
                "RTP_LLM_ADAPTIVE_VERIFY_CAP": "-1",
            },
        ):
            session = AdaptiveVerificationSession(
                14, 5, "cuda", [1, 56, 72, 76, 81, 84], 4, allow_ep_current_scores=True
            )
            unsupported = AdaptiveVerificationSession(
                14, 5, "cuda", [1, 56, 72, 76, 81, 84], 4
            )
        self.assertFalse(unsupported._ep_current_scores)
        cuda_i32 = {"dtype": torch.int32, "device": "cuda"}
        # Requests 12 and 13 are weakest in the old cohort history.
        historical = torch.tensor([5] * 12 + [0, 0], **cuda_i32)
        for _ in range(5):
            session.begin(ids)
            self.assertEqual(session.plan()[2], 84)
            session.observe(historical, [False] * 14)
            session.finish(38.0)
            torch.cuda.synchronize()

        session.begin(ids)
        historical_caps, _, total = session.prepared_plan()
        self.assertEqual(total, 80)
        self.assertEqual(historical_caps.tolist(), [5] * 12 + [3, 3])
        scores = (torch.arange(14, device="cuda", dtype=torch.float32) / 14).repeat(
            4, 1
        )
        current_caps, _, total = session.plan(scores)
        self.assertEqual(total, 80)
        self.assertEqual(current_caps.tolist(), [3, 3] + [5] * 12)
        effective = torch.tensor([2, 3] + [5] * 12, **cuda_i32)
        session.observe(effective, [False] * 14, effective)
        session.finish(36.0)
        torch.cuda.synchronize()
        with patch.object(
            session._ep_policy, "record", wraps=session._ep_policy.record
        ) as record:
            session._collect_ep_samples()
            self.assertEqual(record.call_count, 1)
            np.testing.assert_array_equal(record.call_args.args[-1], [2, 3] + [5] * 12)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_grouped_caps_and_stable_rows(self) -> None:
        scores = (torch.arange(14, dtype=torch.float32, device="cuda") / 14).repeat(
            4, 1
        )
        fallback = torch.tensor([5] * 12 + [3] * 2, dtype=torch.int32, device="cuda")
        capacities, rows = pack_ep_tail_rows(scores, fallback, 80)
        expected = [3, 3] + [5] * 12
        self.assertEqual(capacities.tolist(), expected)
        self.assertEqual(
            rows.tolist(),
            [6 * i + j for i, cap in enumerate(expected) for j in range(cap + 1)],
        )
        self.assertEqual(fallback.tolist(), [5] * 12 + [3] * 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_odd_cut_and_invalid_score_fallback(self) -> None:
        scores = torch.full((4, 14), float("nan"), device="cuda")
        fallback = torch.tensor([5] * 12 + [3, 4], dtype=torch.int32, device="cuda")
        scores[:, 1] = 0.1
        capacities, rows = pack_ep_tail_rows(scores, fallback, 81)
        self.assertEqual(capacities.tolist(), fallback.tolist())
        self.assertEqual(len(rows), 81)
        scores[:, 2] = 0.1
        capacities, rows = pack_ep_tail_rows(scores, fallback, 81)
        expected = [5, 3, 4] + [5] * 11
        self.assertEqual(capacities.tolist(), expected)
        self.assertEqual(
            rows.tolist(),
            [6 * i + j for i, cap in enumerate(expected) for j in range(cap + 1)],
        )


if __name__ == "__main__":
    unittest.main()
