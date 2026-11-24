"""Bounded continuous request supply, without engine imports."""

import json
import tempfile
import time
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.continuous_pressure import session as cp
from rtp_llm.test.continuous_pressure.config import (
    ExecutionLimits,
    resolve_pressure_config,
)
from rtp_llm.test.continuous_pressure.diagnostics import (
    GroupSample,
    LocalStoreSample,
    PoolSample,
    PressureOwnership,
    RankSample,
    select_targets,
)
from rtp_llm.test.continuous_pressure.phases import PressurePhases
from rtp_llm.test.continuous_pressure.seed_plan import (
    FORMAL_FILL_TARGET_TOKENS,
    MIN_FORMAL_CLIENT_INFLIGHT,
    MIN_REUSE_TOKENS,
    SeedPlan,
    align_up,
    build_seed_plan,
    formal_config_reasons,
)
from rtp_llm.test.continuous_pressure.sender import (
    BoundedSender,
    CancellableRequestHandle,
)
from rtp_llm.test.continuous_pressure.sequence import (
    AdaptiveBacklogController,
    ArrivalClock,
    TieredRollingPlanner,
    freeze_seed_families,
)


def config(**changes):
    values = dict(
        seed_timeout_s=30,
        fill_timeout_s=120,
        fill_max_requests=1000,
        fill_no_progress_s=30,
        fill_input_tokens=16,
        gpu_tolerance_blocks=0,
        activity_timeout_s=30,
        drain_timeout_s=60,
        capture_timeout_s=120,
    )
    values.update(changes)
    return resolve_pressure_config(values)


def seed_plan(seed_count=0, input_tokens=16, fill_tokens=16, client_inflight=4):
    return SeedPlan(
        source="manual",
        seed_count=seed_count,
        grid_reuse_tokens=0,
        reusable_tokens=input_tokens - 1,
        commit_tail_tokens=1,
        seed_input_tokens=input_tokens,
        fill_target_tokens=fill_tokens,
        fill_input_tokens=fill_tokens,
        client_inflight=client_inflight,
        formal_config_valid=False,
        nonformal_reasons=("manual_seed_count",),
        groups=(),
    )


class PhaseTest(unittest.TestCase):
    def test_measurement_ends_without_waiting_for_profile(self):
        state = PressurePhases(config(), started_s=0)
        self.assertEqual(state.advance(2, fill_ready=True), "steady")
        self.assertEqual(state.advance(5, steady_covered_s=3), "measurement")
        self.assertEqual(state.measurement_end, 65)
        self.assertEqual(state.advance(65, profile_pending=True), "support")
        self.assertEqual(state.advance(125, profile_pending=True), "drain")
        self.assertEqual(state.failure, "profile_timeout")

    def test_timeouts_and_existing_failure(self):
        state = PressurePhases(config(), started_s=0)
        self.assertEqual(state.advance(120, fill_ready=True), "drain")
        self.assertEqual(state.failure, "pressure_not_reached")
        state = PressurePhases(config(), started_s=0)
        state.advance(1, fill_ready=True)
        self.assertEqual(state.advance(61, steady_covered_s=3), "drain")

    def test_reject_nonfinite_and_negative_arrival_rates(self):
        for rate in (-1, float("inf"), float("nan"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                ArrivalClock(rate, False, 0)

    def test_execution_limits_bound_long_requests(self):
        limits = ExecutionLimits(128, 8, 100, 12, 16)
        self.assertEqual(limits.concurrency_for(33), 2)
        with self.assertRaises(ValueError):
            limits.concurrency_for(128)

    def test_live_pool_admission_keeps_each_owner_geometry(self):
        limits = ExecutionLimits(
            262144,
            1,
            200000,
            400,
            512,
            owner_kv_budgets=(
                ("owner=0 group=1 pool=wide", 2, 65536),
                ("owner=0 group=2 pool=narrow", 400, 512),
            ),
        )
        self.assertIsNone(limits.require_request(102400))
        self.assertEqual(limits.concurrency_for(102400), 1)
        with self.assertRaisesRegex(ValueError, "owner=0 group=1 pool=wide.*KV"):
            limits.require_request(150000)

    def test_tiered_rolling_pressure_settings_are_validated(self):
        try:
            resolved = config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
            )
        except ValueError as exc:
            self.fail(f"tiered rolling settings were rejected: {exc}")
        self.assertEqual(resolved.family_schedule, "tiered_rolling")
        self.assertEqual(resolved.memory_reuse_ratio, 0.5)
        self.assertEqual(resolved.fill_host_max_ratio, 0.8)
        with self.assertRaisesRegex(ValueError, "50%"):
            config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.25,
                fill_host_max_ratio=0.8,
            )

    def test_adaptive_pressure_settings_are_validated(self):
        resolved = config(
            family_schedule="tiered_rolling",
            memory_reuse_ratio=0.5,
            fill_host_max_ratio=0.8,
            rate_control="adaptive_backlog",
            request_rate_min=50,
            request_rate_max=400,
            backlog_low=64,
            backlog_high=192,
            max_outstanding=256,
            cohort_size=32,
        )
        self.assertEqual(resolved.rate_control, "adaptive_backlog")
        self.assertEqual(resolved.request_rate_max, 400)
        self.assertEqual(resolved.max_outstanding, 256)
        for changes in (
            {"request_rate_min": 0},
            {"request_rate_min": 500},
            {"backlog_low": 192},
            {"backlog_high": 256},
            {"cohort_size": 0},
            {"cohort_size": 31},
        ):
            adaptive = dict(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=50,
                request_rate_max=400,
                backlog_low=64,
                backlog_high=192,
                max_outstanding=256,
                cohort_size=32,
            )
            adaptive.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**adaptive)


class CombinationConfigTest(unittest.TestCase):
    def test_combined_targets_cannot_exceed_all_matched_blocks(self):
        with self.assertRaisesRegex(ValueError, "sum"):
            config(
                mock_memory_target_ratio=0.4,
                mock_disk_target_ratio=0.3,
                mock_remote_target_ratio=0.4,
            )
        config(
            mock_memory_target_ratio=0.4,
            mock_disk_target_ratio=0.3,
            mock_remote_target_ratio=0.3,
        )


class AdaptiveBacklogControllerTest(unittest.TestCase):
    def test_piecewise_rate_stays_positive_and_decreases_with_backlog(self):
        controller = AdaptiveBacklogController(
            rate_min=50,
            rate_max=400,
            backlog_low=64,
            backlog_high=192,
        )

        self.assertEqual(controller.rate(0), 400)
        self.assertEqual(controller.rate(64), 400)
        self.assertAlmostEqual(controller.rate(128), 225)
        self.assertEqual(controller.rate(192), 50)
        self.assertEqual(controller.rate(255), 50)

    def test_rejects_invalid_rate_and_watermark_ranges(self):
        for kwargs in (
            dict(rate_min=0, rate_max=400, backlog_low=64, backlog_high=192),
            dict(rate_min=500, rate_max=400, backlog_low=64, backlog_high=192),
            dict(rate_min=50, rate_max=400, backlog_low=192, backlog_high=192),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AdaptiveBacklogController(**kwargs)


class TieredRollingPlannerTest(unittest.TestCase):
    def planner(self, seed_count=1000):
        return TieredRollingPlanner(
            effective_seed_count=seed_count,
            cohort_size=32,
            memory_reuse_ratio=0.5,
            case_indices=(1, 2, 3, 4),
        )

    def test_promotes_each_memory_family_with_the_same_case_next_cohort(self):
        planner = self.planner()

        first = planner.next_cohort()
        second = planner.next_cohort()

        first_device = [item for item in first if item.expected_cache_tier == "device"]
        first_memory = [item for item in first if item.expected_cache_tier == "memory"]
        second_device = [
            item for item in second if item.expected_cache_tier == "device"
        ]
        second_memory = [
            item for item in second if item.expected_cache_tier == "memory"
        ]
        self.assertEqual([item.family for item in first_device], list(range(984, 1000)))
        self.assertEqual([item.family for item in first_memory], list(range(16)))
        self.assertEqual(
            [(item.family, item.case_index) for item in second_device],
            [(item.family, item.case_index) for item in first_memory],
        )
        self.assertEqual([item.family for item in second_memory], list(range(16, 32)))
        self.assertEqual(
            {
                case: sum(item.case_index == case for item in first_memory)
                for case in range(1, 5)
            },
            {1: 4, 2: 4, 3: 4, 4: 4},
        )
        self.assertTrue(all(item.cohort_id == 0 for item in first))
        self.assertTrue(all(item.cohort_id == 1 for item in second))
        self.assertTrue(all(item.promoted_from_cohort == 0 for item in second_device))

    def test_initial_rewarm_matches_first_device_half_without_consuming_cohort(self):
        planner = self.planner()

        rewarm = planner.initial_device_assignments()
        first = planner.next_cohort()
        first_device = [item for item in first if item.expected_cache_tier == "device"]

        self.assertEqual(rewarm, tuple(first_device))
        self.assertEqual([item.family for item in rewarm], list(range(984, 1000)))
        self.assertTrue(all(item.cohort_id == 0 for item in rewarm))
        self.assertTrue(all(item.promoted_from_cohort is None for item in rewarm))

    def test_effective_seed_boundary_wraps_without_using_the_initial_hot_tail(self):
        planner = self.planner(seed_count=48)

        first = planner.next_cohort()
        second = planner.next_cohort()
        third = planner.next_cohort()

        self.assertEqual(
            [item.family for item in first if item.expected_cache_tier == "device"],
            list(range(32, 48)),
        )
        self.assertEqual(
            [item.family for item in second if item.expected_cache_tier == "memory"],
            list(range(16, 32)),
        )
        self.assertEqual(
            [item.family for item in third if item.expected_cache_tier == "device"],
            list(range(16, 32)),
        )
        self.assertEqual(
            [item.family for item in third if item.expected_cache_tier == "memory"],
            list(range(16)),
        )
        self.assertTrue(
            all(
                item.family < 32
                for item in second + third
                if item.expected_cache_tier == "memory"
            )
        )

    def test_assignment_is_random_access_and_independent_of_worker_order(self):
        sequential = self.planner(seed_count=48)
        expected = sequential.next_cohort() + sequential.next_cohort()
        random_access = self.planner(seed_count=48)

        observed = {
            sequence: random_access.assignment(sequence)
            for sequence in reversed(range(64))
        }

        self.assertEqual(tuple(observed[index] for index in range(64)), expected)

    def test_freezes_only_the_contiguous_completed_seed_range(self):
        seeds = [[0], [1], [2], None, None]
        self.assertEqual(freeze_seed_families(seeds), 3)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            freeze_seed_families([[0], None, [2]])


class FakeClock:
    def __init__(self):
        self.now = 1.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now = round(self.now + seconds, 8)


class InlineExecutor:
    def __init__(self, **kwargs):
        self.closed = False

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kwargs):
        self.closed = True


class MockTerminalEvidenceTest(unittest.TestCase):
    def test_hybrid_requires_mock_counters_and_keeps_real_tree_readiness(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        cfg = replace(config(), evidence_policy="tree_then_mock_memory")
        sample = self.sample(1, 0.5)
        for mock in (None, replace(sample.mock_memory, terminal_outcome_version=0)):
            with self.subTest(mock=mock), self.assertRaisesRegex(
                ValueError, "mock memory"
            ):
                PressureCoverage((replace(sample, mock_memory=mock),), cfg)
        coverage = PressureCoverage((sample,), cfg)
        self.assertFalse(coverage.target_valid(coverage.targets[0], (sample,)))

    def test_hybrid_migration_uses_only_mock_outcomes_inside_measurement(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        cfg = replace(
            config(),
            evidence_policy="tree_then_mock_memory",
            mock_memory_target_ratio=0.5,
        )
        for success_at, failure_at, valid in (
            (1.0, None, False),
            (2.0, None, True),
            (2.0, 2.0, False),
            (2.0, 1.0, True),
        ):
            with self.subTest(success_at=success_at, failure_at=failure_at):
                coverage = PressureCoverage((self.sample(1, 0),), cfg)
                for sequence, now in enumerate((0.5, 1.0, 1.5, 2.0, 2.5), 2):
                    coverage.add(
                        (
                            self.sample(
                                sequence,
                                now,
                                completed=4 if now >= success_at else 0,
                                h2d_failed=int(
                                    failure_at is not None and now >= failure_at
                                ),
                                d2h_failed=int(
                                    failure_at is not None and now >= failure_at
                                ),
                            ),
                        ),
                        now,
                    )
                result = coverage.assess_migration(1.5, 2.5)
                self.assertEqual(result.valid, valid, result.reasons)
                if valid:
                    group = result.groups[0]
                    self.assertEqual(
                        group.h2d_evidence[0].source,
                        "mock_memory_successful_h2d_blocks",
                    )
                    self.assertEqual(
                        group.d2h_evidence[0].source, "mock_memory_d2h_completed_blocks"
                    )
                if failure_at == 2.0:
                    self.assertIn("failed_h2d:0:0", result.reasons)
                    self.assertIn("failed_d2h:0:0", result.reasons)

    def test_mock_policy_requires_supported_terminal_counter_semantics(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        baseline = self.sample(1, 0.5)
        # Old servers have the mock counters but date them by settlement.
        for version in (None, 0, 2):
            with self.subTest(version=version):
                fields = asdict(baseline.mock_memory)
                fields.pop("terminal_outcome_version", None)
                if version is not None:
                    fields["terminal_outcome_version"] = version
                old = replace(baseline, mock_memory=SimpleNamespace(**fields))
                with self.assertRaisesRegex(ValueError, "terminal"):
                    PressureCoverage((old,), config(evidence_policy="mock_memory"))

    def sample(self, sequence, now, *, completed=0, h2d_failed=0, d2h_failed=0):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            GroupSample,
            MockMemorySample,
            PoolSample,
            RankSample,
        )

        return RankSample(
            0,
            "service",
            sequence,
            now,
            now,
            now,
            (
                PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                PoolSample("host", 100, 0, 0, 0.8, 0.9),
            ),
            (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
            mock_memory=MockMemorySample(
                16,
                8,
                completed,
                h2d_failed,
                completed,
                completed,
                d2h_failed,
                terminal_outcome_version=1,
            ),
        )

    def test_failure_envelopes_overlapping_either_window_boundary_reject(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        # The break caught here is accepting positive transfers while silently
        # discarding a failure because its sampled envelope straddles a boundary.
        for direction in ("h2d", "d2h"):
            for failure_at in (1.5, 2.5):
                with self.subTest(direction=direction, failure_at=failure_at):
                    coverage = PressureCoverage(
                        (self.sample(1, 0.5),), config(evidence_policy="mock_memory")
                    )
                    for sequence, now in enumerate((1.0, 1.5, 2.0, 2.5), 2):
                        failed = int(now >= failure_at)
                        sample = self.sample(
                            sequence,
                            now,
                            completed=4 if now >= 2 else 0,
                            **{f"{direction}_failed": failed},
                        )
                        coverage.add((sample,), now)
                    result = coverage.assess_migration(1.25, 2.25)
                    self.assertFalse(result.valid)
                    self.assertIn(f"failed_{direction}:0:0", result.reasons)

    def test_delayed_settlement_cannot_move_terminal_evidence_across_boundaries(self):
        from rtp_llm.test.continuous_pressure.diagnostics import PressureCoverage

        # Counters model physical terminal notifications. The later settlement
        # poll carries unchanged counters and must neither create nor hide work.
        for terminal_at, settlement_at, expected in (
            (1.0, 2.0, False),
            (2.0, 3.0, True),
        ):
            with self.subTest(terminal_at=terminal_at, settlement_at=settlement_at):
                coverage = PressureCoverage(
                    (self.sample(1, 0.0),), config(evidence_policy="mock_memory")
                )
                for sequence, now in enumerate((0.5, 1.0, 1.5, 2.0, 2.5, 3.0), 2):
                    coverage.add(
                        (
                            self.sample(
                                sequence, now, completed=4 if now >= terminal_at else 0
                            ),
                        ),
                        now,
                    )
                result = coverage.assess_migration(1.5, 2.5)
                self.assertEqual(result.valid, expected)
                if expected:
                    self.assertEqual(result.groups[0].d2h_intervals, ((1.5, 2.0),))
                    self.assertEqual(result.groups[0].h2d_intervals, ((1.5, 2.0),))


class LifecycleTest(unittest.TestCase):
    def test_remote_policy_real_preparation_mock_measurement_and_final_drain(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            MockCacheSample,
            MockRemoteWriteSample,
        )

        measurement = []
        drained = []

        def remote_poll(evidence, sequence, clock):
            count = len(measurement)
            return replace(
                evidence,
                samples=tuple(
                    replace(
                        sample,
                        mock_cache=MockCacheSample(
                            eligible_blocks=count * 4, terminal_outcome_version=1
                        ),
                        mock_remote_write=MockRemoteWriteSample(
                            submitted=count, completed=count
                        ),
                        remote_read_payload_bytes=0,
                        remote_write_acknowledged_payload_bytes=count * 64,
                        remote_runtime='{"version":1,"session_id":"1"}',
                    )
                    for sample in evidence.samples
                ),
            )

        result, events, _, _ = self.run_session(
            seed_count=3,
            rate=0,
            cfg=replace(
                self.hybrid_config(),
                evidence_policy="tree_then_mock_remote",
                mock_memory_target_ratio=0,
            ),
            poll_transform=remote_poll,
            startup_transform=remote_poll,
            request_observer=lambda identity: (
                measurement.append(identity) if ":measurement:" in identity else None
            ),
            drain_observer=lambda: drained.append(len(measurement)),
        )
        self.assertIsNone(result["failure"], events)
        self.assertTrue(result["mock_remote"]["valid"], result)
        self.assertEqual(drained, [len(measurement)])
        dispatches = [e for e in events if e["event"] == "dispatch"]
        for phase in ("seed", "fill", "steady", "measurement"):
            self.assertTrue(any(e["phase"] == phase for e in dispatches))
        for event in dispatches:
            self.assertIs(
                event["metadata"]["mock_cache_request"], event["phase"] == "measurement"
            )
            self.assertNotIn("mock_memory_cache_request", event["metadata"])

    def test_local_drain_waits_for_stream_release_and_physical_terminal(self):
        for outcome in ("success", "timeout", "extra_release", "abandoned"):
            with self.subTest(outcome=outcome):
                draining = []
                measurement = []

                def observe(event, clock):
                    if event["event"] == "phase" and event["phase"] == "measurement":
                        measurement.append(clock())
                    if event["event"] == "phase" and event["phase"] == "drain":
                        draining.append(0)

                def poll(evidence, sequence, clock):
                    page = self.hybrid_poll(evidence, sequence, clock)
                    sample = page.samples[0]
                    requests = sample.local_store.released_requests
                    submitted = min(max(0, requests - 2), 2)
                    completed = int(
                        bool(
                            submitted
                            and measurement
                            and clock() >= measurement[0] + 0.35
                        )
                    )
                    pending = 0
                    if draining:
                        draining[0] += 1
                        if draining[0] == 1:
                            requests -= 1
                            pending = 1
                        elif draining[0] >= 3 and outcome != "timeout":
                            completed = submitted
                            if outcome == "extra_release":
                                requests += 1
                    return replace(
                        page,
                        samples=(
                            replace(
                                sample,
                                local_store=LocalStoreSample(
                                    released_requests=requests,
                                    pending_requests=pending,
                                    abandoned_requests=int(
                                        bool(draining) and outcome == "abandoned"
                                    ),
                                ),
                                mock_memory=replace(
                                    sample.mock_memory,
                                    d2h_submitted_blocks=submitted,
                                    d2h_completed_blocks=completed,
                                ),
                            ),
                        ),
                    )

                result, events, _, _ = self.run_session(
                    seed_count=3,
                    rate=0,
                    cfg=self.hybrid_config(drain_timeout_s=2),
                    poll_transform=poll,
                    event_sink=observe,
                )
                summary = result["local_writes"]
                self.assertGreaterEqual(draining[0], 3)
                self.assertEqual(summary["valid"], outcome == "success", summary)
                self.assertEqual(
                    result["migration"]["valid"],
                    outcome == "success",
                    result["migration"],
                )
                if outcome == "success":
                    row = summary["owners"]["0"]
                    self.assertGreater(
                        row["released_requests"], row["memory"]["submitted"]
                    )
                    self.assertEqual(row["memory"]["completed"], 2)
                else:
                    self.assertIsNotNone(result["failure"])

    def test_disk_hybrid_uses_generic_flag_only_in_measurement(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            MockCacheSample,
            PoolSample,
        )

        def disk_poll(evidence, sequence, clock):
            sample = evidence.samples[0]
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        mock_cache=MockCacheSample(
                            sequence * 8,
                            sequence * 2,
                            sequence * 2,
                            sequence * 2,
                            sequence * 2,
                            0,
                            sequence * 4,
                            sequence * 4,
                            0,
                            1,
                        ),
                        groups=tuple(
                            replace(g, disk_pool_id="disk") for g in sample.groups
                        ),
                        pools=sample.pools
                        + (
                            PoolSample(
                                "disk",
                                100,
                                10,
                                0,
                                0.8,
                                0.9,
                                tier=3,
                                disk_read_bytes=sequence * 4096,
                                disk_write_bytes=sequence * 4096,
                            ),
                        ),
                    ),
                ),
            )

        result, events, _, _ = self.run_session(
            seed_count=3,
            rate=0,
            cfg=replace(
                self.hybrid_config(),
                evidence_policy="tree_then_mock_disk",
                mock_memory_target_ratio=0,
            ),
            poll_transform=disk_poll,
            startup_transform=disk_poll,
        )
        self.assertIsNone(result["failure"], events)
        self.assertTrue(result["migration"]["valid"])
        self.assertIn("owners", result["mock_cache"])
        dispatches = [e for e in events if e["event"] == "dispatch"]
        for phase in ("seed", "fill", "steady", "measurement"):
            self.assertTrue(any(e["phase"] == phase for e in dispatches))
        for event in dispatches:
            self.assertIs(
                event["metadata"]["mock_cache_request"], event["phase"] == "measurement"
            )
            self.assertNotIn("mock_memory_cache_request", event["metadata"])

    def hybrid_poll(self, evidence, sequence, clock):
        from rtp_llm.test.continuous_pressure.diagnostics import MockMemorySample

        return replace(
            evidence,
            samples=(
                replace(
                    evidence.samples[0],
                    mock_memory=MockMemorySample(
                        sequence * 8,
                        sequence * 4,
                        sequence * 4,
                        0,
                        sequence * 4,
                        sequence * 4,
                        0,
                        terminal_outcome_version=1,
                    ),
                ),
            ),
        )

    def hybrid_config(self, **changes):
        return replace(
            config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.1,
                mock_memory_target_ratio=0.5,
                **changes,
            ),
            evidence_policy="tree_then_mock_memory",
        )

    def test_hybrid_runs_real_fill_without_rewarm_and_flags_only_measurement(self):
        result, events, _, session = self.run_session(
            seed_count=3,
            rate=0,
            cfg=self.hybrid_config(),
            poll_transform=self.hybrid_poll,
        )
        self.assertIsNone(result["failure"], events)
        self.assertEqual(result["rewarm"]["planned"], 0)
        dispatches = [event for event in events if event["event"] == "dispatch"]
        for phase in ("seed", "fill", "steady", "measurement"):
            self.assertTrue(any(event["phase"] == phase for event in dispatches), phase)
        for event in dispatches:
            self.assertIs(
                event["metadata"].get("mock_memory_cache_request"),
                event["phase"] == "measurement",
            )
        self.assertTrue(result["pressure"]["valid"])
        self.assertTrue(result["migration"]["valid"])
        for group in result["migration"]["groups"]:
            self.assertTrue(
                all(
                    item["source"].startswith("mock_memory_")
                    for item in group["h2d_evidence"] + group["d2h_evidence"]
                )
            )

    def test_hybrid_cannot_use_mock_outcomes_to_skip_real_fill(self):
        result, events, _, _ = self.run_session(
            seed_count=1,
            valid=False,
            rate=0,
            cfg=self.hybrid_config(fill_max_requests=3),
            poll_transform=self.hybrid_poll,
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertIsNone(result["measurement_start"])

    def test_hybrid_keeps_host_headroom_fail_closed(self):
        def exhaust(evidence, sequence, clock):
            evidence = self.hybrid_poll(evidence, sequence, clock)
            sample = evidence.samples[0]
            if sequence >= 3:
                sample = replace(
                    sample,
                    pools=tuple(
                        replace(pool, used=85) if pool.pool_id == "host" else pool
                        for pool in sample.pools
                    ),
                )
            return replace(evidence, samples=(sample,))

        result, _, _, _ = self.run_session(
            rate=0,
            cfg=self.hybrid_config(),
            poll_transform=exhaust,
        )
        self.assertEqual(result["failure"], "fill_headroom_exhausted")

    def test_mock_memory_policy_runs_formal_measurement_without_tree_pressure(self):
        from rtp_llm.test.continuous_pressure.diagnostics import MockMemorySample

        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return False

            def stop(self, reason, now):
                self.stopped = (reason, now)

            def finish(self, coverage):
                return dict(trace_valid=True, pressure_valid=True)

        controller = Controller()

        def mock_only(evidence, sequence, clock):
            sample = evidence.samples[0]
            return cp.PressurePoll(
                (
                    replace(
                        sample,
                        pools=tuple(
                            replace(pool, used=0, pending_release=0, host_resident=0)
                            for pool in sample.pools
                        ),
                        groups=tuple(
                            replace(group, triggers=0, gpu_host_published=0)
                            for group in sample.groups
                        ),
                        counters=(),
                        mock_memory=MockMemorySample(
                            eligible_blocks=sequence * 8,
                            planned_host_blocks=sequence * 4,
                            successful_h2d_blocks=100 + max(0, sequence - 1) * 4,
                            failed_h2d_blocks=7 if sequence > 1 else 0,
                            d2h_submitted_blocks=100 + max(0, sequence - 1) * 4,
                            d2h_completed_blocks=100 + max(0, sequence - 1) * 4,
                            d2h_failed_blocks=9 if sequence > 1 else 0,
                            terminal_outcome_version=1,
                        ),
                    ),
                ),
                evidence.raw,
                evidence.activity,
            )

        result, events, _, session = self.run_session(
            valid=False,
            seed_count=3,
            controller=controller,
            poll_transform=mock_only,
            rate=0,
            cfg=config(
                evidence_policy="mock_memory",
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=10,
                request_rate_max=20,
                backlog_low=1,
                backlog_high=2,
                max_outstanding=4,
                cohort_size=2,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.1,
            ),
        )

        dispatches = [event for event in events if event["event"] == "dispatch"]
        self.assertIsNone(result["failure"], events)
        self.assertEqual(result["seed"]["completed"], 3)
        self.assertTrue(any(event["event"] == "fill_baseline" for event in events))
        baseline = session.coverage.baselines[0].mock_memory
        self.assertGreater(baseline.successful_h2d_blocks, 100)
        self.assertEqual(baseline.failed_h2d_blocks, 7)
        self.assertEqual(baseline.d2h_failed_blocks, 9)
        self.assertIsNotNone(result["measurement_start"])
        self.assertEqual(controller.started, result["measurement_start"])
        self.assertFalse(any(event["phase"] == "fill" for event in dispatches))
        self.assertTrue(any(event["phase"] == "measurement" for event in dispatches))
        self.assertTrue(result["migration"]["valid"])
        for group in result["migration"]["groups"]:
            self.assertTrue(group["valid"])
            self.assertTrue(
                all(
                    item["before_blocks"] >= baseline.successful_h2d_blocks
                    for item in group["h2d_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["before_blocks"] >= baseline.d2h_completed_blocks
                    for item in group["d2h_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["source"] == "mock_memory_successful_h2d_blocks"
                    for item in group["h2d_evidence"]
                )
            )
            self.assertTrue(
                all(
                    item["source"] == "mock_memory_d2h_completed_blocks"
                    for item in group["d2h_evidence"]
                )
            )

    def test_mock_memory_policy_rejects_completion_failures(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            GroupSample,
            MockMemorySample,
            PoolSample,
            PressureCoverage,
            RankSample,
        )

        def sample(sequence, now, *, completed, failed):
            return RankSample(
                0,
                "service",
                sequence,
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 0, 0, 0.8, 0.9),
                    PoolSample("host", 100, 0, 0, 0.8, 0.9),
                ),
                (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
                counters=(),
                mock_memory=MockMemorySample(
                    eligible_blocks=16,
                    planned_host_blocks=8,
                    successful_h2d_blocks=completed,
                    failed_h2d_blocks=failed,
                    d2h_submitted_blocks=completed,
                    d2h_completed_blocks=completed,
                    d2h_failed_blocks=failed,
                    terminal_outcome_version=1,
                ),
            )

        coverage = PressureCoverage(
            (sample(1, 1.0, completed=0, failed=0),),
            config(evidence_policy="mock_memory"),
        )
        coverage.add((sample(2, 1.5, completed=0, failed=0),), 1.5)
        coverage.add((sample(3, 2.0, completed=4, failed=1),), 2.0)
        assessment = coverage.assess_migration(1.5, 2.0)
        self.assertFalse(assessment.valid)
        self.assertIn("failed_h2d:0:0", assessment.reasons)
        self.assertIn("failed_d2h:0:0", assessment.reasons)

    def test_early_capture_completion_preserves_measurement_and_finalizes_before_finished(
        self,
    ):
        cleanup = []
        original_drain = cp.PressureSession._drain
        original_shutdown = InlineExecutor.shutdown

        def drain(session):
            original_drain(session)
            cleanup.append("drained")

        def shutdown(executor, **kwargs):
            original_shutdown(executor, **kwargs)
            cleanup.append("shutdown")

        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return False

            def stop(self, reason, now):
                self.stopped = now

            def consume(self, event):
                if event["event"] == "finished":
                    assert self.finalized

            def finish(self, coverage):
                assert cleanup == ["drained", "shutdown", "shutdown"]
                self.finalized = True
                return dict(trace_valid=True, pressure_valid=False)

        controller = Controller()
        with patch.object(cp.PressureSession, "_drain", drain), patch.object(
            InlineExecutor, "shutdown", shutdown
        ):
            result, events, _, _ = self.run_session(
                controller=controller, cfg=config(measure_s=4, steady_s=1)
            )
        self.assertAlmostEqual(
            result["measurement_end"] - result["measurement_start"], 4
        )
        self.assertFalse(any(e.get("phase") == "support" for e in events))
        self.assertTrue(result["profile"]["trace_valid"])
        self.assertFalse(result["profile"]["pressure_valid"])

    def run_session(
        self,
        startup_transform=None,
        *,
        valid=True,
        cfg=None,
        controller=None,
        poll_transform=None,
        payload_hook=None,
        seed_count=0,
        rate=3,
        diagnostics_control=None,
        concurrency=4,
        request_observer=None,
        drain_observer=None,
        transport_observer=None,
        event_sink=None,
        executor_factory=InlineExecutor,
    ):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock, events, batches = FakeClock(), [], []
        measured = []
        counter = [0]

        def sample(sequence):
            group = GroupSample(
                0, "FULL", ("gpu",), "host", max(0, sequence - 1), max(0, sequence - 1)
            )
            return RankSample(
                0,
                "service",
                sequence,
                clock(),
                clock(),
                clock(),
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 30, 0, 0.8, 0.9, 25 if valid else 0),
                ),
                (group,),
                local_store=LocalStoreSample(released_requests=len(measured)),
            )

        def poll(timeout, drain_remote_completions=False):
            if drain_remote_completions and drain_observer:
                drain_observer()
            counter[0] += 1
            evidence = cp.PressurePoll((sample(counter[0]),), {}, ())
            return (
                poll_transform(evidence, counter[0], clock)
                if poll_transform
                else evidence
            )

        def request(payload, register, before):
            identity = before(payload)
            if ":measurement:" in identity:
                measured.append(identity)
            if request_observer:
                request_observer(identity)
            if transport_observer:
                transport_observer(identity, payload, clock())
            return dict(
                success=True,
                input_len=16,
                output_len=1,
                reuse_len=8,
            )

        def payload(kind, sequence, formal_sequence):
            if payload_hook:
                payload_hook(kind, sequence, clock)
            return {"input_ids": [1] * 16, "case": "cold"}

        def sink(event):
            events.append(event)
            if event_sink:
                event_sink(event, clock)

        state = cp.PressureSession(
            cfg or config(measure_s=3, steady_s=1, sample_period_s=0.5),
            session_id="session",
            limits=ExecutionLimits(128, 8, 256, 64, 16),
            concurrency=concurrency,
            max_input_tokens=16,
            seed_plan=seed_plan(seed_count, client_inflight=concurrency),
            startup_targets=select_targets(
                (
                    startup_transform(
                        cp.PressurePoll((sample(0),), {}, ()), 0, clock
                    ).samples
                    if startup_transform
                    else (sample(0),)
                ),
                (0,),
            ),
            make_payload=payload,
            request=request,
            poll=poll,
            sink=sink,
            controller=controller,
            diagnostics_control=diagnostics_control,
            clock=clock,
            sleep=clock.sleep,
            executor_factory=executor_factory,
            rate=rate,
        )
        result = state.run()
        return result, events, batches, state

    def test_diagnostics_enable_and_disable_enclose_baseline_and_work(self):
        controls = []
        polls_after_disable = []

        def poll(evidence, count, clock):
            self.assertEqual(controls[0][0], "enable")
            polls_after_disable.append(controls[-1][0] == "disable")
            return evidence

        result, events, _, _ = self.run_session(
            diagnostics_control=lambda action, phase, timeout: controls.append(
                (action, phase)
            ),
            poll_transform=poll,
        )
        self.assertIsNone(result["failure"])
        self.assertEqual(controls[0], ("enable", ""))
        self.assertEqual(controls[-1], ("disable", ""))
        self.assertIn(("phase", "measurement"), controls)
        self.assertEqual(events[-1]["event"], "finished")
        self.assertTrue(polls_after_disable[-1])

    def test_diagnostics_control_uses_configured_rpc_and_application_timeouts(self):
        rpc_timeouts = []
        application_timeouts = []

        class TrackingFuture:
            def __init__(self, future):
                self.future = future

            def result(self, timeout=None):
                application_timeouts.append(timeout)
                return self.future.result(timeout=timeout)

            def __getattr__(self, name):
                return getattr(self.future, name)

        class TrackingExecutor(InlineExecutor):
            def submit(self, fn, *args):
                return TrackingFuture(super().submit(fn, *args))

        def control(action, phase, timeout):
            rpc_timeouts.append(timeout)
            return {"success": True}

        result, _, _, _ = self.run_session(
            cfg=config(control_timeout_s=7.5),
            diagnostics_control=control,
            executor_factory=TrackingExecutor,
        )

        self.assertIsNone(result["failure"])
        self.assertTrue(rpc_timeouts)
        self.assertEqual(set(rpc_timeouts), {7.5})
        self.assertIn(7.6, application_timeouts)

    def test_partial_enable_and_disable_failure_are_final_failures(self):
        for fail_action in ("enable", "disable", "phase"):
            calls = []

            def control(action, phase, timeout):
                calls.append(action)
                if action == fail_action:
                    raise ValueError("partial rank failure")

            result, events, _, _ = self.run_session(diagnostics_control=control)
            self.assertIsNotNone(result["failure"])
            self.assertEqual(calls[-1], "disable")
            self.assertTrue(any(e["event"] == "error" for e in events))
            if fail_action == "enable":
                self.assertFalse(any(e["event"] == "dispatch" for e in events))

    def test_invalid_outstanding_final_poll_cannot_be_hidden_by_clean_page(self):
        self._assert_cleanup_diagnostic_failure(final_steps=True)

    def test_profile_failure_does_not_hide_invalid_drain_poll(self):
        self._assert_cleanup_diagnostic_failure(final_steps=False)

    def _assert_cleanup_diagnostic_failure(self, *, final_steps):
        from rtp_llm.test.continuous_pressure.results import (
            PressureEvidence,
            PressureWriter,
        )

        _, _, _, session = self.run_session()
        session.state.phase = "drain"
        session.state.failure = None if final_steps else "profile_timeout"
        invalid = cp.PressurePoll(
            (),
            dict(
                diagnostics_error="invalid clock bounds",
                response=dict(ranks=[dict(world_rank=0)]),
            ),
        )
        session.poll_future = Future()
        session.poll_future.set_result(invalid)
        session.poll_executor = InlineExecutor()
        session.poll = lambda timeout: cp.PressurePoll(
            (), dict(response=dict(ranks=[]))
        )
        session.next_poll = float("inf")
        with tempfile.TemporaryDirectory() as directory:
            writer = PressureWriter(Path(directory), {})
            events = []
            session.sink = PressureEvidence(
                writer, events.append, profile_requested=True
            )
            if final_steps:
                session._final_steps()
            else:
                session._drain()
            session.emit("finished", failure=session.state.failure)
            writer.close()
            summary = json.loads(
                (Path(directory) / "pressure/summary.json").read_text()
            )
            self.assertFalse(summary["validity"]["execution_valid"])
            self.assertTrue(any(event["event"] == "error" for event in events))
            self.assertTrue(
                any(
                    event["event"] == "pressure" and event["raw"] == invalid.raw
                    for event in events
                )
            )
            if not final_steps:
                self.assertEqual(summary["result"]["failure"], "profile_timeout")

    def test_all_cold_fills_then_uninterrupted_formal_schedule(self):
        result, events, batches, state = self.run_session()
        dispatch = [e for e in events if e["event"] == "dispatch"]
        self.assertIsNone(result["failure"])
        self.assertEqual(batches, [])
        self.assertTrue(any(e["phase"] == "fill" for e in dispatch))
        self.assertFalse(any(e["phase"] == "seed" for e in dispatch))
        formal = [e for e in dispatch if e["phase"] in ("steady", "measurement")]
        self.assertEqual(
            [e["sequence"] for e in dispatch], sorted({e["sequence"] for e in dispatch})
        )
        self.assertTrue(any(e["phase"] == "measurement" for e in formal))
        for first, second in zip(formal, formal[1:]):
            self.assertAlmostEqual(
                second["scheduled_offset_s"] - first["scheduled_offset_s"], 1 / 3
            )
        self.assertTrue(
            all(
                e["attempted_dispatch_s"] < result["measurement_end"]
                for e in formal
                if e["phase"] == "measurement"
            )
        )
        self.assertGreater(state.last_poll_s, result["measurement_end"])

    def test_tiered_real_pressure_rewarms_device_half_before_steady(self):
        kinds = []
        cfg = config(
            family_schedule="tiered_rolling",
            memory_reuse_ratio=0.5,
            fill_host_max_ratio=0.8,
            rate_control="adaptive_backlog",
            request_rate_min=1,
            request_rate_max=2,
            backlog_low=1,
            backlog_high=3,
            max_outstanding=8,
            cohort_size=4,
            measure_s=1,
            steady_s=0.5,
            sample_period_s=0.25,
        )

        result, events, _, _ = self.run_session(
            cfg=cfg,
            seed_count=8,
            rate=0,
            payload_hook=lambda kind, sequence, clock: kinds.append(kind),
        )

        self.assertIsNone(result["failure"])
        self.assertEqual(
            result["rewarm"],
            {
                "planned": 2,
                "submitted": 2,
                "completed": 2,
                "failed": 0,
                "pending": 0,
            },
        )
        self.assertEqual(kinds.count("rewarm"), 2)
        dispatch = [event for event in events if event["event"] == "dispatch"]
        rewarm_indices = [
            index for index, event in enumerate(dispatch) if event["phase"] == "rewarm"
        ]
        formal_indices = [
            index
            for index, event in enumerate(dispatch)
            if event["phase"] in ("steady", "measurement", "support")
        ]
        self.assertEqual(len(rewarm_indices), 2)
        self.assertLess(max(rewarm_indices), min(formal_indices))

    def test_tiered_fixed_supply_rewarms_half_the_client_concurrency(self):
        result, _, _, _ = self.run_session(
            seed_count=8,
            concurrency=4,
            rate=0,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.25,
            ),
        )

        self.assertIsNone(result["failure"])
        self.assertEqual(result["rewarm"]["planned"], 2)
        self.assertEqual(result["rewarm"]["completed"], 2)

    def test_tiered_fixed_supply_requires_even_client_concurrency(self):
        with self.assertRaisesRegex(ValueError, "even effective cohort"):
            self.run_session(
                seed_count=8,
                concurrency=3,
                rate=0,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                ),
            )

    def test_rewarm_submits_in_waves_when_cohort_half_exceeds_sender_limit(self):
        result, _, _, _ = self.run_session(
            seed_count=12,
            concurrency=4,
            rate=0,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=1,
                request_rate_max=2,
                backlog_low=0,
                backlog_high=1,
                max_outstanding=2,
                cohort_size=8,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.25,
            ),
        )

        self.assertIsNone(result["failure"])
        self.assertEqual(result["rewarm"]["planned"], 4)
        self.assertEqual(result["rewarm"]["submitted"], 4)
        self.assertEqual(result["rewarm"]["completed"], 4)

    def test_rewarm_stops_before_next_wave_when_host_headroom_is_exhausted(self):
        rewarm_payloads = [0]

        def count_rewarm(kind, sequence, clock):
            if kind == "rewarm":
                rewarm_payloads[0] += 1

        def exhaust_after_first_wave(evidence, count, clock):
            if rewarm_payloads[0] < 2:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, _, _, _ = self.run_session(
            seed_count=12,
            concurrency=4,
            rate=0,
            payload_hook=count_rewarm,
            poll_transform=exhaust_after_first_wave,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=1,
                request_rate_max=2,
                backlog_low=0,
                backlog_high=1,
                max_outstanding=2,
                cohort_size=8,
            ),
        )

        self.assertEqual(result["failure"], "fill_headroom_exhausted")
        self.assertEqual(result["rewarm"]["submitted"], 2)

    def test_rewarm_rechecks_host_headroom_before_steady(self):
        rewarm_started = [False]

        def mark_rewarm(kind, sequence, clock):
            if kind == "rewarm":
                rewarm_started[0] = True

        def exhaust_after_rewarm(evidence, count, clock):
            if not rewarm_started[0]:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, events, _, _ = self.run_session(
            seed_count=8,
            rate=0,
            payload_hook=mark_rewarm,
            poll_transform=exhaust_after_rewarm,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=1,
                request_rate_max=2,
                backlog_low=1,
                backlog_high=3,
                max_outstanding=8,
                cohort_size=4,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.25,
            ),
        )

        self.assertEqual(result["failure"], "fill_headroom_exhausted")
        self.assertFalse(
            any(
                event["event"] == "dispatch" and event["kind"] == "formal"
                for event in events
            )
        )

    def test_failed_rewarm_immediately_cancels_pending_siblings(self):
        active = [None]
        rewarm_handles = []

        class FailedThenPendingRewarmHandle:
            def __init__(self, executor, work, on_done=None):
                self.phase = None
                self.cancelled = False
                self.done_checks = 0
                active[0] = self
                self.value = work(lambda call: None)
                active[0] = None
                if self.phase == "rewarm":
                    rewarm_handles.append(self)
                    self.rewarm_position = len(rewarm_handles)
                    if self.rewarm_position == 1:
                        self.value["success"] = False

            def done(self):
                if self.phase == "rewarm" and getattr(self, "rewarm_position", 0) == 2:
                    self.done_checks += 1
                    return self.cancelled
                return True

            def result(self):
                return self.value

            def cancel(self):
                self.cancelled = True
                return True

        def observe(identity):
            active[0].phase = identity.rsplit(":", 2)[-2]

        with patch.object(
            cp, "CancellableRequestHandle", FailedThenPendingRewarmHandle
        ):
            result, events, _, _ = self.run_session(
                seed_count=8,
                rate=0,
                request_observer=observe,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                    rate_control="adaptive_backlog",
                    request_rate_min=1,
                    request_rate_max=2,
                    backlog_low=1,
                    backlog_high=3,
                    max_outstanding=8,
                    cohort_size=4,
                    drain_timeout_s=1,
                ),
            )

        self.assertEqual(result["failure"], "request_failed", events)
        self.assertEqual(len(rewarm_handles), 2)
        self.assertTrue(rewarm_handles[1].cancelled)
        self.assertLessEqual(rewarm_handles[1].done_checks, 2)

    def test_rewarm_payload_failures_are_counted(self):
        def fail_rewarm_payload(kind, sequence, clock):
            if kind == "rewarm":
                raise RuntimeError("rewarm payload failed")

        result, _, _, _ = self.run_session(
            seed_count=8,
            rate=0,
            payload_hook=fail_rewarm_payload,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                rate_control="adaptive_backlog",
                request_rate_min=1,
                request_rate_max=2,
                backlog_low=1,
                backlog_high=3,
                max_outstanding=8,
                cohort_size=4,
            ),
        )

        self.assertEqual(result["failure"], "request_failed")
        self.assertEqual(result["rewarm"]["submitted"], 2)
        self.assertEqual(result["rewarm"]["failed"], 2)

    def test_insufficient_residency_stops_at_request_budget(self):
        result, events, _, _ = self.run_session(
            valid=False, cfg=config(fill_max_requests=3)
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertEqual(
            sum(e["event"] == "dispatch" and e["phase"] == "fill" for e in events), 3
        )

    def test_support_capture_deadline_stops_controller(self):
        class Controller:
            def __init__(self):
                self.stops = []

            def start(self, now):
                self.started = now

            def pending(self, now):
                return True

            def stop(self, reason, now):
                self.stops.append((reason, now))

        controller = Controller()
        result, events, _, _ = self.run_session(
            controller=controller,
            cfg=config(
                measure_s=2,
                steady_s=1,
                capture_timeout_s=3,
                support_s=10,
                sample_period_s=0.5,
            ),
        )
        self.assertEqual(result["failure"], "profile_timeout")
        self.assertTrue(any(e.get("phase") == "support" for e in events))
        self.assertIn("support_pressure", result)
        self.assertAlmostEqual(controller.stops[0][1], controller.started + 3, places=2)

    def test_fill_time_and_no_progress_budgets(self):
        for settings in (dict(fill_timeout_s=1), dict(fill_no_progress_s=1)):
            with self.subTest(settings=settings):
                result, _, _, state = self.run_session(
                    valid=False, cfg=config(**settings)
                )
                self.assertEqual(result["failure"], "pressure_not_reached")
                self.assertLess(state.clock(), 4)

    def test_seed_transfers_cannot_satisfy_fill_evidence(self):
        def seed_only_transfers(evidence, count, clock):
            sample = evidence.samples[0]
            group = replace(
                sample.groups[0],
                triggers=int(count > 1),
                gpu_host_published=int(count > 1),
            )
            return replace(evidence, samples=(replace(sample, groups=(group,)),))

        result, events, _, session = self.run_session(
            seed_count=1,
            cfg=config(fill_max_requests=2),
            poll_transform=seed_only_transfers,
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertEqual(result["seed"]["completed"], 1)
        self.assertFalse(
            any(e.get("phase") in ("steady", "measurement") for e in events)
        )
        self.assertEqual(session.coverage.baselines[0].groups[0].triggers, 1)

    def test_terminal_migration_serializes_window_local_owner_publications(self):
        def transfers(evidence, count, clock):
            sample = evidence.samples[0]
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), count),
                            (("published_blocks", 0, "gpu", 1, 2, 1), count),
                        ),
                    ),
                ),
            )

        result, events, _, session = self.run_session(poll_transform=transfers)
        self.assertIsNone(result["failure"])
        self.assertTrue(result["migration"]["valid"])
        self.assertEqual(result["migration"]["reasons"], ())
        self.assertEqual(len(result["migration"]["groups"]), 1)
        group = result["migration"]["groups"][0]
        self.assertEqual(group["d2h_evidence"][0]["source"], "published_blocks")
        for direction in ("d2h", "h2d"):
            for start, end in group[f"{direction}_intervals"]:
                self.assertGreaterEqual(start, result["measurement_start"])
                self.assertLessEqual(end, result["measurement_end"])
        self.assertGreater(session.last_poll_s, result["measurement_end"])
        self.assertEqual(events[-1]["migration"], result["migration"])
        json.dumps(result, allow_nan=False)

    def test_preload_and_fill_publications_cannot_satisfy_measurement_migration(self):
        def seed_only(evidence, count, clock):
            sample = evidence.samples[0]
            published = int(count > 1)
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), published),
                            (("published_blocks", 0, "gpu", 1, 2, 1), published),
                        ),
                    ),
                ),
            )

        result, _, _, session = self.run_session(seed_count=1, poll_transform=seed_only)
        self.assertIsNone(result["failure"])
        self.assertEqual(
            dict(session.coverage.baselines[0].counters)[
                ("published_blocks", 0, "host", 2, 1, 2)
            ],
            1,
        )
        self.assertTrue(result["pressure"]["valid"])
        self.assertFalse(result["migration"]["valid"])
        self.assertEqual(
            result["migration"]["reasons"], ("missing_d2h:0:0", "missing_h2d_load:0:0")
        )

    def test_support_only_publications_do_not_satisfy_measurement_migration(self):
        phase, published = ["seed"], [0]

        class Controller:
            def start(self, now):
                pass

            def pending(self, now):
                return True

            def stop(self, reason, now):
                pass

        def control(action, current, timeout):
            if action == "phase":
                phase[0] = current

        def support_only(evidence, count, clock):
            if phase[0] == "support":
                published[0] += 1
            sample = evidence.samples[0]
            return replace(
                evidence,
                samples=(
                    replace(
                        sample,
                        counters=(
                            (("published_blocks", 0, "host", 2, 1, 2), published[0]),
                            (("published_blocks", 0, "gpu", 1, 2, 1), published[0]),
                        ),
                    ),
                ),
            )

        result, events, _, _ = self.run_session(
            poll_transform=support_only,
            diagnostics_control=control,
            controller=Controller(),
            cfg=config(
                measure_s=2, steady_s=1, capture_timeout_s=4, sample_period_s=0.5
            ),
        )
        self.assertTrue(any(e.get("phase") == "support" for e in events))
        self.assertGreater(published[0], 0)
        self.assertFalse(result["migration"]["valid"])
        self.assertEqual(result["migration"]["groups"][0]["d2h_intervals"], ())

    def test_seed_count_is_exact_and_baseline_precedes_sends(self):
        result, events, batches, _ = self.run_session(seed_count=7)
        self.assertIsNone(result["failure"])
        self.assertEqual(events[0]["event"], "frozen")
        self.assertEqual(batches, [])
        self.assertEqual(result["seed"]["submitted"], 7)
        self.assertEqual(result["seed"]["completed"], 7)
        self.assertEqual(
            sum(e["event"] == "dispatch" and e["phase"] == "seed" for e in events), 7
        )

    def test_tiered_seed_limit_freezes_the_completed_contiguous_range(self):
        def reach_headroom(evidence, count, clock):
            if count < 2:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, events, _, _ = self.run_session(
            seed_count=20,
            concurrency=4,
            rate=0,
            poll_transform=reach_headroom,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                measure_s=1,
                steady_s=0.5,
                sample_period_s=0.5,
            ),
        )

        self.assertEqual(result["seed"].get("configured_limit"), 20)
        self.assertEqual(result["seed"].get("effective_seed_count"), 8)
        self.assertEqual(result["seed"].get("stop_reason"), "host_max_ratio")
        self.assertEqual(result["seed"]["planned"], 8)
        self.assertEqual(result["seed"]["submitted"], 8)
        self.assertEqual(result["seed"]["completed"], 8)
        self.assertEqual(result["failure"], "fill_headroom_exhausted")
        self.assertFalse(
            any(
                event["event"] == "dispatch" and event["phase"] == "fill"
                for event in events
            )
        )
        self.assertTrue(
            any(event["event"] == "seed_limit" for event in events),
            "the effective seed boundary must be persisted",
        )

    def test_round_robin_seed_does_not_apply_tiered_host_headroom_guard(self):
        result, _, _, _ = self.run_session(
            seed_count=4,
            cfg=config(fill_host_max_ratio=0.2, measure_s=1, steady_s=0.5),
        )

        self.assertIsNone(result["failure"])

    def test_fill_stops_before_consuming_reserved_host_headroom(self):
        def exhaust_headroom(evidence, count, clock):
            # Let fill first satisfy the pressure target, then prove that the
            # reserved host headroom remains a hard stop rather than a success.
            if count < 3:
                return evidence
            sample = evidence.samples[0]
            pools = tuple(
                replace(pool, used=85) if pool.pool_id == "host" else pool
                for pool in sample.pools
            )
            return replace(evidence, samples=(replace(sample, pools=pools),))

        result, events, _, _ = self.run_session(
            valid=True,
            rate=0,
            poll_transform=exhaust_headroom,
            cfg=config(
                family_schedule="tiered_rolling",
                memory_reuse_ratio=0.5,
                fill_host_max_ratio=0.8,
                fill_max_requests=100,
                fill_no_progress_s=30,
            ),
        )

        self.assertEqual(result["failure"], "fill_headroom_exhausted")
        fill_dispatches = [
            event
            for event in events
            if event["event"] == "dispatch" and event["phase"] == "fill"
        ]
        self.assertLessEqual(len(fill_dispatches), 4)

    def test_tiered_formal_supply_refills_before_the_whole_cohort_completes(self):
        active = [None]
        formal_dispatches = []
        formal_completions = []
        completions_before_refill = []

        class StaggeredHandle:
            def __init__(self, executor, work, on_done=None):
                self.phase = None
                self.formal_index = None
                self.checks = 0
                active[0] = self
                self.value = work(lambda call: None)
                active[0] = None

            def done(self):
                if self.phase not in ("steady", "measurement", "support"):
                    return True
                self.checks += 1
                lane = self.formal_index % 32
                return self.checks > lane

            def result(self):
                if self.formal_index is not None:
                    formal_completions.append(self.formal_index)
                return self.value

            def cancel(self):
                return True

        def observe(identity):
            handle = active[0]
            handle.phase = identity.rsplit(":", 2)[-2]
            if handle.phase in ("steady", "measurement", "support"):
                handle.formal_index = len(formal_dispatches)
                if handle.formal_index == 32 and len(formal_completions) < 32:
                    completions_before_refill.append(len(formal_completions))
                formal_dispatches.append(identity)

        with patch.object(cp, "CancellableRequestHandle", StaggeredHandle):
            result, _, _, _ = self.run_session(
                concurrency=32,
                rate=0,
                request_observer=observe,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                    steady_s=0.2,
                    measure_s=0.5,
                    sample_period_s=0.1,
                ),
            )

        self.assertIsNone(result["failure"])
        self.assertGreaterEqual(len(formal_dispatches), 64)
        self.assertEqual(completions_before_refill, [1])

    def test_adaptive_supply_dispatches_without_completions_until_hard_limit(self):
        active = [None]

        class PendingFormalHandle:
            def __init__(self, executor, work, on_done=None):
                self.formal = False
                self.cancelled = False
                active[0] = self
                self.value = work(lambda call: None)
                active[0] = None

            def done(self):
                return not self.formal or self.cancelled

            def result(self):
                return self.value

            def cancel(self):
                self.cancelled = True
                return True

        def observe(identity):
            phase = identity.rsplit(":", 2)[-2]
            if phase in ("steady", "measurement", "support"):
                active[0].formal = True

        with patch.object(cp, "CancellableRequestHandle", PendingFormalHandle):
            result, events, _, _ = self.run_session(
                concurrency=2,
                rate=0,
                request_observer=observe,
                cfg=config(
                    family_schedule="tiered_rolling",
                    memory_reuse_ratio=0.5,
                    fill_host_max_ratio=0.8,
                    rate_control="adaptive_backlog",
                    request_rate_min=2,
                    request_rate_max=8,
                    backlog_low=1,
                    backlog_high=3,
                    max_outstanding=4,
                    cohort_size=2,
                    steady_s=0.1,
                    measure_s=1,
                    sample_period_s=0.1,
                    drain_timeout_s=1,
                ),
            )

        formal = [
            event
            for event in events
            if event["event"] == "dispatch"
            and event["phase"] in ("steady", "measurement", "support")
        ]
        self.assertEqual(result["failure"], "sender_saturated")
        self.assertEqual(len(formal), 4)
        self.assertEqual(
            [event["outstanding_before"] for event in formal], [0, 1, 2, 3]
        )
        self.assertEqual([event["effective_rate"] for event in formal], [8, 8, 5, 2])
        self.assertEqual([event["formal_sequence"] for event in formal], list(range(4)))
        saturation = [event for event in events if event["event"] == "sender_saturated"]
        self.assertEqual(len(saturation), 1)
        self.assertEqual(saturation[0]["outstanding"], 4)
        self.assertEqual(result["supply"]["max_outstanding"], 4)
        self.assertTrue(result["supply"]["saturated"])

    def test_steady_uses_elapsed_coverage_and_times_out(self):
        def alternate(evidence, count, clock):
            if count > 3 and count % 2:
                return cp.PressurePoll((), {"missing": True})
            return evidence

        result, _, _, _ = self.run_session(
            poll_transform=alternate,
            cfg=config(steady_timeout_s=3, steady_s=2, sample_period_s=0.5),
        )
        self.assertEqual(result["failure"], "pressure_not_reached")
        self.assertIsNone(result["measurement_start"])

    def test_measurement_pressure_loss_never_injects_repair_fill(self):
        def lose(evidence, count, clock):
            if clock() >= 4:
                sample = evidence.samples[0]
                evidence = replace(
                    evidence,
                    samples=(
                        replace(
                            sample,
                            pools=(
                                sample.pools[0],
                                replace(sample.pools[1], host_resident=0),
                            ),
                        ),
                    ),
                )
            return evidence

        result, events, _, _ = self.run_session(poll_transform=lose)
        self.assertFalse(result["pressure"]["valid"])
        self.assertTrue(any(e["event"] == "pressure_lost" for e in events))
        formal_start = next(
            e["attempted_dispatch_s"]
            for e in events
            if e["event"] == "dispatch" and e["phase"] == "steady"
        )
        self.assertFalse(
            any(
                e["event"] == "dispatch"
                and e["phase"] == "fill"
                and e["attempted_dispatch_s"] >= formal_start
                for e in events
            )
        )

    def test_service_restart_ends_epoch(self):
        def restart(evidence, count, clock):
            return (
                replace(
                    evidence, samples=(replace(evidence.samples[0], instance="new"),)
                )
                if count > 5
                else evidence
            )

        result, _, _, _ = self.run_session(poll_transform=restart)
        self.assertEqual(result["failure"], "pressure_epoch_invalid")

    def test_complete_changed_target_inventory_ends_epoch(self):
        def change(evidence, count, clock):
            if count > 5:
                sample = evidence.samples[0]
                return replace(
                    evidence,
                    samples=(
                        replace(
                            sample, groups=(replace(sample.groups[0], group_id=1),)
                        ),
                    ),
                )
            return evidence

        result, _, _, _ = self.run_session(
            poll_transform=change, cfg=config(steady_timeout_s=4)
        )
        self.assertEqual(result["failure"], "pressure_epoch_invalid")

    def test_support_finishes_before_cap(self):
        class Controller:
            def start(self, now):
                self.started = now

            def pending(self, now):
                return now < self.started + 4

            def stop(self, reason, now):
                self.stopped = now

        controller = Controller()
        result, events, _, _ = self.run_session(controller=controller)
        self.assertIsNone(result["failure"])
        self.assertAlmostEqual(controller.stopped, controller.started + 4, places=2)
        self.assertTrue(any(e.get("phase") == "support" for e in events))

    def test_controller_stop_failure_still_returns_terminal_record(self):
        class Controller:
            def start(self, now):
                pass

            def pending(self, now):
                return False

            def stop(self, reason, now):
                raise RuntimeError("stop failed")

        result, events, _, _ = self.run_session(controller=Controller())
        self.assertEqual(result["failure"], "profile_stop_failed")
        self.assertEqual(events[-1]["event"], "finished")

    def test_dispatch_exactly_at_t1_is_support_after_input_generation(self):
        class Controller:
            def start(self, now):
                self.end = now + 3

            def pending(self, now):
                return now < self.end + 1

            def stop(self, reason, now):
                pass

        controller = Controller()
        moved = []

        def delay(kind, sequence, clock):
            if kind == "formal" and hasattr(controller, "end") and not moved:
                clock.now = controller.end
                moved.append(True)

        result, events, _, _ = self.run_session(
            controller=controller, payload_hook=delay
        )
        dispatch = [e for e in events if e["event"] == "dispatch"]
        at_end = [
            e
            for e in dispatch
            if e["attempted_dispatch_s"] == result["measurement_end"]
        ]
        self.assertTrue(at_end)
        self.assertTrue(all(e["phase"] == "support" for e in at_end))

    def test_concurrent_preparation_preserves_transport_order_and_retires_turns(self):
        for scenario in (
            "order",
            "cutoff",
            "prepare_error",
            "dispatch_error",
            "cancel",
        ):
            with self.subTest(scenario=scenario), ThreadPoolExecutor(2) as executor:
                ready, release, response, next_sent = (Event() for _ in range(4))
                calls = []
                creating, created = Event(), Event()

                def payload(kind, job, sequence):
                    if sequence == 0:
                        if not release.wait(3):
                            raise RuntimeError("timed out preparing payload")
                        if scenario == "prepare_error":
                            raise ValueError("preparation failed")
                    else:
                        ready.set()
                    return {"input_ids": [sequence]}

                def request(payload, register, before):
                    sequence = payload["input_ids"][0]
                    identity = before(payload)
                    if sequence == 0 and scenario == "order":
                        creating.set()
                        if not created.wait(2):
                            raise RuntimeError("timed out creating transport")
                    if sequence == 0 and scenario == "dispatch_error":
                        raise RuntimeError("transport creation failed")
                    calls.append((sequence, identity))
                    register(SimpleNamespace(cancel=lambda: True))
                    if sequence == 0:
                        response.wait(3)
                    else:
                        next_sent.set()
                    return {"success": True}

                session = cp.PressureSession(
                    config(),
                    session_id="s",
                    limits=ExecutionLimits(128, 8, 256, 64, 16),
                    concurrency=2,
                    max_input_tokens=16,
                    seed_plan=seed_plan(),
                    startup_targets=(),
                    make_payload=payload,
                    request=request,
                    poll=None,
                    sink=lambda event: None,
                )
                session.executor = executor
                session.formal_origin = time.monotonic()
                session.state = PressurePhases(session.config, session.formal_origin)
                session.state.phase = "measurement"
                session.state.measurement_end = session.formal_origin + 30
                try:
                    self.assertTrue(session._submit("formal", 0, 0))
                    self.assertTrue(session._submit("formal", 0.1, 1))
                    self.assertTrue(ready.wait(2))
                    self.assertFalse(
                        next_sent.wait(0.1), "later payload passed the waiting arrival"
                    )
                    if scenario == "cutoff":
                        session.fail("cutoff")
                    if scenario == "cancel":
                        session._cancel_pending()
                    release.set()
                    if scenario == "order":
                        self.assertTrue(creating.wait(2))
                        self.assertFalse(next_sent.wait(0.1))
                        created.set()
                    if scenario not in ("cutoff", "cancel"):
                        self.assertTrue(
                            next_sent.wait(2), "turn held after initiation or failure"
                        )
                    response.set()
                    for handle in session.sender.pending.values():
                        try:
                            handle.result()
                        except (ValueError, RuntimeError):
                            pass
                    expected = {
                        "order": [0, 1],
                        "cutoff": [],
                        "cancel": [],
                        "prepare_error": [1],
                        "dispatch_error": [1],
                    }
                    self.assertEqual([seq for seq, _ in calls], expected[scenario])
                finally:
                    created.set()
                    release.set()
                    response.set()

    def test_formal_phase_recomputed_after_admission_or_hash_crosses_measurement(self):
        class Capture:
            end = None

            def start(self, now):
                self.end = now + 1

            def pending(self, now):
                return now < self.end + 0.5

            def stop(self, reason, now):
                pass

        original_admission = ExecutionLimits.concurrency_for
        original_hash = cp.hashlib.sha256
        for stage in ("admission", "hash"):
            with self.subTest(stage=stage):
                capture = Capture()
                pending, shifted, sent = [], [], []

                def payload(kind, sequence, clock):
                    if kind == "formal" and capture.end is not None and not shifted:
                        pending.append(clock)

                def cross_cutoff():
                    if pending:
                        pending.pop().now = capture.end
                        shifted.append(capture.end)

                def admission(limits, input_tokens):
                    admitted = original_admission(limits, input_tokens)
                    if stage == "admission":
                        cross_cutoff()
                    return admitted

                def digest(data):
                    hashed = original_hash(data)
                    if stage == "hash":
                        cross_cutoff()
                    return hashed

                with patch.object(
                    ExecutionLimits, "concurrency_for", admission
                ), patch.object(cp.hashlib, "sha256", digest):
                    result, events, _, _ = self.run_session(
                        seed_count=3,
                        rate=0,
                        cfg=self.hybrid_config(),
                        controller=capture,
                        poll_transform=self.hybrid_poll,
                        payload_hook=payload,
                        transport_observer=lambda identity, payload, now: sent.append(
                            (identity, payload["mock_memory_cache_request"], now)
                        ),
                    )
                self.assertIsNone(result["failure"])
                self.assertEqual(shifted, [result["measurement_end"]])
                at_cutoff = [
                    (identity, opt_in)
                    for identity, opt_in, at in sent
                    if at == capture.end
                ]
                self.assertTrue(at_cutoff)
                for identity, opt_in in at_cutoff:
                    self.assertIn(":support:", identity)
                    self.assertFalse(opt_in)
                    dispatch = next(
                        e
                        for e in events
                        if e["event"] == "dispatch" and e["request_id"] == identity
                    )
                    self.assertEqual(dispatch["attempted_dispatch_s"], capture.end)
                    self.assertEqual(dispatch["phase"], "support")
                    self.assertFalse(dispatch["metadata"]["mock_memory_cache_request"])

    def test_accepted_formal_dispatch_uses_transport_time_after_sink_delay(self):
        attempts, sent = [], {}

        def sink(event, clock):
            if (
                event["event"] == "dispatch"
                and event["phase"] == "measurement"
                and not attempts
            ):
                attempts.append((event, clock()))
                clock.sleep(0.25)

        def observe_transport(identity, payload, now):
            sent[identity] = now

        result, events, _, state = self.run_session(
            event_sink=sink, transport_observer=observe_transport
        )
        self.assertIsNone(result["failure"])
        ((attempt, attempted_at),) = attempts
        sent_at = sent[attempt["request_id"]]
        self.assertAlmostEqual(sent_at - attempted_at, 0.25)
        self.assertLess(sent_at, result["measurement_end"])
        completion = next(
            event
            for event in events
            if event["event"] == "completion" and event["job_id"] == attempt["job_id"]
        )
        self.assertTrue(completion["result"]["success"])
        self.assertEqual(completion["dispatched_s"], sent_at)
        self.assertAlmostEqual(
            completion["schedule_lag_s"],
            sent_at - state.formal_origin - attempt["scheduled_offset_s"],
        )
        self.assertEqual(attempt["attempted_dispatch_s"], attempted_at)
        self.assertNotIn("dispatched_s", attempt)
        self.assertNotIn("schedule_lag_s", attempt)

    def test_formal_sink_delay_rejects_tentative_dispatch_at_drain_cutoff(self):
        sent = []
        tentative = []
        cfg = config(measure_s=3, steady_s=1, sample_period_s=0.5)
        measurement_end = None

        def sink(event, clock):
            nonlocal measurement_end
            if event["event"] == "phase" and event["phase"] == "measurement":
                measurement_end = event["at_s"] + cfg.measure_s
            if (
                event["event"] == "dispatch"
                and event["phase"] == "measurement"
                and not tentative
            ):
                tentative.append(event)
                clock.now = measurement_end

        result, events, _, _ = self.run_session(
            cfg=cfg, event_sink=sink, request_observer=sent.append
        )
        self.assertEqual(len(tentative), 1)
        self.assertEqual(result["measurement_end"], measurement_end)
        self.assertNotIn(tentative[0]["request_id"], sent)
        completion = next(
            e
            for e in events
            if e["event"] == "completion" and e["job_id"] == tentative[0]["job_id"]
        )
        self.assertEqual(completion["result"], {"not_dispatched": True})
        self.assertFalse(completion["crossing"])
        self.assertNotIn("dispatched_s", completion)
        self.assertNotIn("schedule_lag_s", completion)
        self.assertTrue(
            any(e["event"] == "phase" and e["phase"] == "drain" for e in events)
        )

    def test_failed_poll_breaks_coverage_and_is_retained(self):
        def fail(evidence, count, clock):
            if 4 <= clock() <= 5:
                raise TimeoutError("poll timed out")
            return evidence

        result, events, _, _ = self.run_session(poll_transform=fail)
        self.assertFalse(result["pressure"]["valid"])
        self.assertTrue(any("error" in e.get("raw", {}) for e in events))

    def test_invalid_baseline_tolerance_sends_nothing_and_cleans_up(self):
        result, events, _, _ = self.run_session(cfg=config(gpu_tolerance_blocks=1000))
        self.assertEqual(result["failure"], "supervisor_failed")
        self.assertFalse(any(e["event"] == "dispatch" for e in events))

    def test_blocked_rpc_is_cancelled_at_drain_without_boundary_drain(self):
        self.run_blocked_rpc(seed_count=0)

    def test_blocked_seed_rpc_is_cancelled_and_reaped_before_finished(self):
        self.run_blocked_rpc(seed_count=2)

    def run_blocked_rpc(self, *, seed_count):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        events, cancels = [], []
        sequence = [0]

        def poll(timeout):
            now = time.monotonic()
            sequence[0] += 1
            group = GroupSample(0, "FULL", ("gpu",), "host", sequence[0], sequence[0])
            sample = RankSample(
                0,
                "service",
                sequence[0],
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 30, 0, 0.8, 0.9, 25),
                ),
                (group,),
            )
            return cp.PressurePoll((sample,), {})

        def request(payload, register, before):
            identity = before(payload)
            if ":fill:" not in identity:
                cancelled = Event()

                class Call:
                    def cancel(self):
                        cancels.append(time.monotonic())
                        cancelled.set()
                        return True

                register(Call())
                if not cancelled.wait(2):
                    raise AssertionError("missing bounded cancellation")
            return dict(success=True, input_len=16, output_len=1)

        startup_targets = select_targets(poll(0.01).samples, (0,))
        sequence[0] = 0
        session = cp.PressureSession(
            config(
                seed_timeout_s=0.03,
                sample_period_s=0.01,
                steady_s=0.02,
                measure_s=0.03,
                drain_timeout_s=0.03,
                activity_timeout_s=1,
            ),
            session_id="blocked",
            limits=ExecutionLimits(128, 2, 128, 64, 16),
            concurrency=2,
            max_input_tokens=16,
            seed_plan=seed_plan(seed_count, client_inflight=2),
            startup_targets=startup_targets,
            make_payload=lambda kind, i, formal_sequence: {"input_ids": [1] * 16},
            request=request,
            poll=poll,
            sink=events.append,
        )
        began = time.monotonic()
        result = session.run()
        self.assertLess(time.monotonic() - began, 1)
        self.assertEqual(
            result["failure"], "seed_timeout" if seed_count else "drain_timeout"
        )
        self.assertEqual(len(cancels), 2)
        if seed_count:
            self.assertEqual(result["seed"]["submitted"], 2)
            self.assertEqual(result["seed"]["cancelled"], 2)
            self.assertEqual(result["seed"]["completed"], 2)
            self.assertEqual(result["seed"]["pending"], 0)
            self.assertEqual(session.sender.pending, {})
            self.assertFalse(any(e.get("phase") == "fill" for e in events))
        else:
            self.assertTrue(all(t >= result["measurement_end"] for t in cancels))
        self.assertEqual(sum(e["event"] == "cancellation" for e in events), 2)
        self.assertTrue(
            any(e["event"] == "phase" and e["phase"] == "measurement" for e in events)
            == (not seed_count)
        )


class SeedLifecycleTest(unittest.TestCase):
    def run_seed(
        self,
        *,
        mode="rolling",
        count=20,
        cfg=None,
        payload_hook=None,
        request_observer=None,
    ):
        handles, reaped = [], []

        class Handle:
            def __init__(self, executor, work, on_done=None):
                self.family = len(handles)
                self.cancel_calls = 0
                self.result_value = work(lambda call: None)
                handles.append(self)

            def done(self):
                if self.family >= count:
                    return True
                if self.cancel_calls:
                    return mode != "reject_cancel"
                if mode in ("timeout", "reject_cancel", "cancel_raises"):
                    return False
                if mode == "failure":
                    return self.family == 0
                return self.family == 0 or len(handles) > 16

            def result(self):
                reaped.append(self.family)
                if self.family < count and (mode == "failure" or self.cancel_calls):
                    return dict(success=False, input_len=16, output_len=1)
                return self.result_value

            def cancel(self):
                self.cancel_calls += 1
                if mode == "cancel_raises":
                    raise RuntimeError("transport cancellation failed")
                return mode != "reject_cancel"

        def payload(kind, family, clock):
            if kind == "seed" and family == 16 and mode == "rolling":
                self.assertEqual(
                    reaped, [0], "replacement waited for the initial window"
                )
            if payload_hook:
                payload_hook(kind, family, clock)

        with patch.object(cp, "CancellableRequestHandle", Handle):
            result, events, batches, session = LifecycleTest().run_session(
                seed_count=count,
                concurrency=16,
                payload_hook=payload,
                request_observer=request_observer,
                cfg=cfg or config(seed_timeout_s=0.2, drain_timeout_s=0.1),
            )
        return result, events, session, handles

    def test_one_completion_replenishes_before_fifteen_initial_requests_drain(self):
        result, events, session, handles = self.run_seed()
        self.assertIsNone(result["failure"])
        self.assertEqual(session.max_seed_pending, 16)
        self.assertEqual(result["seed"]["planned"], 20)
        self.assertEqual(result["seed"]["submitted"], 20)
        self.assertEqual(result["seed"]["completed"], 20)
        self.assertEqual(result["seed"]["failed"], 0)
        self.assertEqual(result["seed"]["cancelled"], 0)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertEqual(result["seed"]["max_inflight"], 16)
        self.assertFalse(any(e.get("phase") == "padding" for e in events))
        self.assertEqual(events[-1]["seed"], result["seed"])
        phases = [e["phase"] for e in events if e["event"] == "phase"]
        self.assertEqual(phases, ["seed", "fill", "steady", "measurement", "drain"])

    def test_seed_timeout_cancels_every_accepted_handle_and_never_enters_fill(self):
        result, events, _, handles = self.run_seed(mode="timeout")
        self.assertEqual(result["failure"], "seed_timeout")
        self.assertEqual(result["seed"]["submitted"], 16)
        self.assertEqual(result["seed"]["completed"], 0)
        self.assertEqual(result["seed"]["cancelled"], 16)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertTrue(all(handle.cancel_calls == 1 for handle in handles))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))
        self.assertIsNone(result["measurement_start"])

    def test_seed_failure_stops_replacements_and_cancels_other_accepted_work(self):
        result, events, _, handles = self.run_seed(mode="failure")
        self.assertEqual(result["failure"], "request_failed")
        self.assertEqual(result["seed"]["submitted"], 16)
        self.assertEqual(result["seed"]["failed"], 1)
        self.assertEqual(result["seed"]["cancelled"], 15)
        self.assertEqual(result["seed"]["pending"], 0)
        self.assertTrue(all(handle.cancel_calls == 1 for handle in handles[1:]))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_rejected_or_raising_cancel_is_attempted_once_for_every_seed(self):
        for mode, pending in (("reject_cancel", 16), ("cancel_raises", 0)):
            with self.subTest(mode=mode):
                result, events, _, handles = self.run_seed(mode=mode)
                self.assertEqual(result["failure"], "seed_timeout")
                self.assertEqual(result["seed"]["cancelled"], 16)
                self.assertEqual(result["seed"]["pending"], pending)
                self.assertTrue(all(handle.cancel_calls == 1 for handle in handles))
                self.assertEqual(sum(e["event"] == "cancellation" for e in events), 16)
                self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_seed_budget_is_independent_of_fill_budget(self):
        def delay(kind, family, clock):
            if kind == "seed" and family == 0:
                clock.now += 2

        result, events, _, _ = self.run_seed(
            cfg=config(seed_timeout_s=5, fill_timeout_s=1, fill_no_progress_s=1),
            payload_hook=delay,
        )
        self.assertIsNone(result["failure"])
        self.assertGreaterEqual(result["seed"]["elapsed_s"], 2)
        fill = next(e for e in events if e["event"] == "phase" and e["phase"] == "fill")
        self.assertGreaterEqual(fill["at_s"], 3)

    def test_seed_input_preparation_cannot_dispatch_after_seed_deadline(self):
        def delay(kind, family, clock):
            if kind == "seed":
                clock.now += 1

        result, events, _, _ = self.run_seed(payload_hook=delay)
        self.assertEqual(result["failure"], "seed_timeout")
        self.assertFalse(any(e["event"] == "dispatch" for e in events))
        self.assertFalse(any(e.get("phase") == "fill" for e in events))

    def test_seed_dispatch_preparation_cannot_cross_transport_deadline(self):
        original_admission = ExecutionLimits.concurrency_for
        original_sha256 = cp.hashlib.sha256
        original_emit = cp.PressureSession.emit
        for stage in ("admission", "hash", "event"):
            with self.subTest(stage=stage):
                clocks, sends = [], []

                def payload(kind, family, clock):
                    if kind == "seed":
                        self.assertEqual(clock(), 1.0)
                        clocks.append(clock)

                def admission(limits, input_tokens):
                    result = original_admission(limits, input_tokens)
                    if stage == "admission" and clocks:
                        clocks[0].now = 2.0
                    return result

                def sha256(value):
                    result = original_sha256(value)
                    if stage == "hash" and clocks:
                        clocks[0].now = 2.0
                    return result

                def emit(session, event, **values):
                    original_emit(session, event, **values)
                    if stage == "event" and event == "dispatch":
                        clocks[0].now = 2.0

                with patch.object(
                    ExecutionLimits, "concurrency_for", admission
                ), patch.object(cp.hashlib, "sha256", sha256), patch.object(
                    cp.PressureSession, "emit", emit
                ):
                    result, events, session, handles = self.run_seed(
                        count=1, payload_hook=payload, request_observer=sends.append
                    )
                self.assertEqual(sends, [])
                self.assertEqual(result["failure"], "seed_timeout")
                self.assertEqual(result["seed"]["submitted"], 1)
                self.assertEqual(result["seed"]["completed"], 0)
                self.assertEqual(result["seed"]["failed"], 1)
                self.assertEqual(result["seed"]["pending"], 0)
                self.assertEqual(session.sender.pending, {})
                self.assertFalse(any(e.get("phase") == "fill" for e in events))
                if stage == "event":
                    completion = next(e for e in events if e["event"] == "completion")
                    self.assertEqual(completion["result"], {"not_dispatched": True})
                else:
                    self.assertFalse(any(e["event"] == "dispatch" for e in events))


class WarmupCutoffTest(unittest.TestCase):
    def run_cutoff(self):
        from rtp_llm.test.continuous_pressure.diagnostics import (
            GroupSample,
            PoolSample,
            RankSample,
        )

        clock = FakeClock()
        first_sent, crossed = Event(), Event()
        events, dispatched, cancelled, batches = [], [], [], []
        sample_sequence = [0]

        def poll(timeout):
            sample_sequence[0] += 1
            now = clock()
            sample = RankSample(
                0,
                "service",
                sample_sequence[0],
                now,
                now,
                now,
                (
                    PoolSample("gpu", 100, 90, 0, 0.8, 0.9),
                    PoolSample("host", 100, 0, 0, 0.8, 0.9, 0),
                ),
                (GroupSample(0, "FULL", ("gpu",), "host", 0, 0),),
            )
            return cp.PressurePoll((sample,), {})

        def payload(request_kind, job_id, formal_sequence):
            self.assertEqual(request_kind, "fill")
            if job_id == 1:
                if not first_sent.wait(2):
                    raise AssertionError("first request never actually dispatched")
                clock.now = 1.21
                crossed.set()
            return {"input_ids": [job_id + 1] * 16}

        def request(payload, register, before):
            identity = before(payload)
            dispatched.append(identity)
            if len(dispatched) == 1:
                first_sent.set()

            class Call:
                def cancel(self):
                    cancelled.append(identity)
                    return True

            register(Call())
            return dict(success=True, input_len=16, output_len=1)

        def sleep(seconds):
            # Only the second worker advances across the fill deadline. After
            # that, let cleanup progress even if the partial batch is blocked.
            if crossed.is_set():
                clock.now = round(clock.now + 0.01, 8)
            time.sleep(0.001)

        startup_targets = select_targets(poll(0.01).samples, (0,))
        sample_sequence[0] = 0
        session = cp.PressureSession(
            config(fill_timeout_s=0.2, drain_timeout_s=0.2, sample_period_s=0.01),
            session_id="partial",
            limits=ExecutionLimits(128, 2, 128, 64, 16),
            concurrency=2,
            max_input_tokens=16,
            seed_plan=seed_plan(client_inflight=2),
            startup_targets=startup_targets,
            make_payload=payload,
            request=request,
            poll=poll,
            sink=events.append,
            clock=clock,
            sleep=sleep,
        )
        result = session.run()
        self.assertTrue(first_sent.is_set())
        self.assertTrue(crossed.is_set())
        self.assertEqual(result["failure"], "pressure_not_reached")
        return events, dispatched, cancelled, batches, session

    def test_fifo_never_dispatches_warmup_padding(self):
        _, dispatched, cancelled, batches, _ = self.run_cutoff()
        self.assertEqual(dispatched, ["partial:fill:0"])
        self.assertEqual(batches, [])
        self.assertEqual(cancelled, [])


class BoundedSenderTest(unittest.TestCase):
    def test_does_not_construct_or_submit_when_full(self):
        sender = BoundedSender(2)
        first, second, third = Future(), Future(), Future()
        calls = []
        self.assertTrue(sender.submit(0, lambda: first))
        self.assertTrue(sender.submit(1, lambda: second))
        self.assertFalse(sender.submit(2, lambda: calls.append(2) or third))
        self.assertEqual(calls, [])
        first.set_result({"success": True})
        self.assertEqual(sender.collect(), [(0, {"success": True})])
        self.assertTrue(sender.submit(2, lambda: third))
        sender.cancel_all()
        self.assertTrue(second.cancelled())
        self.assertTrue(third.cancelled())

    def test_failure_becomes_record_and_releases_slot(self):
        sender = BoundedSender(1)
        failed = Future()
        failed.set_exception(RuntimeError("broken"))
        sender.submit(1, lambda: failed)
        self.assertEqual(
            sender.collect(),
            [(1, {"success": False, "error": "RuntimeError('broken')"})],
        )
        self.assertEqual(sender.collect(), [])
        self.assertTrue(sender.submit(2, Future))

    def test_invalid_limit_and_duplicate_id(self):
        with self.assertRaises(ValueError):
            BoundedSender(0)
        sender = BoundedSender(1)
        sender.submit(0, Future)
        with self.assertRaises(ValueError):
            sender.submit(0, Future)

    def test_backpressure_does_not_advance_schedule(self):
        clock = ArrivalClock(2, False, 0)
        sender = BoundedSender(1)
        first = Future()
        due = clock.next_offset()
        sender.submit(0, lambda: first)
        due = clock.next_offset()
        self.assertFalse(sender.submit(1, Future))
        self.assertEqual(due, 0.5)
        first.set_result({})
        sender.collect()
        self.assertTrue(sender.submit(1, Future))
        self.assertEqual(clock.next_offset(), 1.0)


class ArrivalClockTest(unittest.TestCase):
    def test_burst_and_constant_offsets(self):
        burst = ArrivalClock(0, False, 0)
        regular = ArrivalClock(2, False, 0)
        self.assertEqual([burst.next_offset() for _ in range(3)], [0, 0, 0])
        self.assertEqual([regular.next_offset() for _ in range(3)], [0, 0.5, 1])

    def test_poisson_replays_seed(self):
        first, second = ArrivalClock(2, True, 9), ArrivalClock(2, True, 9)
        offsets = [first.next_offset() for _ in range(10)]
        self.assertEqual(offsets, [second.next_offset() for _ in range(10)])
        self.assertTrue(all(b > a for a, b in zip(offsets, offsets[1:])))


class CancellableRequestHandleTest(unittest.TestCase):
    def test_completion_wakes_supervisor(self):
        woke = Event()
        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(
                executor, lambda register: {}, on_done=woke.set
            )
            self.assertTrue(woke.wait(2))
            self.assertTrue(handle.done())

    def test_cancels_running_rpc(self):
        started, cancelled = Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            register_call(Call())
            started.set()
            if not cancelled.wait(2):
                raise AssertionError("RPC was not cancelled")
            return {"success": False}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(started.wait(2))
            self.assertTrue(handle.cancel())
            self.assertTrue(cancelled.is_set())
            self.assertEqual(handle.result(), {"success": False})
            self.assertTrue(handle.done())

    def test_cancel_queued_worker_does_not_start_request(self):
        entered, release = Event(), Event()
        calls = []

        def occupy_worker():
            entered.set()
            release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            blocker = executor.submit(occupy_worker)
            self.assertTrue(entered.wait(2))
            handle = CancellableRequestHandle(
                executor, lambda register: calls.append("started") or {}
            )
            self.assertTrue(handle.cancel())
            release.set()
            blocker.result()
            self.assertTrue(handle.done())
            self.assertEqual(calls, [])

    def test_cancel_before_rpc_registration_cancels_late_call(self):
        entered, proceed, cancelled = Event(), Event(), Event()

        class Call:
            def cancel(self):
                cancelled.set()
                return True

        def request(register_call):
            entered.set()
            proceed.wait(2)
            register_call(Call())
            return {}

        with ThreadPoolExecutor(max_workers=1) as executor:
            handle = CancellableRequestHandle(executor, request)
            self.assertTrue(entered.wait(2))
            handle.cancel()
            proceed.set()
            self.assertEqual(handle.result(), {})
            self.assertTrue(cancelled.is_set())


class FormalConfigTest(unittest.TestCase):
    def test_explicit_smoke_is_ordered_and_requires_a_boolean(self):
        values = dict(
            manual_seed_count=2,
            client_inflight=2,
            test_block_num=128,
            fill_target_tokens=16384,
        )
        self.assertEqual(
            formal_config_reasons(**values, explicit_smoke=True),
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
                "explicit_smoke",
            ),
        )
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                formal_config_reasons(**values, explicit_smoke=value)

    def test_constants_and_monotonic_reasons(self):
        self.assertEqual(MIN_REUSE_TOKENS, 32768)
        self.assertEqual(FORMAL_FILL_TARGET_TOKENS, 102400)
        self.assertEqual(MIN_FORMAL_CLIENT_INFLIGHT, 16)
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=None,
                client_inflight=16,
                test_block_num=0,
                fill_target_tokens=102400,
            ),
            (),
        )
        self.assertEqual(
            formal_config_reasons(
                manual_seed_count=2,
                client_inflight=2,
                test_block_num=128,
                fill_target_tokens=16384,
            ),
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
            ),
        )

    def test_invalid_numeric_inputs_are_rejected(self):
        valid = dict(
            manual_seed_count=None,
            client_inflight=16,
            test_block_num=0,
            fill_target_tokens=102400,
        )
        for name, invalid_values in (
            ("manual_seed_count", [True, False, 0, -1, 1.5]),
            ("client_inflight", [True, False, 0, -1, 1.5]),
            ("test_block_num", [True, False, -1, 1.5]),
            ("fill_target_tokens", [True, False, 0, -1, 1.5]),
        ):
            for value in invalid_values:
                with self.subTest(name=name, value=value), self.assertRaises(
                    ValueError
                ):
                    formal_config_reasons(**(valid | {name: value}))


def owner(rank_id=0, *, capacity=127, available=256, block_tokens=512):
    return RankSample(
        rank_id,
        f"instance-{rank_id}",
        1,
        1.0,
        1.0,
        1.1,
        (
            PoolSample(
                "gpu",
                capacity,
                0,
                0,
                0.5,
                0.9,
                pool_name="gpu",
                tier=1,
                block_size_bytes=4096,
            ),
            PoolSample(
                "host",
                4096,
                0,
                0,
                0,
                0,
                pool_name="host",
                tier=2,
                block_size_bytes=4096,
                available=available,
            ),
        ),
        (
            GroupSample(
                0,
                "FULL",
                ("gpu",),
                "host",
                0,
                0,
                ("full",),
                geometry=(block_tokens, 4096, 4096, False),
            ),
        ),
    )


def shared_samples():
    declaration = PressureOwnership(2, 0, (0, 1), 2)
    first = replace(owner(), ownership=declaration)
    # Participant counters can differ. Only owner counters are authoritative.
    peer = replace(
        first,
        rank=1,
        instance="instance-1",
        pools=(
            first.pools[0],
            replace(first.pools[1], available=0),
        ),
    )
    return first, peer


class SeedPlannerTest(unittest.TestCase):
    def plan(self, samples=None, targets=None, **overrides):
        samples = samples if samples is not None else shared_samples()
        targets = (
            targets
            if targets is not None
            else select_targets(samples, tuple(sample.rank for sample in samples))
        )
        settings = dict(
            cases=(dict(input_len=36864, cache_len=32768, batch_size=1),),
            cache_alignment=4096,
            commit_tail_tokens=4096,
            client_inflight=16,
            manual_seed_count=None,
            fill_target_tokens=102400,
            test_block_num=0,
        )
        return build_seed_plan(samples, targets, **(settings | overrides))

    def test_cp4_global_owner_geometry_is_used_once_for_physical_blocks(self):
        first = owner(capacity=22584, available=9838, block_tokens=2048)
        first = replace(
            first,
            pools=(
                replace(first.pools[0], low_ratio=0.82),
                replace(first.pools[1], capacity=9838),
            ),
            ownership=PressureOwnership(2, 0, (0, 1, 2, 3), 2),
        )
        samples = tuple(
            replace(first, rank=rank, instance=f"instance-{rank}") for rank in range(4)
        )
        plan = self.plan(
            samples=samples,
            cases=(dict(input_len=69632, cache_len=65536, batch_size=1),),
        )
        group = plan.groups[0]
        self.assertEqual(group.seed_blocks, 32)
        self.assertEqual(group.gpu_family_capacity, 705)
        self.assertEqual(group.gpu_low_families, 578)
        self.assertEqual(group.host_families, 307)
        self.assertEqual((plan.seed_count, plan.global_maximum_families), (706, 885))
        self.assertEqual(group.device_pools[0].capacity_blocks, 22584)
        self.assertEqual(group.device_pools[0].block_tokens, 2048)
        self.assertEqual(group.device_pools[0].geometry_source, "owner_group")

    def test_shared_owner_capacity_is_counted_once_and_auditable(self):
        plan = self.plan()
        self.assertIsInstance(plan, SeedPlan)
        self.assertEqual(plan.reusable_tokens, 32768)
        self.assertEqual(plan.seed_input_tokens, 36864)
        self.assertEqual(len(plan.groups), 1)
        group = plan.groups[0]
        self.assertEqual(group.seed_blocks, 64)
        self.assertEqual(group.gpu_family_capacity, 1)
        self.assertEqual(group.gpu_low_families, 0)
        self.assertEqual(group.host_families, 4)
        self.assertEqual(group.minimum_families, 2)
        self.assertEqual(group.maximum_families, 4)
        self.assertEqual(plan.seed_count, 2)
        self.assertEqual(plan.source, "automatic")
        self.assertTrue(plan.formal_config_valid)
        evidence = asdict(plan)
        self.assertEqual(evidence["cache_alignment"], 4096)
        self.assertEqual(evidence["test_block_num"], 0)
        self.assertEqual(
            evidence["groups"][0]["participants"],
            ((0, "instance-0"), (1, "instance-1")),
        )
        pool = evidence["groups"][0]["device_pools"][0]
        self.assertEqual(pool["capacity_blocks"], 127)  # already allocatable
        self.assertEqual(pool["low_blocks"], 63)
        self.assertEqual(pool["block_tokens"], 512)
        self.assertEqual(pool["low_ratio"], 0.5)
        self.assertEqual(pool["high_ratio"], 0.9)
        self.assertEqual(pool["geometry_source"], "owner_group")
        with self.assertRaises(FrozenInstanceError):
            plan.seed_count = 9

    def test_duplicate_participant_views_and_physical_records_do_not_inflate(self):
        samples = shared_samples()
        samples = (
            replace(samples[0], pools=samples[0].pools + samples[0].pools),
            samples[1],
        )
        # Freeze against the inventory actually presented, including duplicate records.
        targets = select_targets(samples, (0, 1))
        plan = self.plan(samples, targets + targets)
        self.assertEqual(plan.seed_count, 2)
        self.assertEqual(len(plan.groups), 1)
        self.assertEqual(len(plan.groups[0].device_pools), 1)

    def test_zero_live_device_capacity_cannot_qualify_for_formal_pressure(self):
        with self.assertRaisesRegex(
            ValueError, r"owner_rank=0.*owner_group_id=0.*positive.*capacity"
        ):
            self.plan((owner(capacity=0),))

    def test_partial_blocks_round_up_for_both_device_and_host(self):
        plan = self.plan((owner(capacity=100, available=99, block_tokens=1000),))
        self.assertEqual(plan.groups[0].seed_blocks, 33)
        self.assertEqual(plan.groups[0].host_pool.seed_blocks, 33)
        self.assertEqual(plan.groups[0].gpu_family_capacity, 3)
        self.assertEqual(plan.groups[0].host_families, 3)
        self.assertEqual(plan.seed_count, 4)

    def test_distinct_full_groups_on_one_owner_remain_required(self):
        sample = owner(available=2048)
        second = owner(capacity=255, available=1024, block_tokens=1024)
        extra_pools = tuple(
            replace(pool, pool_id=pool.pool_id + "-2") for pool in second.pools
        )
        extra_group = replace(
            second.groups[0],
            group_id=7,
            device_pool_ids=("gpu-2",),
            host_pool_id="host-2",
            group_tags=("second",),
        )
        sample = replace(
            sample,
            pools=sample.pools + extra_pools,
            groups=sample.groups + (extra_group,),
        )
        plan = self.plan((sample,))
        self.assertEqual([g.owner_group_id for g in plan.groups], [0, 7])
        self.assertEqual(plan.seed_count, 8)
        self.assertEqual(plan.global_minimum_families, 8)
        self.assertEqual(plan.global_maximum_families, 32)

    def test_exact_gpu_capacity_needs_one_additional_family(self):
        plan = self.plan((owner(capacity=128),))
        self.assertEqual(plan.groups[0].gpu_family_capacity, 2)
        self.assertEqual(plan.seed_count, 3)

    def test_global_count_uses_largest_minimum_and_each_group_geometry(self):
        first = owner(available=2048)
        second = owner(1, capacity=255, available=1024, block_tokens=1024)
        plan = self.plan((second, first))
        self.assertEqual([g.owner_rank for g in plan.groups], [0, 1])
        self.assertEqual([g.seed_blocks for g in plan.groups], [64, 32])
        self.assertEqual([g.minimum_families for g in plan.groups], [2, 8])
        self.assertEqual([g.host_pool.seed_blocks for g in plan.groups], [64, 32])
        self.assertEqual(plan.seed_count, 8)

    def test_multiple_device_pools_use_bottleneck_capacity_and_low_watermark(self):
        sample = owner(capacity=255)
        extra = replace(
            sample.pools[0], pool_id="gpu-2", pool_name="gpu-2", capacity=128
        )
        sample = replace(
            sample,
            pools=sample.pools + (extra,),
            groups=(replace(sample.groups[0], device_pool_ids=("gpu", "gpu-2")),),
        )
        plan = self.plan((sample,))
        self.assertEqual(plan.groups[0].gpu_family_capacity, 2)
        self.assertEqual(plan.groups[0].gpu_low_families, 1)
        self.assertEqual(plan.seed_count, 3)

    def test_empty_interval_reports_constraining_owner_and_both_bounds(self):
        with self.assertRaisesRegex(
            ValueError, r"owner_rank=0.*owner_group_id=0.*minimum=2.*maximum=0"
        ):
            self.plan((owner(available=63),))
        with self.assertRaisesRegex(ValueError, r"lower=8.*upper=4"):
            self.plan(
                (owner(), owner(1, capacity=255, available=1024, block_tokens=1024))
            )

    def test_manual_count_is_smoke_and_still_cannot_exceed_retention(self):
        plan = self.plan(manual_seed_count=1)
        self.assertEqual(plan.source, "manual")
        self.assertEqual(plan.seed_count, 1)
        self.assertEqual(plan.groups[0].minimum_families, 2)
        self.assertFalse(plan.formal_config_valid)
        self.assertEqual(plan.nonformal_reasons, ("manual_seed_count",))
        with self.assertRaisesRegex(ValueError, r"manual.*5.*upper=4"):
            self.plan(manual_seed_count=5)

    def test_32k_formal_grid_boundary_and_independent_smoke_reasons(self):
        case = dict(input_len=36864, cache_len=32767, batch_size=1)
        with self.assertRaisesRegex(ValueError, "grid.*32768"):
            self.plan(cases=(case,))
        plan = self.plan(
            cases=(case,),
            manual_seed_count=1,
            client_inflight=2,
            test_block_num=128,
            fill_target_tokens=16385,
        )
        self.assertEqual(plan.reusable_tokens, 32768)
        self.assertEqual(plan.fill_input_tokens, 20480)
        self.assertEqual(
            plan.nonformal_reasons,
            (
                "manual_seed_count",
                "client_inflight_below_16",
                "synthetic_test_block_capacity",
                "nonstandard_fill_target",
                "grid_reuse_below_32768",
            ),
        )

    def test_reuse_and_fill_align_up_without_counting_commit_tail(self):
        plan = self.plan(
            cases=(dict(input_len=40000, cache_len=33000, batch_size=1),),
            cache_alignment=8192,
            commit_tail_tokens=512,
        )
        self.assertEqual(plan.grid_reuse_tokens, 33000)
        self.assertEqual(plan.reusable_tokens, 40960)
        self.assertEqual(plan.seed_input_tokens, 41472)
        self.assertEqual(plan.groups[0].seed_blocks, 80)
        self.assertEqual(plan.fill_target_tokens, 102400)
        self.assertEqual(plan.fill_input_tokens, 106496)

    def test_invalid_or_missing_live_evidence_is_rejected(self):
        samples = shared_samples()
        targets = select_targets(samples, (0, 1))
        for bad_samples, bad_targets in (
            ((), targets),
            (samples, ()),
            (samples[:1], targets),
            (
                (
                    samples[0],
                    replace(
                        samples[1],
                        groups=(
                            replace(
                                samples[1].groups[0], geometry=(1024, 4096, 4096, False)
                            ),
                        ),
                    ),
                ),
                targets,
            ),
            ((replace(samples[0], instance="restarted"), samples[1]), targets),
        ):
            with self.subTest(samples=bad_samples), self.assertRaises(ValueError):
                self.plan(bad_samples, bad_targets)
        for tokens in (0, -1):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(
                ValueError, "block_tokens"
            ):
                self.plan((owner(block_tokens=tokens),))

    def test_align_and_planner_input_validation(self):
        self.assertEqual(align_up(32768, 4096), 32768)
        self.assertEqual(align_up(33000, 4096), 36864)
        for value in (0, -1, True, 1.5):
            for args in ((value, 4096), (32768, value)):
                with self.subTest(args=args), self.assertRaises(ValueError):
                    align_up(*args)
        for name, invalid in (
            ("cache_alignment", 0),
            ("commit_tail_tokens", -1),
            ("commit_tail_tokens", True),
            ("cases", ()),
            ("cases", (dict(cache_len=-1),)),
            ("manual_seed_count", True),
            ("manual_seed_count", 0),
        ):
            with self.subTest(name=name, invalid=invalid), self.assertRaises(
                ValueError
            ):
                self.plan(**{name: invalid})


if __name__ == "__main__":
    unittest.main()
