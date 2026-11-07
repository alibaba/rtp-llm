from __future__ import annotations

import unittest

from rtp_llm.kv_cache_subscriber.models import CacheDiff, CacheSnapshot
from rtp_llm.kv_cache_subscriber.service import SubscriberService
from rtp_llm.kv_cache_subscriber.test_utils import make_config


class _FakeSource:
    async def fetch_snapshot(self) -> CacheSnapshot:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class _FakeReporter:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.diff_errors: list[Exception] = []
        self.host_down_errors: list[Exception] = []
        self.heartbeat_errors: list[Exception] = []

    async def start(self, block_size: int) -> None:
        self.calls.append(("start", block_size))

    async def register_node(self) -> None:
        self.calls.append("register_node")

    async def report_host_down(self) -> None:
        self.calls.append("host_down")
        if self.host_down_errors:
            raise self.host_down_errors.pop(0)

    async def report_heartbeat(self) -> None:
        self.calls.append("heartbeat")
        if self.heartbeat_errors:
            raise self.heartbeat_errors.pop(0)

    async def report_diff(self, diff: CacheDiff, block_size: int) -> None:
        self.calls.append(("diff", diff, block_size))
        if self.diff_errors:
            raise self.diff_errors.pop(0)

    async def close(self) -> None:
        self.calls.append("close")


class SubscriberServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_snapshot_resets_stale_node_then_adds_every_key(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=True),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )

        diff = await service.process_snapshot(
            CacheSnapshot(frozenset({2, 1}), block_size=16, version=5)
        )

        self.assertEqual(diff.added, (1, 2))
        self.assertEqual(
            reporter.calls[:3],
            [("start", 16), "host_down", "register_node"],
        )
        self.assertEqual(service.tracker.acknowledged_keys, frozenset({1, 2}))

    async def test_failed_report_does_not_advance_baseline(self) -> None:
        reporter = _FakeReporter()
        reporter.diff_errors.append(RuntimeError("KVCM unavailable"))
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        snapshot = CacheSnapshot(frozenset({4, 5}), block_size=16, version=1)

        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await service.process_snapshot(snapshot)
        self.assertEqual(service.tracker.acknowledged_keys, frozenset())

        retry = await service.process_snapshot(snapshot)
        self.assertEqual(retry.added, (4, 5))
        self.assertEqual(service.tracker.acknowledged_keys, frozenset({4, 5}))

    async def test_engine_failure_threshold_reports_host_down_once(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=1)
        )

        self.assertFalse(await service.handle_source_failure())
        self.assertTrue(await service.handle_source_failure())
        self.assertFalse(await service.handle_source_failure())

        self.assertEqual(reporter.calls.count("host_down"), 1)
        self.assertFalse(service.node_registered)
        self.assertEqual(service.tracker.acknowledged_keys, frozenset())

    async def test_recovery_registers_node_and_replays_full_snapshot(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        snapshot = CacheSnapshot(frozenset({1, 2}), block_size=16, version=1)
        await service.process_snapshot(snapshot)
        await service.handle_source_failure()
        await service.handle_source_failure()

        recovered = await service.process_snapshot(snapshot)

        self.assertEqual(recovered.added, (1, 2))
        self.assertEqual(reporter.calls.count("register_node"), 2)

    async def test_delete_is_reported_only_after_confirmed_missing_snapshots(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1, 2}), block_size=16, version=1)
        )

        first_missing = await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=2)
        )
        second_missing = await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=3)
        )

        self.assertTrue(first_missing.empty)
        self.assertEqual(second_missing.removed, (2,))
        self.assertEqual(service.tracker.acknowledged_keys, frozenset({1}))

    async def test_periodic_refresh_replays_all_acknowledged_keys(self) -> None:
        now = [10.0]
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False, full_refresh_interval_s=30.0),
            _FakeSource(),
            reporter,
            clock=lambda: now[0],
        )
        snapshot = CacheSnapshot(frozenset({1, 2}), block_size=16, version=1)
        await service.process_snapshot(snapshot)

        now[0] = 39.0
        before_deadline = await service.process_snapshot(snapshot)
        now[0] = 40.0
        at_deadline = await service.process_snapshot(snapshot)

        self.assertTrue(before_deadline.empty)
        self.assertEqual(at_deadline.added, (1, 2))

    async def test_block_size_change_is_rejected_without_advancing_state(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=1)
        )

        with self.assertRaisesRegex(RuntimeError, "block size changed"):
            await service.process_snapshot(
                CacheSnapshot(frozenset({1, 2}), block_size=32, version=2)
            )

        self.assertEqual(service.tracker.acknowledged_keys, frozenset({1}))
        self.assertEqual(reporter.calls.count(("start", 16)), 1)

    async def test_node_not_registered_error_forces_registration_and_full_replay(
        self,
    ) -> None:
        class NodeNotRegisteredError(RuntimeError):
            code = "NODE_NOT_REGISTERED"

        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=1)
        )
        reporter.diff_errors.append(NodeNotRegisteredError("node disappeared"))

        with self.assertRaises(NodeNotRegisteredError):
            await service.process_snapshot(
                CacheSnapshot(frozenset({1, 2}), block_size=16, version=2)
            )
        self.assertFalse(service.node_registered)

        recovered = await service.process_snapshot(
            CacheSnapshot(frozenset({1, 2}), block_size=16, version=2)
        )
        self.assertEqual(recovered.added, (1, 2))
        self.assertEqual(reporter.calls.count("register_node"), 2)

    async def test_failed_host_down_keeps_acknowledged_state_for_retry(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False),
            _FakeSource(),
            reporter,
            clock=lambda: 10.0,
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=1)
        )
        reporter.host_down_errors.append(RuntimeError("KVCM unavailable"))

        self.assertFalse(await service.handle_source_failure())
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await service.handle_source_failure()

        self.assertTrue(service.node_registered)
        self.assertEqual(service.tracker.acknowledged_keys, frozenset({1}))
        self.assertTrue(await service.handle_source_failure())
        self.assertEqual(service.tracker.acknowledged_keys, frozenset())

    async def test_heartbeat_is_throttled_and_lost_registration_closes_gate(
        self,
    ) -> None:
        class NodeNotRegisteredError(RuntimeError):
            code = "NODE_NOT_REGISTERED"

        now = [10.0]
        reporter = _FakeReporter()
        service = SubscriberService(
            make_config(reset_on_start=False, kvcm_heartbeat_interval_s=2.0),
            _FakeSource(),
            reporter,
            clock=lambda: now[0],
        )
        await service.process_snapshot(
            CacheSnapshot(frozenset({1}), block_size=16, version=1)
        )

        await service._maybe_heartbeat()
        await service._maybe_heartbeat()
        self.assertEqual(reporter.calls.count("heartbeat"), 1)

        now[0] = 12.0
        reporter.heartbeat_errors.append(NodeNotRegisteredError("node disappeared"))
        await service._maybe_heartbeat()
        self.assertFalse(service.node_registered)


if __name__ == "__main__":
    unittest.main()
