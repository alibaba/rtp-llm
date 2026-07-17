from __future__ import annotations

import unittest

from rtp_llm.kv_cache_subscriber.config import SubscriberConfig
from rtp_llm.kv_cache_subscriber.models import CacheDiff, CacheSnapshot
from rtp_llm.kv_cache_subscriber.service import SubscriberService


def _config(*, reset_on_start: bool = True) -> SubscriberConfig:
    return SubscriberConfig(
        rtp_endpoints=("127.0.0.1:8089",),
        rtp_rpc_timeout_s=1.0,
        poll_interval_s=1.0,
        deletion_confirmations=2,
        engine_failure_threshold=2,
        full_refresh_interval_s=300.0,
        kvcm_url="http://kvcm.test:6382",
        kvcm_request_timeout_s=5.0,
        kvcm_heartbeat_interval_s=1.0,
        kvcm_report_batch_size=1000,
        instance_id="instance-a",
        instance_group="group-a",
        host_ip_port="10.0.0.8:8088",
        storage_type="ST_EVENT_REPORT",
        medium="hbm",
        reset_on_start=reset_on_start,
        log_level="INFO",
    )


class _FakeSource:
    async def fetch_snapshot(self) -> CacheSnapshot:
        raise NotImplementedError

    async def close(self) -> None:
        return None


class _FakeReporter:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.fail_next_diff = False

    async def start(self, block_size: int) -> None:
        self.calls.append(("start", block_size))

    async def register_node(self) -> None:
        self.calls.append("register_node")

    async def report_host_down(self) -> None:
        self.calls.append("host_down")

    async def report_heartbeat(self) -> None:
        self.calls.append("heartbeat")

    async def report_diff(self, diff: CacheDiff, block_size: int) -> None:
        self.calls.append(("diff", diff, block_size))
        if self.fail_next_diff:
            self.fail_next_diff = False
            raise RuntimeError("KVCM unavailable")

    async def close(self) -> None:
        self.calls.append("close")


class SubscriberServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_first_snapshot_resets_stale_node_then_adds_every_key(self) -> None:
        reporter = _FakeReporter()
        service = SubscriberService(
            _config(reset_on_start=True),
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
        reporter.fail_next_diff = True
        service = SubscriberService(
            _config(reset_on_start=False),
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
            _config(reset_on_start=False),
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
            _config(reset_on_start=False),
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


if __name__ == "__main__":
    unittest.main()
