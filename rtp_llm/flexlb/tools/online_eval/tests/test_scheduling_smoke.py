from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ONLINE_EVAL_DIR = Path(__file__).resolve().parents[1]
if str(ONLINE_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(ONLINE_EVAL_DIR))

from scheduling_smoke import SchedulingSmokeTest


class _SchedulingSmokeProbe(SchedulingSmokeTest):
    def __init__(self, prefill_addr: str) -> None:
        self.prefill_addr = prefill_addr
        self.requested_role = None
        self.stream_started = False

    async def _schedule_auto(self, request_id: int, **kwargs):
        return SimpleNamespace(
            code=200,
            success=True,
            error_message="",
            enqueued_by_master=True,
        )

    def _role_addr(self, response, role: str) -> str:
        self.requested_role = role
        return self.prefill_addr

    async def _start_stream(self, response, request_id: int, input_pb=None):
        self.stream_started = True
        return object()

    async def _consume_stream(self, stream, snap) -> None:
        snap.completed = True

    async def _wait_for_stream_end(self, task, timeout_s: float) -> bool:
        await task
        return True


class _PendingFilterProbe(SchedulingSmokeTest):
    def __init__(
        self, routed_names: list[str], *, injection_succeeds: bool = True
    ) -> None:
        self._request_counter = 20000
        self._routed_names = iter(routed_names)
        self._injection_succeeds = injection_succeeds
        self.queue_depth_updates: list[tuple[str, int]] = []

    async def _snapshot_by_name(self) -> dict[str, dict]:
        return {
            "prefill-0": {"role": "prefill"},
            "prefill-1": {"role": "prefill"},
        }

    async def _set_queue_depth(self, engine_name: str, queue_depth: int) -> bool:
        self.queue_depth_updates.append((engine_name, queue_depth))
        return self._injection_succeeds or queue_depth == 0

    async def _run_one_request(self, rid: int, **kwargs) -> tuple[str, str | None]:
        return f"{next(self._routed_names)}:55151", None

    async def _addr_to_name(self) -> dict[str, str]:
        return {
            "prefill-0:55151": "prefill-0",
            "prefill-1:55151": "prefill-1",
        }


class SchedulingSmokeRoleAddressTest(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_response_role_is_matched_by_protocol_name(self) -> None:
        smoke = _SchedulingSmokeProbe("127.0.0.1:55151")

        address, error = await smoke._run_one_request(20001)

        self.assertEqual("PREFILL", smoke.requested_role)
        self.assertEqual("127.0.0.1:55151", address)
        self.assertIsNone(error)
        self.assertTrue(smoke.stream_started)

    async def test_missing_prefill_address_fails_before_stream_start(self) -> None:
        smoke = _SchedulingSmokeProbe("")

        address, error = await smoke._run_one_request(20002)

        self.assertEqual("", address)
        self.assertEqual("schedule response has no PREFILL address", error)
        self.assertFalse(smoke.stream_started)


class SchedulingSmokePendingFilterTest(unittest.IsolatedAsyncioTestCase):
    async def test_s9_passes_only_when_hot_endpoint_is_excluded(self) -> None:
        smoke = _PendingFilterProbe(
            [
                "prefill-0",  # seed cache-affinity leader
                "prefill-0",  # prove the pre-injection affinity baseline
                "prefill-0",  # status has not propagated yet
                "prefill-1",
                "prefill-1",
                "prefill-1",  # three consecutive exclusion confirmations
                *(["prefill-1"] * 5),
            ]
        )

        with patch("scheduling_smoke.asyncio.sleep", new=AsyncMock()) as sleep:
            result = await smoke.test_pending_hard_filter()

        self.assertTrue(result.passed)
        self.assertEqual("S9: pending_hard_filter", result.name)
        self.assertIn("target=prefill-0(0)", result.detail)
        self.assertEqual(
            [("prefill-0", 200_000), ("prefill-0", 0)],
            smoke.queue_depth_updates,
        )
        self.assertGreaterEqual(sleep.await_count, 2)

    async def test_s9_fails_when_any_request_reaches_hot_endpoint(self) -> None:
        smoke = _PendingFilterProbe(
            [
                "prefill-0",
                "prefill-0",
                "prefill-1",
                "prefill-1",
                "prefill-1",
                "prefill-0",
                *(["prefill-1"] * 4),
            ]
        )

        with patch("scheduling_smoke.asyncio.sleep", new=AsyncMock()):
            result = await smoke.test_pending_hard_filter()

        self.assertFalse(result.passed)
        self.assertIn("target=prefill-0(1)", result.detail)
        self.assertEqual(
            [("prefill-0", 200_000), ("prefill-0", 0)],
            smoke.queue_depth_updates,
        )

    async def test_s9_fails_closed_when_queue_depth_injection_fails(self) -> None:
        smoke = _PendingFilterProbe(
            ["prefill-0", "prefill-0"], injection_succeeds=False
        )

        with patch("scheduling_smoke.asyncio.sleep", new=AsyncMock()):
            result = await smoke.test_pending_hard_filter()

        self.assertFalse(result.passed)
        self.assertIn("failed to inject pending depth", result.detail)
        self.assertEqual([("prefill-0", 200_000)], smoke.queue_depth_updates)


if __name__ == "__main__":
    unittest.main()
