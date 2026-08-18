from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


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


if __name__ == "__main__":
    unittest.main()
