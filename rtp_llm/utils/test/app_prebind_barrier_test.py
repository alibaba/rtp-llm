import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from rtp_llm.utils.app_prebind_barrier import (
    AppPreBindBarrierClient,
    AppPreBindPhase,
    UnixSocketBarrierTransport,
    get_app_prebind_barrier_client,
)


class FakeTransport:
    def __init__(self, responses=None):
        self.requests = []
        self.responses = list(responses or [{"status": "released"}])

    def request(self, payload, timeout):
        self.requests.append((dict(payload), timeout))
        return self.responses.pop(0) if self.responses else {"status": "released"}


class AppPreBindBarrierTest(unittest.TestCase):
    def test_disabled_client_is_noop(self):
        transport = FakeTransport()
        client = AppPreBindBarrierClient("frontend", 1, transport=transport, enabled=False)
        self.assertTrue(client.prebind_ready())
        self.assertEqual(transport.requests, [])

    def test_arrive_wait_and_idempotency(self):
        transport = FakeTransport([{"status": "accepted"}, {"status": "released"}])
        client = AppPreBindBarrierClient("dash_sc", 3, generation="g1", transport=transport)
        self.assertTrue(client.prebind_ready({"port": 123}))
        self.assertTrue(client.prebind_ready({"port": 123}))
        self.assertEqual([r[0]["op"] for r in transport.requests], ["arrive", "wait_release"])
        self.assertEqual(transport.requests[0][0]["generation"], "g1")
        self.assertEqual(transport.requests[0][0]["phase"], AppPreBindPhase.PREBIND_READY.value)

    def test_arrive_without_wait_for_release(self):
        transport = FakeTransport([{"status": "accepted"}])
        client = AppPreBindBarrierClient("backend_rank", 0, transport=transport)
        self.assertTrue(client.prebind_ready(wait_for_release=False))
        self.assertEqual([r[0]["op"] for r in transport.requests], ["arrive"])

    def test_disconnect_fails_open_and_sends_abort_best_effort(self):
        class Failing(FakeTransport):
            def request(self, payload, timeout):
                self.requests.append((dict(payload), timeout))
                raise ConnectionError("gone")

        transport = Failing()
        client = AppPreBindBarrierClient("frontend", 0, transport=transport)
        self.assertFalse(client.prebind_ready())
        self.assertEqual([r[0]["op"] for r in transport.requests], ["arrive", "abort"])
        self.assertFalse(client.prebind_ready())
        self.assertEqual(len(transport.requests), 2)

    def test_restore_phase_and_final_release(self):
        transport = FakeTransport([{"status": "accepted"}, {"status": "released"}])
        client = AppPreBindBarrierClient("frontend", 0, transport=transport)
        self.assertTrue(client.restore_fixup_ready({"ip": "10.0.0.2"}))
        self.assertTrue(client.final_release())
        self.assertEqual(
            [(p[0]["op"], p[0]["phase"]) for p in transport.requests],
            [("arrive", "RESTORE_FIXUP_READY"), ("wait_release", "FINAL_RELEASE")],
        )

    def test_from_env_requires_both_gate_and_socket(self):
        with patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "0"}, clear=False):
            self.assertIsNone(get_app_prebind_barrier_client("frontend", 0))
        with patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "checkpoint"}, clear=True
        ):
            self.assertIsNone(get_app_prebind_barrier_client("frontend", 0))
        with patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "1"}, clear=False):
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(get_app_prebind_barrier_client("frontend", 0))


class UnixSocketBarrierTransportTest(unittest.TestCase):
    def test_json_lines_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "barrier.sock")
            ready = threading.Event()

            def serve():
                server = __import__("socket").socket(__import__("socket").AF_UNIX)
                server.bind(path)
                server.listen(1)
                ready.set()
                conn, _ = server.accept()
                with conn:
                    conn.recv(4096)
                    conn.sendall((json.dumps({"status": "released"}) + "\n").encode())
                server.close()

            thread = threading.Thread(target=serve)
            thread.start()
            self.assertTrue(ready.wait(1))
            response = UnixSocketBarrierTransport(path).request({"op": "wait_release"}, 1)
            thread.join(1)
            self.assertEqual(response["status"], "released")


if __name__ == "__main__":
    unittest.main()
