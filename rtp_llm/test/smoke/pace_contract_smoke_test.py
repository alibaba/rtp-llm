import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.smoke.pace_contract_smoke import event_contract


class EventContractTest(unittest.TestCase):
    def make_server(self, retry_after_ms):
        server = Mock()
        server._rpc_port = 12345
        server.post_json.side_effect = [
            {"snapshot_required": True},
            {},
            {"committed_snapshot_version": 1},
            {"retry_after_ms": retry_after_ms},
            {"header": {"status": {"code": "OK"}}},
            {"hosts": [{"host_ip_port": "127.0.0.1:12345", "local": 1}]},
            {},
            {"hosts": []},
        ]
        return server

    @patch("rtp_llm.test.smoke.pace_contract_smoke.time.sleep")
    def test_waits_for_the_retry_hint_before_resending(self, sleep):
        server = self.make_server("1000")
        sleep.side_effect = lambda delay: self.assertEqual(server.post_json.call_count, 4)
        event_contract(server, "instance", [101])
        sleep.assert_called_once_with(1.05)
        self.assertEqual(server.post_json.call_count, 8)

    @patch("rtp_llm.test.smoke.pace_contract_smoke.time.sleep")
    def test_rejects_invalid_or_excessive_retry_without_waiting(self, sleep):
        for retry in (0, -1, 10_001, "18446744073709551615"):
            with self.subTest(retry=retry):
                server = self.make_server(retry)
                with self.assertRaisesRegex(AssertionError, "retry_after_ms must be within"):
                    event_contract(server, "instance", [101])
                self.assertEqual(server.post_json.call_count, 4)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
