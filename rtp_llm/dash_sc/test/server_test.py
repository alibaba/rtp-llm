"""Keepalive defaults and deployment override contract."""

import unittest

from rtp_llm.dash_sc.proxy.servicer import _FORWARD_CHANNEL_OPTS
from rtp_llm.dash_sc.server import _merge_server_keepalive


class KeepaliveOptionsTest(unittest.TestCase):
    def test_only_active_streams_are_kept_alive(self):
        client = dict(_FORWARD_CHANNEL_OPTS)
        server = dict(_merge_server_keepalive([]))
        for options in (client, server):
            self.assertEqual(options["grpc.keepalive_permit_without_calls"], 0)
            self.assertEqual(options["grpc.http2.max_pings_without_data"], 0)
            self.assertEqual(options["grpc.keepalive_time_ms"], 30000)
            self.assertEqual(options["grpc.keepalive_timeout_ms"], 10000)
        self.assertLessEqual(
            server["grpc.http2.min_recv_ping_interval_without_data_ms"],
            client["grpc.keepalive_time_ms"],
        )
        self.assertNotIn("grpc.http2.min_ping_interval_without_data_ms", server)

    def test_explicit_configuration_and_idle_policy_are_preserved(self):
        overrides = {
            "grpc.keepalive_permit_without_calls": 1,
            "grpc.keepalive_time_ms": 60000,
            "grpc.keepalive_timeout_ms": 20000,
            "grpc.http2.min_recv_ping_interval_without_data_ms": 1000,
            "grpc.http2.max_ping_strikes": 1000,
            "grpc.max_connection_idle_ms": 600000,
        }
        merged = dict(_merge_server_keepalive(list(overrides.items())))
        for key, value in overrides.items():
            self.assertEqual(merged[key], value)
        self.assertEqual(
            _merge_server_keepalive(list(merged.items())), sorted(merged.items())
        )


if __name__ == "__main__":
    unittest.main()
