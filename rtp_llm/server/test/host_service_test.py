"""
Unit tests for MasterService: VIP discovery and heartbeat probes are mocked.
"""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

from rtp_llm.server.host_service import FlexlbHeartbeatInfo, HostService, MasterService
from rtp_llm.vipserver.host import Host


def _json_response(payload: dict, status_code: int = 200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload
    return r


class TestMasterService(unittest.TestCase):
    def _make_service(self, vip: MagicMock):
        with patch("rtp_llm.server.host_service.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value.start = MagicMock()
            svc = MasterService(vip)
        return svc, {}

    def _refresh(
        self,
        svc: MasterService,
        host_health_map: Dict[str, FlexlbHeartbeatInfo],
    ):
        svc._refresh_route_snapshot(host_health_map)

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_refresh_single_host_sets_master_and_queue(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        mock_post.return_value = _json_response(
            {"real_master_host": "10.0.0.1:8000", "queue_length": 7}
        )

        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)

        self.assertEqual(svc.get_master_addr(), "10.0.0.1:8000")
        self.assertEqual(svc.get_queue_length(), 7)
        self.assertIsNone(svc.get_slave_addr())
        mock_post.assert_called_once()

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_refresh_two_hosts_prefers_server_marked_master(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        h1 = Host("10.0.0.1", "8000")
        h2 = Host("10.0.0.2", "8000")
        vip.get_hosts.return_value = [h1, h2]

        def post_side_effect(url, **_kwargs):
            if "10.0.0.1:8000" in url:
                return _json_response(
                    {"real_master_host": "10.0.0.2:8000", "queue_length": 1}
                )
            if "10.0.0.2:8000" in url:
                return _json_response(
                    {"real_master_host": "10.0.0.2:8000", "queue_length": 2}
                )
            raise AssertionError(f"unexpected url {url}")

        mock_post.side_effect = post_side_effect

        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)

        self.assertEqual(svc.get_master_addr(), "10.0.0.2:8000")
        self.assertEqual(svc.get_queue_length(), 2)
        self.assertEqual(svc.get_slave_addr(), "10.0.0.1:8000")
        self.assertEqual(mock_post.call_count, 2)

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_refresh_no_discovery_hosts_empty_snapshot(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        vip.get_hosts.return_value = []

        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)

        self.assertIsNone(svc.get_master_addr())
        self.assertEqual(svc.get_queue_length(), 0)
        mock_post.assert_not_called()

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_refresh_failover_after_previous_master_unhealthy(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        h1 = Host("10.0.0.1", "8000")
        h2 = Host("10.0.0.2", "8000")
        vip.get_hosts.return_value = [h1, h2]

        def round1(url, **_kwargs):
            if "10.0.0.1:8000" in url:
                return _json_response(
                    {"real_master_host": "10.0.0.1:8000", "queue_length": 1}
                )
            return _json_response(
                {"real_master_host": "10.0.0.1:8000", "queue_length": 1}
            )

        mock_post.side_effect = round1
        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_master_addr(), "10.0.0.1:8000")

        def round2(url, **_kwargs):
            if "10.0.0.1:8000" in url:
                return _json_response({}, status_code=503)
            return _json_response(
                {"real_master_host": "10.0.0.1:8000", "queue_length": 3}
            )

        mock_post.side_effect = round2
        self._refresh(svc, host_health_map)

        def round3(url, **_kwargs):
            if "10.0.0.1:8000" in url:
                return _json_response({}, status_code=503)
            return _json_response(
                {"real_master_host": "10.0.0.2:8000", "queue_length": 4}
            )

        mock_post.side_effect = round3
        self._refresh(svc, host_health_map)

        self.assertEqual(svc.get_master_addr(), "10.0.0.2:8000")
        self.assertEqual(svc.get_queue_length(), 4)
        status = svc.get_host_health_status()
        self.assertIn("10.0.0.1:8000", status)
        self.assertEqual(status["10.0.0.1:8000"]["health"], "unhealthy")

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_get_host_health_status_reads_snapshot(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        mock_post.return_value = _json_response(
            {"real_master_host": "10.0.0.1:8000", "queue_length": 0}
        )

        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)
        status = svc.get_host_health_status()

        self.assertIn("10.0.0.1:8000", status)
        self.assertEqual(status["10.0.0.1:8000"]["health"], "healthy")
        self.assertTrue(status["10.0.0.1:8000"]["is_master"])

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_refresh_keeps_previous_master_when_still_healthy(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        h1 = Host("10.0.0.1", "8000")
        h2 = Host("10.0.0.2", "8000")
        vip.get_hosts.return_value = [h1, h2]

        def round1(url, **_kwargs):
            if "10.0.0.1:8000" in url:
                return _json_response(
                    {"real_master_host": "10.0.0.1:8000", "queue_length": 2}
                )
            return _json_response(
                {"real_master_host": "10.0.0.1:8000", "queue_length": 1}
            )

        mock_post.side_effect = round1
        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_master_addr(), "10.0.0.1:8000")
        self.assertEqual(svc.get_queue_length(), 2)

        # round2: no host claims to be master → falls into previous_master sticky path
        def round2(url, **_kwargs):
            return _json_response(
                {"real_master_host": "10.0.0.99:8000", "queue_length": 5}
            )

        mock_post.side_effect = round2
        self._refresh(svc, host_health_map)

        self.assertEqual(svc.get_master_addr(), "10.0.0.1:8000")
        self.assertEqual(svc.get_queue_length(), 5)
        self.assertEqual(svc.get_slave_addr(), "10.0.0.2:8000")

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_collect_hosts_uses_refresh_discovery(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        mock_post.return_value = _json_response(
            {"real_master_host": "10.0.0.1:8000", "queue_length": 1}
        )

        svc, host_health_map = self._make_service(vip)
        self._refresh(svc, host_health_map)

        vip.get_hosts.assert_called_with(refresh=True)
        mock_post.assert_called_once()

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    @patch("rtp_llm.server.host_service.time.time")
    def test_cleanup_removes_expired_unhealthy_host(
        self,
        mock_time: MagicMock,
        mock_post: MagicMock,
        _mock_kmonitor: MagicMock,
    ):
        vip = MagicMock()
        vip.domain = "master.vip"
        host = Host("10.0.0.1", "8000")
        vip.get_hosts.return_value = [host]
        mock_post.return_value = _json_response({}, status_code=503)

        svc, host_health_map = self._make_service(vip)

        mock_time.return_value = 100.0
        self._refresh(svc, host_health_map)
        mock_time.return_value = 101.0
        self._refresh(svc, host_health_map)

        status = svc.get_host_health_status()
        self.assertIn("10.0.0.1:8000", status)
        self.assertEqual(status["10.0.0.1:8000"]["health"], "unhealthy")

        vip.get_hosts.return_value = []
        mock_time.return_value = 200.0
        self._refresh(svc, host_health_map)

        self.assertNotIn("10.0.0.1:8000", svc.get_host_health_status())
        self.assertIsNone(svc.get_master_addr())

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_payload_omission_requires_explicit_boolean_capability(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        addr = "10.0.0.1:8000"
        vip = MagicMock(domain="master.vip")
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        svc, host_health_map = self._make_service(vip)
        host_service = object.__new__(HostService)
        host_service.master_service = svc
        self.assertTrue(host_service.requires_generate_input(addr))

        for value in (False, True, None, 0, 1, "false", "true", [], {}):
            with self.subTest(value=value):
                mock_post.return_value = _json_response(
                    {"real_master_host": addr, "requires_generate_input": value}
                )
                self._refresh(svc, host_health_map)
                self.assertEqual(
                    host_service.requires_generate_input(addr), value is not False
                )
                self.assertIs(
                    svc.get_host_health_status()[addr]["requires_generate_input"],
                    value if type(value) is bool else None,
                )

        self.assertTrue(host_service.requires_generate_input("unknown:8000"))

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_old_heartbeat_clears_previously_advertised_payload_capability(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        addr = "10.0.0.1:8000"
        vip = MagicMock(domain="master.vip")
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        svc, host_health_map = self._make_service(vip)
        mock_post.return_value = _json_response(
            {"real_master_host": addr, "requires_generate_input": False}
        )
        self._refresh(svc, host_health_map)
        self.assertFalse(svc.requires_generate_input(addr))
        previous_snapshot = svc._route_snapshot

        mock_post.return_value = _json_response({"real_master_host": addr})
        self._refresh(svc, host_health_map)

        self.assertTrue(svc.requires_generate_input(addr))
        self.assertIsNone(svc.get_host_health_status()[addr]["requires_generate_input"])
        self.assertIs(
            previous_snapshot.host_health_status[addr].requires_generate_input, False
        )

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_first_failed_probe_restores_payload_before_host_becomes_unhealthy(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        addr = "10.0.0.1:8000"
        vip = MagicMock(domain="master.vip")
        vip.get_hosts.return_value = [Host("10.0.0.1", "8000")]
        svc, host_health_map = self._make_service(vip)
        mock_post.return_value = _json_response(
            {"real_master_host": addr, "requires_generate_input": False}
        )
        self._refresh(svc, host_health_map)
        self.assertFalse(svc.requires_generate_input(addr))

        mock_post.return_value = _json_response({}, status_code=503)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_host_health_status()[addr]["health"], "healthy")
        self.assertTrue(svc.requires_generate_input(addr))
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_host_health_status()[addr]["health"], "unhealthy")
        self.assertTrue(svc.requires_generate_input(addr))

    @patch("rtp_llm.server.host_service.kmonitor.report")
    @patch("requests.post")
    def test_payload_capability_follows_target_host_across_master_change(
        self, mock_post: MagicMock, _mock_kmonitor: MagicMock
    ):
        first, second, new = "10.0.0.1:8000", "10.0.0.2:8000", "10.0.0.3:8000"
        vip = MagicMock(domain="master.vip")
        vip.get_hosts.return_value = [
            Host("10.0.0.1", "8000"),
            Host("10.0.0.2", "8000"),
        ]
        svc, host_health_map = self._make_service(vip)

        def heartbeat(master):
            def post(url, **_kwargs):
                payload = {"real_master_host": master}
                if first in url:
                    payload["requires_generate_input"] = False
                elif second in url:
                    payload["requires_generate_input"] = True
                return _json_response(payload)

            return post

        mock_post.side_effect = heartbeat(first)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_master_addr(), first)
        self.assertFalse(svc.requires_generate_input(svc.get_master_addr()))
        self.assertTrue(svc.requires_generate_input(svc.get_slave_addr()))

        mock_post.side_effect = heartbeat(second)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_master_addr(), second)
        self.assertTrue(svc.requires_generate_input(svc.get_master_addr()))
        self.assertFalse(svc.requires_generate_input(svc.get_slave_addr()))

        vip.get_hosts.return_value = [Host("10.0.0.3", "8000")]
        mock_post.side_effect = heartbeat(new)
        self._refresh(svc, host_health_map)
        self.assertEqual(svc.get_master_addr(), new)
        self.assertTrue(svc.requires_generate_input(new))


if __name__ == "__main__":
    unittest.main()
