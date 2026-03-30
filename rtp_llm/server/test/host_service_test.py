"""
Unit tests for MasterService: VIP discovery and heartbeat probes are mocked.
"""

import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

from rtp_llm.server.host_service import FlexlbHeartbeatInfo
from rtp_llm.server.host_service import MasterService
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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
    @patch("rtp_llm.server.host_service.requests.post")
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


if __name__ == "__main__":
    unittest.main()
