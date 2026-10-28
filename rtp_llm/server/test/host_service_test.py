"""
Unit tests for MasterService: VIP discovery and heartbeat probes are mocked.
"""

import os
import threading
import time
import unittest
from typing import Dict
from unittest.mock import MagicMock, patch

from rtp_llm.server.host_service import (
    FlexlbHeartbeatInfo,
    HostService,
    HostServiceArgs,
    MasterService,
    VipServerWrapper,
)
from rtp_llm.vipserver.host import Host
from rtp_llm.vipserver.host_reactor import HostReactor
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


def _json_response(payload: dict, status_code: int = 200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload
    return r


class TestMasterService(unittest.TestCase):
    def test_close_joins_refresh_and_preserves_immutable_pair(self):
        vip = MagicMock()
        vip.domain = ""
        svc = MasterService(vip)
        pair = svc.get_route_snapshot()
        self.assertIsNone(pair.master_addr)
        self.assertIsNone(pair.slave_addr)
        svc.close()
        self.assertFalse(svc.backend_refresh_thread.is_alive())
        svc.close()

    def _make_service(self, vip: MagicMock):
        with patch("rtp_llm.server.host_service.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value.start = MagicMock()
            svc = MasterService(vip)
        self.addCleanup(svc.close)
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


class TestVipTimeout(unittest.TestCase):
    def test_idle_reactors_close_without_interval_wait_or_final_callback(self):
        callbacks = []
        reactors = []
        for index in range(5):
            proxy = VIPServerProxy(http_timeout=0.25)
            # Only replace the network callbacks; lifecycle uses real threads.
            proxy.refresh_srv_lst = lambda: None
            proxy.update_srv_thread.func = lambda: callbacks.append("server")
            reactor = HostReactor(proxy)
            reactor.update_domain_thread.func = lambda: callbacks.append("domain")
            reactor.start()
            reactors.append(reactor)
            self.addCleanup(reactor.close)
        started = time.monotonic()
        for reactor in reactors:
            reactor.close()
            self.assertFalse(reactor.update_domain_thread.is_alive())
            self.assertFalse(reactor.proxy.update_srv_thread.is_alive())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(callbacks, [])

    def test_close_signals_both_workers_before_joining_inflight_callback(self):
        entered = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        proxy = VIPServerProxy(http_timeout=0.25)
        proxy.refresh_srv_lst = lambda: None
        reactor = HostReactor(proxy)

        def blocked_callback():
            entered.set()
            release.wait(5)

        reactor.update_domain_thread.func = blocked_callback
        reactor.update_domain_thread.interval = 0.001
        reactor.start()
        self.addCleanup(reactor.close)
        self.addCleanup(release.set)
        self.assertTrue(entered.wait(2))

        def close():
            reactor.close()
            closed.set()

        closer = threading.Thread(target=close)
        closer.start()
        try:
            # The server worker must exit even while domain join is blocked.
            proxy.update_srv_thread.join(timeout=2)
            self.assertFalse(proxy.update_srv_thread.is_alive())
            self.assertFalse(closed.is_set())
        finally:
            release.set()
            closer.join(timeout=2)
        self.assertTrue(closed.is_set())
        self.assertFalse(reactor.update_domain_thread.is_alive())

    def test_timeout_is_instance_local_and_default_ignores_native_env(self):
        response = MagicMock()
        response.text = "127.0.0.1"
        response.json.return_value = {"hosts": []}
        with patch.dict(
            os.environ, {"PG_NATIVE_VIP_HTTP_TIMEOUT_SECONDS": "0.75"}
        ), patch(
            "rtp_llm.vipserver.vipserver_proxy.requests.get", return_value=response
        ) as get, patch(
            "rtp_llm.vipserver.vipserver_proxy.get_address_server_params",
            return_value={},
        ):
            for timeout in (None, 0.25):
                proxy = VIPServerProxy(http_timeout=timeout)
                proxy.refresh_srv_lst()
                expected = {} if timeout is None else {"timeout": timeout}
                self.assertEqual(get.call_args.kwargs, expected)
                proxy.req_api("srvIPXT", {"dom": "example"})
                self.assertEqual(get.call_args.kwargs, expected)

    def test_wrapper_owns_bounded_reactor(self):
        with patch("rtp_llm.vipserver.host_reactor.HostReactor") as reactor, patch(
            "rtp_llm.vipserver.vipserver_proxy.VIPServerProxy"
        ) as proxy:
            wrapper = VipServerWrapper("example", http_timeout=0.25)
            proxy.assert_called_once_with(http_timeout=0.25)
            reactor.return_value.start.assert_called_once_with()
            wrapper.get_hosts(refresh=True)
            reactor.return_value.get_host_list_by_domain_now.assert_called_once_with(
                "example"
            )
            wrapper.close()
            wrapper.close()
            reactor.return_value.close.assert_called_once_with()

    def test_host_service_passes_optional_timeout_to_every_role(self):
        args = HostServiceArgs(
            master_domain="master",
            pdfusion_domain="pdfusion",
            prefill_domain="prefill",
            decode_domain="decode",
            vit_domain="vit",
        )
        for timeout in (None, 0.25):
            with patch(
                "rtp_llm.server.host_service.VipServerWrapper"
            ) as wrapper, patch("rtp_llm.server.host_service.MasterService"):
                service = HostService(args, http_timeout=timeout)
                self.assertEqual(wrapper.call_count, 5)
                for call in wrapper.call_args_list:
                    self.assertEqual(call.kwargs, {"http_timeout": timeout})
                service.close()
                self.assertEqual(wrapper.return_value.close.call_count, 5)


if __name__ == "__main__":
    unittest.main()
