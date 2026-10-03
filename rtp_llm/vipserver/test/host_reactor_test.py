import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests


class HostReactorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the real modules without starting the package's global client.
        package = types.ModuleType("rtp_llm.vipserver")
        package.__path__ = [str(Path(__file__).resolve().parents[1])]
        with patch.dict(sys.modules, {"rtp_llm.vipserver": package}):
            cls.module = importlib.import_module("rtp_llm.vipserver.host_reactor")

    def setUp(self):
        self.http = self.enterContext(patch.object(requests, "get"))
        self.enterContext(
            patch.object(self.module.NetUtils, "get_ip_addr", return_value="127.0.0.1")
        )
        self.enterContext(patch.object(self.module.logging, "warning"))
        self.enterContext(patch.object(self.module.logging, "error"))
        proxy = self.module.VIPServerProxy()
        proxy.srv_hosts = ["192.0.2.1"]
        self.reactor = self.module.HostReactor(proxy)

    def respond(self, payload, domain="test-domain"):
        self.http.return_value = Mock()
        self.http.return_value.json.return_value = payload
        return self.reactor.get_host_list_by_domain_now(domain)

    def hosts(self, ip):
        return {"hosts": [{"ip": ip, "port": 8000, "valid": True}]}

    def test_repeated_empty_results_keep_cached_hosts(self):
        original = self.respond(self.hosts("10.0.0.1"))
        for _ in range(120):
            self.assertIs(self.respond({"hosts": []}), original)
        self.assertIs(self.reactor.get_host_list_by_domain("test-domain"), original)
        self.assertEqual(self.http.call_count, 121)

    def test_network_failures_keep_cached_hosts(self):
        original = self.respond(self.hosts("10.0.0.1"))
        for failure in (requests.ConnectionError("unreachable"), requests.Timeout("timeout")):
            with self.subTest(failure=type(failure).__name__):
                self.http.side_effect = failure
                for _ in range(65):
                    self.assertIs(
                        self.reactor.get_host_list_by_domain_now("test-domain"), original
                    )

    def test_non_empty_result_updates_after_empty_results(self):
        self.respond(self.hosts("10.0.0.1"))
        for _ in range(65):
            self.respond({"hosts": []})
        updated = self.respond(self.hosts("10.0.0.2"))
        self.assertEqual([(host.ip, host.port) for host in updated], [("10.0.0.2", 8000)])
        self.assertIs(self.reactor.get_host_list_by_domain("test-domain"), updated)

    def test_empty_initial_result_can_recover(self):
        self.assertIsNone(self.respond({"hosts": []}))
        self.assertEqual(self.respond(self.hosts("10.0.0.1"))[0].ip, "10.0.0.1")

    def test_missing_hosts_keep_cached_hosts(self):
        original = self.respond(self.hosts("10.0.0.1"))
        for _ in range(65):
            self.assertIs(self.respond({}), original)

    def test_all_invalid_hosts_keep_cached_hosts(self):
        original = self.respond(self.hosts("10.0.0.1"))
        for _ in range(65):
            self.assertIs(
                self.respond({"hosts": [{"ip": "10.0.0.2", "port": 8000, "valid": False}]}),
                original,
            )

    def test_empty_domain_does_not_block_other_domain_updates(self):
        original = self.respond(self.hosts("10.0.0.1"))
        self.respond(self.hosts("10.0.0.2"), domain="other-domain")
        self.http.side_effect = [
            Mock(json=Mock(return_value={"hosts": []})),
            Mock(json=Mock(return_value=self.hosts("10.0.0.3"))),
        ]
        self.reactor.refresh_cache_domain_srv_lst()
        self.assertIs(self.reactor.get_host_list_by_domain("test-domain"), original)
        self.assertEqual(self.reactor.get_host_list_by_domain("other-domain")[0].ip, "10.0.0.3")


if __name__ == "__main__":
    unittest.main()
