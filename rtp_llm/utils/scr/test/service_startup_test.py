"""Verify the resources actually present at the pre-service template boundary."""

import os
import unittest
from unittest.mock import patch

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.host_service import HostService, HostServiceArgs, VipServerWrapper
from rtp_llm.utils.scr.restore_context import RestoreContext
from rtp_llm.utils.scr.lifecycle import TemplateLifecycle
from rtp_llm.vipserver.host import Host
from rtp_llm.vipserver.host_reactor import HostReactor
from rtp_llm.vipserver.vip_client import VipClient
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


class ScrServiceStartupTest(unittest.TestCase):
    def test_first_role_lookup_uses_current_instances_without_cache_fixup(self):
        # Each restore of a service-preparation template starts from empty
        # discovery state; it does not inherit another clone's served requests.
        for suffix in (20, 30):
            lifecycle = TemplateLifecycle()
            reactor = HostReactor(VIPServerProxy())
            client = VipClient(reactor)
            self.assertFalse(reactor.started)
            self.assertFalse(reactor.proxy.started)
            self.assertFalse(reactor.domain_map)
            ip = f"192.0.2.{suffix}"
            with patch.dict(
                os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"}
            ), patch("rtp_llm.vipserver.vip_client.global_vip_client", client), patch(
                "rtp_llm.utils.scr.lifecycle.get_template_lifecycle",
                return_value=lifecycle,
            ), patch.object(
                reactor, "start", side_effect=lambda: setattr(reactor, "started", True)
            ) as start, patch.object(
                reactor.proxy,
                "req_api",
                return_value={"hosts": [{"valid": True, "ip": ip, "port": "9000"}]},
            ) as query, patch(
                "rtp_llm.vipserver.host_reactor.NetUtils.get_ip_addr", return_value=ip
            ):
                lifecycle.prepare_for_template("seed", "checkpoint")
                lifecycle.restore_fixup(RestoreContext("seed", ip))
                lifecycle.release_template("seed")
                service = HostService(HostServiceArgs(decode_domain="decode.vip"))
                self.addCleanup(service.master_service._probe_executor.shutdown)
                roles = service.get_backend_role_addrs([RoleType.DECODE])
                self.assertEqual([(r.ip, r.grpc_port) for r in roles], [(ip, 9001)])
                self.assertEqual(query.call_args.args[1]["clientIP"], ip)
                service.get_backend_role_addrs([RoleType.DECODE])
                start.assert_called_once()

    def test_normal_startup_keeps_eager_role_discovery(self):
        with patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "0"}), patch(
            "rtp_llm.vipserver.get_host_list_by_domain",
            return_value=[Host("192.0.2.10", "9000")],
        ) as discover:
            vip = VipServerWrapper("decode.vip")
        discover.assert_called_once_with("decode.vip")
        self.assertEqual(vip.hosts[0].ip, "192.0.2.10")

    def test_unused_discovery_can_close_without_starting_threads(self):
        reactor = HostReactor(VIPServerProxy())
        VipClient(reactor)
        reactor.close()
        self.assertFalse(reactor.started)
        self.assertFalse(reactor.proxy.started)


if __name__ == "__main__":
    unittest.main()
