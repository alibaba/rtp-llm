"""Verify the resources actually present at the pre-service template boundary."""

import asyncio
import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.server.host_service import HostService, HostServiceArgs, VipServerWrapper
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper
from rtp_llm.utils.scr_restore_context import RestoreContext
from rtp_llm.utils.scr_template_lifecycle import TemplateLifecycle
from rtp_llm.vipserver.host import Host
from rtp_llm.vipserver.host_reactor import HostReactor
from rtp_llm.vipserver.vip_client import VipClient
from rtp_llm.vipserver.vipserver_proxy import VIPServerProxy


def service_config():
    def endpoint(domain):
        return dict(type="vipserver", address=domain, protocol="http", path="")

    return json.dumps(
        dict(
            service_id="model",
            use_local=False,
            master_endpoint=endpoint("master.vip"),
            role_endpoints=[
                dict(
                    group="default",
                    prefill_endpoint=endpoint("prefill.vip"),
                    decode_endpoint=endpoint("decode.vip"),
                )
            ],
        )
    )


class ScrServiceStartupTest(unittest.TestCase):
    def test_template_clients_have_no_connections_and_start_master_only_on_release(
        self,
    ):
        for phase in ("checkpoint", "restore"):
            for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.FRONTEND):
                lifecycle = TemplateLifecycle()
                with self.subTest(phase=phase, role=role), patch.dict(
                    os.environ,
                    {
                        "RTPLLM_ENABLE_SCR": "1",
                        "SCR_PHASE": phase,
                        "MODEL_SERVICE_CONFIG": service_config(),
                    },
                ), patch(
                    "rtp_llm.server.host_service.get_template_lifecycle",
                    return_value=lifecycle,
                ), patch(
                    "grpc.aio.insecure_channel",
                    side_effect=AssertionError("gRPC before request"),
                ), patch(
                    "requests.sessions.Session.request",
                    side_effect=AssertionError("HTTP before release"),
                ), patch(
                    "rtp_llm.server.host_service.threading.Thread"
                ) as thread:
                    visitor = BackendRPCServerVisitor(
                        max_seq_len=1024,
                        seq_size_per_block=16,
                        pd_sep_config=NS(
                            max_rpc_timeout_ms=1000,
                            decode_entrance=False,
                            role_type=role,
                            to_string=lambda: "template test",
                        ),
                        addresses=["127.0.0.1:9001"],
                        server_config=NS(ip="192.0.2.10"),
                    )
                    master = visitor.host_service.master_service
                    self.addCleanup(master._probe_executor.shutdown)
                    self.addCleanup(lambda v=visitor: asyncio.run(v.close()))
                    health = GrpcClientWrapper(9001)
                    lifecycle.prepare_for_template("seed", phase)
                    self.assertFalse(visitor.model_rpc_client._channel_pool._channels)
                    self.assertIsNone(
                        visitor.model_rpc_client._channel_pool._cleanup_task
                    )
                    self.assertFalse(visitor.master_client._channels)
                    self.assertIsNone(health.channel)
                    self.assertFalse(master._probe_executor._threads)
                    self.assertIsNone(master.get_master_addr())
                    thread.assert_not_called()
                    lifecycle.restore_fixup(RestoreContext("seed", "192.0.2.20"))
                    thread.assert_not_called()
                    lifecycle.release_template("seed")
                    master.start_refresh()
                    thread.assert_called_once()
                    thread.return_value.start.assert_called_once()

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
                "rtp_llm.server.host_service.get_template_lifecycle",
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
                service = HostService(HostServiceArgs(decode_domain="decode.vip"))
                self.addCleanup(service.master_service._probe_executor.shutdown)
                start.assert_not_called()
                query.assert_not_called()
                lifecycle.prepare_for_template("seed", "checkpoint")
                lifecycle.restore_fixup(RestoreContext("seed", ip))
                lifecycle.release_template("seed")
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
