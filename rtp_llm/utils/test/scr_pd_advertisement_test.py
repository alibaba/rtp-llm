"""Exercise the real config writer and restore hook with separate PD/control IPs."""

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from rtp_llm.config.engine_config import update_worker_addrs
from rtp_llm.distribute.distributed_server import (
    WorldInfo,
    get_dp_addrs_from_world_info,
)
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.ops import FfnDisAggregateConfig
from rtp_llm.utils import scr_runtime_fixup as fixup
from rtp_llm.utils import scr_template_utils as scr
from rtp_llm.utils.scr_endpoint_provider import resolve_world_info
from rtp_llm.utils.scr_restore_context import RestoreContext
from rtp_llm.utils.scr_template_lifecycle import CallbackHook, TemplateLifecycle
from rtp_llm.utils.scr_template_utils import _BackendVisitorTemplateHook


class ScrPdAdvertisementTest(unittest.TestCase):
    def setUp(self):
        self.pc = NS(
            world_size=2,
            local_world_size=2,
            tp_size=2,
            dp_size=1,
            dp_rank=0,
            local_rank=0,
            world_rank=0,
        )
        self.world = NS(
            num_nodes=1,
            members=[
                NS(
                    ip="127.0.0.1",
                    world_rank=i,
                    local_rank=i,
                    cache_store_listen_port=18632 + i * 10,
                    cache_store_rdma_listen_port=18634 + i * 10,
                    rpc_server_port=18631 + i * 10,
                )
                for i in range(2)
            ],
        )
        for context in (
            patch.object(fixup, "_runtime_identity", None),
            patch.object(fixup, "_restore_env_provider", None),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.env = patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "restore",
                "RTP_LLM_SCR_ENDPOINT_MANIFEST": "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_checkpoint_uses_loopback_control_and_routable_kv_advertisement(self):
        self.pc.ffn_disaggregate_config = FfnDisAggregateConfig()
        members = [
            WorkerInfo("192.0.2.10", i, i, f"rank-{i}", 18630, 10) for i in range(2)
        ]
        current = WorldInfo(members, members[0], members[0], 1, True)
        runtime = NS()
        with patch.dict(os.environ, {"SCR_PHASE": "checkpoint"}), patch(
            "socket.gethostbyname", return_value="192.0.2.20"
        ):
            local_world = resolve_world_info(current, generation="g1")
            update_worker_addrs(runtime, self.pc, local_world)
            self.assertEqual(
                get_dp_addrs_from_world_info(local_world, self.pc), ["127.0.0.1:18631"]
            )
        self.assertEqual([member.ip for member in current.members], ["192.0.2.10"] * 2)
        self.assertEqual(
            [member.ip for member in local_world.members], ["127.0.0.1"] * 2
        )
        self.assertEqual(
            runtime.worker_grpc_addrs, ["127.0.0.1:18631", "127.0.0.1:18641"]
        )
        self.assertEqual(
            runtime.worker_addrs, ["192.0.2.20:18632:18634", "192.0.2.20:18642:18644"]
        )

    def test_prefill_preserves_control_loopback_and_advertises_current_pod(self):
        runtime = NS()
        with patch("socket.gethostbyname", side_effect=["192.0.2.20", "192.0.2.21"]):
            for ip in ["192.0.2.20", "192.0.2.21"]:
                update_worker_addrs(runtime, self.pc, self.world)
                self.assertEqual(
                    runtime.worker_addrs, [ip + ":18632:18634", ip + ":18642:18644"]
                )
                self.assertEqual(
                    runtime.worker_grpc_addrs, ["127.0.0.1:18631", "127.0.0.1:18641"]
                )

    def test_decode_dp_group_filter_is_preserved(self):
        self.pc.tp_size, self.pc.dp_size, self.pc.dp_rank = 1, 2, 1
        runtime = NS()
        with patch("socket.gethostbyname", return_value="192.0.2.30"):
            update_worker_addrs(runtime, self.pc, self.world)
        self.assertEqual(runtime.worker_addrs, ["192.0.2.30:18642:18644"])
        self.assertEqual(runtime.worker_grpc_addrs, ["127.0.0.1:18641"])

    def test_frontend_identity_refreshes_without_publishing_loopback(self):
        configs = NS(
            server_config=object(),
            distribute_config=object(),
            parallelism_config=self.pc,
            role_config=NS(role_type="PREFILL"),
        )
        visitor = Mock(source_ip="192.0.2.1")
        with patch(
            "rtp_llm.distribute.distributed_server.get_world_info",
            return_value=self.world,
        ), patch(
            "rtp_llm.utils.scr_endpoint_provider.resolve_world_info",
            return_value=self.world,
        ), patch(
            "rtp_llm.distribute.distributed_server.get_dp_addrs_from_world_info",
            return_value=["127.0.0.1:18631"],
        ), patch(
            "socket.gethostbyname", return_value="192.0.2.20"
        ):
            _BackendVisitorTemplateHook(visitor, configs).restore_fixup(
                RestoreContext("generation-2", "192.0.2.20")
            )
        self.assertEqual(visitor.source_ip, "192.0.2.20")
        visitor.update_addresses.assert_called_once_with(["127.0.0.1:18631"])

    def test_repeated_restore_repairs_all_consumers_before_release(self):
        self.pc.ffn_disaggregate_config = FfnDisAggregateConfig()
        members = [
            WorkerInfo(
                ip="127.0.0.1",
                local_rank=i,
                world_rank=i,
                server_port=18630,
                name=f"worker-{i}",
                worker_info_port_num=10,
            )
            for i in range(2)
        ]
        world = WorldInfo(
            members=members,
            master=members[0],
            self=members[0],
            num_nodes=1,
            initialized=True,
        )
        configs = NS(
            server_config=NS(ip="192.0.2.1"),
            distribute_config=NS(),
            parallelism_config=self.pc,
            role_config=NS(role_type="PREFILL"),
        )
        visitor = Mock(source_ip="192.0.2.1")
        runtime_config = NS()
        releases = []
        lifecycle = TemplateLifecycle()
        lifecycle.register_fixup(
            "server-config", scr._ServerConfigTemplateHook(configs)
        )
        lifecycle.register_fixup(
            "visitor", scr._BackendVisitorTemplateHook(visitor, configs)
        )
        lifecycle.register(
            "kv",
            CallbackHook(
                fixup=lambda _g: update_worker_addrs(runtime_config, self.pc, world),
                release=lambda _g: releases.append(
                    (
                        configs.server_config.ip,
                        visitor.source_ip,
                        list(runtime_config.worker_addrs),
                    )
                ),
            ),
        )
        provider = Mock(
            side_effect=[
                {"RequestedIP": "192.0.2.20"},
                {"RequestedIP": "192.0.2.30"},
                {"RequestedIP": "192.0.2.40"},
            ]
        )
        fixup.register_restore_env_provider(provider)
        # Exercise the seed return and clones restored with the same generation.
        with patch.dict(os.environ, {"SCR_PHASE": "checkpoint"}), patch.dict(
            sys.modules,
            {
                "libth_transformer": None,
                "rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper": None,
            },
        ), patch.object(
            scr, "get_template_lifecycle", return_value=lifecycle
        ), patch.object(
            scr, "arrive_scr_checkpoint_barrier", return_value=0
        ), patch(
            "rtp_llm.distribute.distributed_server.get_world_info", return_value=world
        ):
            for phase, ip in (
                ("checkpoint", "192.0.2.20"),
                ("restore", "192.0.2.30"),
                ("restore", "192.0.2.40"),
            ):
                os.environ["SCR_PHASE"] = phase
                self.assertEqual(
                    scr.arrive_scr_template_barrier(
                        worker_id=0, worker_num=1, generation="same-seed"
                    ),
                    0,
                )
                self.assertEqual(
                    releases[-1],
                    (
                        ip,
                        ip,
                        [
                            f"{ip}:{member.cache_store_listen_port}:{member.cache_store_rdma_listen_port}"
                            for member in members
                        ],
                    ),
                )
                self.assertEqual(
                    runtime_config.worker_grpc_addrs,
                    [f"127.0.0.1:{member.rpc_server_port}" for member in members],
                )
                self.assertEqual(os.environ["SCR_PHASE"], phase)
        self.assertEqual(provider.call_count, 3)
        self.assertEqual(len(releases), 3)


if __name__ == "__main__":
    unittest.main()
