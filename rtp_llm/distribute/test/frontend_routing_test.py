"""Frontend routing tests: PP stage-0 selection plus sCR-aware TP leader discovery."""

import os
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.distribute import distributed_server as ds
from rtp_llm.distribute.distributed_server import get_dp_rpc_addresses
from rtp_llm.utils.scr import network
from rtp_llm.utils.scr.restore_context import RestoreContext
from rtp_llm.utils.scr.template_utils import _BackendVisitorTemplateHook


class FrontendRoutingTest(unittest.TestCase):
    def config(self, *, pp=2, dp=1, tp=2, local_size=2, world_rank=2):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.pp_size, pc.dp_size, pc.tp_size = pp, dp, tp
        pc.ep_size = 0
        pc.world_size = pp * dp * tp
        pc.local_world_size = local_size
        set_parallelism_config(pc, world_rank)
        config.server_config.start_port = 8000
        config.server_config.worker_info_port_num = 10
        return config

    def test_pp_selects_only_local_stage_zero_dp_roots(self):
        cases = [
            # Both stage-0 DP lanes are local.
            (
                self.config(dp=2, local_size=8, world_rank=4),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            # Stage 0 spans two nodes; each frontend keeps only its local lanes.
            (
                self.config(dp=4, local_size=4, world_rank=0),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            (
                self.config(dp=4, local_size=4, world_rank=4),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            # This node straddles stages: rank 4 serves, rank 6 does not.
            (
                self.config(dp=3, local_size=4, world_rank=4),
                ["127.0.0.1:8001"],
            ),
        ]
        for config, expected in cases:
            pc = config.parallelism_config
            with (
                self.subTest(dp=pc.dp_size, world_rank=pc.world_rank),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
                self.assertEqual(get_dp_rpc_addresses(config), expected)
                store.assert_not_called()

    def test_no_local_serving_rank_does_not_discover_remote_backends(self):
        for config in (
            self.config(),
            self.config(dp=2, local_size=4, world_rank=4),
            self.config(dp=4, local_size=4, world_rank=8),
            self.config(pp=1, dp=1, tp=4, local_size=2, world_rank=2),
        ):
            pc = config.parallelism_config
            with (
                self.subTest(pp=pc.pp_size, dp=pc.dp_size, world_rank=pc.world_rank),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
                self.assertEqual(get_dp_rpc_addresses(config), [])
                store.assert_not_called()

    def test_non_pp_keeps_local_dp_addresses(self):
        for config in (
            self.config(pp=1, dp=4, local_size=4, world_rank=0),
            self.config(pp=1, dp=4, local_size=4, world_rank=4),
        ):
            with (
                self.subTest(pp=config.parallelism_config.pp_size),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
                self.assertEqual(
                    get_dp_rpc_addresses(config),
                    ["127.0.0.1:8001", "127.0.0.1:8021"],
                )
                store.assert_not_called()


    def test_warmup_without_local_serving_rank_keeps_resolution_error(self):
        from rtp_llm.start_server import (
            StartupRealWarmupAddressResolutionError,
            _get_startup_real_warmup_grpc_addresses,
        )

        with (
            patch(
                "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                return_value="127.0.0.1",
            ),
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            self.assertRaises(StartupRealWarmupAddressResolutionError),
        ):
            _get_startup_real_warmup_grpc_addresses(self.config())
        store.assert_not_called()

    def test_routes_only_local_tp_groups_and_preserves_local_members(self):
        for scr in (False, True):
            for tp, node, expected_ranks in (
                (4, 0, [0]),
                (4, 1, [0]),
                (2, 0, [0]),
                (2, 1, [2]),
                (1, 0, [0, 1]),
                (1, 1, [2, 3]),
            ):
                with self.subTest(scr=scr, tp=tp, node=node):
                    configs = PyEnvConfigs()
                    pc = configs.parallelism_config
                    pc.world_size, pc.local_world_size = 4, 2
                    pc.world_rank, pc.local_rank = node * 2, 0
                    pc.tp_size, pc.dp_size = tp, 4 // tp
                    configs.server_config.ip = f"10.0.0.{node + 2}"
                    configs.server_config.start_port = 22290
                    configs.server_config.worker_info_port_num = 10
                    prefix = "22.0.1" if scr else "10.0.0"
                    with patch.dict(
                        os.environ,
                        {
                            "RTPLLM_ENABLE_SCR": "1" if scr else "0",
                            "SCR_PHASE": "checkpoint",
                            "RANK_ID": str(node),
                        },
                        clear=True,
                    ), patch.object(
                        network, "topology", return_value={0: "22.0.1.2", 1: "22.0.1.3"}
                    ), patch.object(
                        ds, "get_master", return_value=(f"{prefix}.2", "22290")
                    ), patch.object(
                        ds, "TCPStore"
                    ) as store:
                        store.return_value.get.return_value = (
                            f"{prefix}.2:23000".encode()
                        )
                        local = ds.get_local_world_info(
                            configs.server_config, configs.distribute_config, pc
                        )
                        before = [vars(member).copy() for member in local.members]
                        with patch.object(ds, "get_world_info", return_value=local):
                            routed = ds.get_frontend_world_info(
                                configs.server_config, configs.distribute_config, pc
                            )
                        self.assertEqual(
                            [vars(member) for member in local.members], before
                        )
                        self.assertEqual(
                            [member.world_rank for member in local.members],
                            [node * 2, node * 2 + 1],
                        )
                        leaders = [
                            member
                            for member in routed.members
                            if member.world_rank % tp == 0
                        ]
                        self.assertEqual(
                            [member.world_rank for member in leaders], expected_ranks
                        )
                        addresses = ds.get_dp_addrs_from_world_info(routed, pc)
                        if tp == 4 and node == 1:
                            self.assertEqual(addresses, [f"{prefix}.2:23001"])
                            store.return_value.get.assert_called_once_with(
                                "registry_rank_address_0"
                            )
                            store.return_value.set.assert_not_called()
                        else:
                            self.assertEqual(
                                addresses,
                                [
                                    f"{prefix}.{node + 2}:{22291 + (rank % 2) * 10}"
                                    for rank in expected_ranks
                                ],
                            )
                            store.assert_not_called()

    def test_remote_leader_uses_actual_tcpstore_registration(self):
        configs = PyEnvConfigs()
        pc = configs.parallelism_config
        pc.world_size, pc.local_world_size = 4, 2
        pc.world_rank, pc.local_rank = 2, 0
        pc.tp_size, pc.dp_size = 4, 1
        configs.server_config.ip = "10.0.0.3"
        store = ds.TCPStore(
            "127.0.0.1", 0, None, True, timedelta(seconds=3), wait_for_workers=False
        )
        store.set("registry_rank_address_0", "10.0.0.2:24000")
        with patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "0"}, clear=True
        ), patch.object(
            ds, "get_master", return_value=("127.0.0.1", str(store.port + 1))
        ):
            world = ds.get_frontend_world_info(
                configs.server_config, configs.distribute_config, pc
            )
        self.assertEqual(ds.get_dp_addrs_from_world_info(world, pc), ["10.0.0.2:24001"])

    def test_remote_leader_retries_transient_registry_failure(self):
        configs = PyEnvConfigs()
        pc = configs.parallelism_config
        pc.world_size, pc.local_world_size, pc.world_rank = 2, 1, 1
        pc.tp_size = 2
        configs.server_config.ip = "10.0.0.3"
        with patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "0"}, clear=True), patch.object(
            ds, "get_master", return_value=("10.0.0.2", "24000")
        ), patch.object(ds, "TCPStore") as store, patch.object(ds.time, "sleep") as sleep:
            store.return_value.get.side_effect = [
                RuntimeError("rank is still starting"),
                b"10.0.0.2:24000",
            ]
            world = ds.get_frontend_world_info(
                configs.server_config, configs.distribute_config, pc
            )
        self.assertEqual(ds.get_dp_addrs_from_world_info(world, pc), ["10.0.0.2:24001"])
        self.assertEqual(store.call_count, 2)
        sleep.assert_called_once()

    def test_restore_reuses_checkpointed_route_without_registry_lookup(self):
        configs = PyEnvConfigs()
        pc = configs.parallelism_config
        pc.world_size, pc.local_world_size = 4, 2
        pc.world_rank, pc.local_rank = 2, 0
        pc.tp_size, pc.dp_size = 4, 1
        configs.server_config.ip = "10.0.0.3"
        visitor = MagicMock()
        with patch.dict(
            os.environ,
            {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint", "RANK_ID": "1"},
            clear=True,
        ), patch.object(
            network, "topology", return_value={0: "22.0.1.2", 1: "22.0.1.3"}
        ), patch.object(
            ds, "get_master", return_value=("22.0.1.2", "22290")
        ), patch.object(
            ds, "TCPStore"
        ) as store:
            store.return_value.get.return_value = b"22.0.1.2:23000"
            world = ds.get_frontend_world_info(
                configs.server_config, configs.distribute_config, pc
            )
            hook = _BackendVisitorTemplateHook(visitor, configs, world)
            store.reset_mock()
            with patch.dict(os.environ, {"SCR_PHASE": "restore"}):
                hook.restore_fixup(RestoreContext("seed", "10.1.0.3"))
                hook.restore_fixup(RestoreContext("seed", "10.2.0.3"))
            store.assert_not_called()
            visitor.update_addresses.assert_called_with(["22.0.1.2:23001"])
            self.assertEqual(visitor.source_ip, "10.2.0.3")
            with patch.object(
                network, "topology", return_value={0: "22.0.1.9", 1: "22.0.1.3"}
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot change"):
                    hook.restore_fixup(RestoreContext("seed", "10.3.0.3"))


if __name__ == "__main__":
    unittest.main()
