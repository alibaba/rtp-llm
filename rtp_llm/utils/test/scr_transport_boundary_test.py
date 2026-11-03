"""Select loopback automatically only for a single-Pod SCR template."""

import os
import unittest
from itertools import product
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    WorldInfo,
    get_dp_addrs_from_world_info,
)
from rtp_llm.distribute.worker_info import WorkerInfo


class ScrTransportBoundaryTest(unittest.TestCase):
    def test_automatic_loopback_covers_tcpstore_nccl_and_rank_registration(self):
        for enabled, phase, local_size, rank in product(
            ("0", "1"), ("normal", "checkpoint", "restore"), (1, 2), (0, 1)
        ):
            with self.subTest(
                enabled=enabled, phase=phase, local_size=local_size, rank=rank
            ):
                configs = PyEnvConfigs()
                configs.server_config.ip = "192.0.2.20"
                pc = configs.parallelism_config
                pc.world_size, pc.local_world_size = 2, local_size
                pc.world_rank, pc.local_rank = rank, rank % local_size
                active = enabled == "1" and phase != "normal" and local_size == 2
                env = {
                    "RTPLLM_ENABLE_SCR": enabled,
                    "SCR_PHASE": phase,
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "NCCL_IB_HCA": "mlx5_0",
                    "GLOO_SOCKET_IFNAME": "eth0",
                }
                with patch.dict(os.environ, env, clear=True), patch(
                    "rtp_llm.distribute.distributed_server.get_master",
                    return_value=("192.0.2.10", "19000"),
                ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
                    server = DistributedServer(configs, wait_for_workers=False)
                    server.regist()
                    master_ip = "127.0.0.1" if active else "192.0.2.10"
                    rank_ip = "127.0.0.1" if active else "192.0.2.20"
                    self.assertEqual(server.master_ip, master_ip)
                    nccl = server.get_nccl_comm_config()
                    self.assertEqual(nccl.nccl_ip, master_ip)
                    self.assertEqual(nccl.tp_nccl_port, 18998)
                    self.assertEqual(nccl.dp_tp_nccl_port, 18990)
                    self.assertEqual(nccl.ffn_tp_nccl_port, 18995)
                    self.assertEqual(store.call_args.kwargs["host_name"], master_ip)
                    self.assertEqual(store.call_args.kwargs["port"], 18999)
                    self.assertEqual(store.call_args.kwargs["is_master"], rank == 0)
                    store.return_value.set.assert_called_once_with(
                        f"registry_rank_address_{rank}",
                        f"{rank_ip}:{server.worker_info.server_port}",
                    )
                    self.assertEqual(server.worker_info.ip, "192.0.2.20")
                    self.assertEqual(dict(os.environ), env)

    def test_dp_fanout_uses_loopback_only_for_single_node_templates(self):
        for nodes, enabled, phase in product(
            (1, 2), ("0", "1"), ("normal", "checkpoint", "restore")
        ):
            with self.subTest(nodes=nodes, enabled=enabled, phase=phase):
                configs = PyEnvConfigs()
                pc = configs.parallelism_config
                pc.world_size, pc.local_world_size, pc.tp_size = 4, 4 // nodes, 2
                members = [
                    WorkerInfo(
                        ip=f"192.0.2.{10 + i // pc.local_world_size}",
                        local_rank=i % pc.local_world_size,
                        world_rank=i,
                        name="",
                        server_port=19000,
                        worker_info_port_num=10,
                    )
                    for i in range(4)
                ]
                world = WorldInfo(members, members[0], members[0], nodes, True)
                active = enabled == "1" and phase != "normal" and nodes == 1
                with patch.dict(
                    os.environ,
                    {"RTPLLM_ENABLE_SCR": enabled, "SCR_PHASE": phase},
                    clear=True,
                ):
                    self.assertEqual(
                        get_dp_addrs_from_world_info(world, pc),
                        [
                            f"{'127.0.0.1' if active else members[i].ip}:{members[i].rpc_server_port}"
                            for i in (0, 2)
                        ],
                    )


if __name__ == "__main__":
    unittest.main()
