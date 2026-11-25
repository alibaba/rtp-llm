import json
import os
import socket
import unittest
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import copy
from datetime import timedelta
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import torch.distributed as dist
from torch.distributed import DistStoreError

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    get_local_workers,
    get_master,
)
from rtp_llm.distribute.host_bootstrap import prepare_host
from rtp_llm.distribute.worker_topology import WorkerTopology
from rtp_llm.test.utils.port_util import PortsContext


def _validate_worker_partition(config, port, counts):
    pc = config.parallelism_config
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=pc.world_size,
        rank=pc.world_rank,
        timeout=timedelta(seconds=15),
    )
    try:
        store = dist.distributed_c10d._get_default_store()
        group_timeout = store.timeout
        error = None
        if counts is not None:
            pc.pp_stage_layer_counts = counts
            try:
                WorkerTopology(config).validate_pp_partition_consistency(pc)
            except RuntimeError as exc:
                error = str(exc)
        if store.timeout != group_timeout:
            raise AssertionError("PP validation changed the process group's timeout")
        dist.barrier()
        return error
    finally:
        dist.destroy_process_group()


@patch.dict(os.environ)
class DistributedServerTest(unittest.TestCase):
    def hosts(self, local_size=2, pp_size=1):
        (store_port,) = PortsContext.reserve_for_test(self)
        base = store_port + 1
        configs = []
        for i in range(2):
            cfg = PyEnvConfigs()
            pc = cfg.parallelism_config
            pc.world_size = local_size * 2
            pc.local_world_size = local_size
            pc.tp_size = local_size
            pc.pp_size = pp_size
            pc.dp_size = 2 // pp_size
            pc.ep_size = 0
            set_parallelism_config(pc, i * local_size)
            sc = cfg.server_config
            sc.ip = f"127.0.0.{i + 1}"
            sc.start_port = base
            sc.worker_info_port_num = 10
            cfg.distribute_config.dist_comm_timeout = 2
            cfg.distribute_config.gang_timeout_min = 5 / 60
            cfg.distribute_config.gang_config_string = (
                f"name:worker_part0,ip:127.0.0.1,port:{base}"
            )
            configs.append(cfg)
        return configs

    def prepare(self, configs):
        with ThreadPoolExecutor(max_workers=len(configs)) as pool:
            results = list(pool.map(prepare_host, configs))
        self.assertEqual(results, [True] * len(configs))
        with socket.socket() as client:
            self.assertNotEqual(
                client.connect_ex(
                    ("127.0.0.1", configs[0].server_config.start_port - 1)
                ),
                0,
            )

    def test_host_layout_rejects_zero_or_nondividing_local_size(self):
        config = PyEnvConfigs()
        config.parallelism_config.world_size = 4
        for local_size in (0, 3):
            config.parallelism_config.local_world_size = local_size
            with (
                self.subTest(local_size=local_size),
                self.assertRaisesRegex(ValueError, "Invalid host layout"),
            ):
                DistributedServer(config)

    def test_two_hosts_register_two_records_for_eight_ranks(self):
        configs = self.hosts(local_size=4)
        configs[1].server_config.start_port += 200
        servers = [DistributedServer(config) for config in configs]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(DistributedServer.start, servers))
            for index, config in enumerate(configs):
                sc = config.server_config
                self.assertEqual(
                    servers[index].store.timeout,
                    timedelta(minutes=config.distribute_config.gang_timeout_min),
                )
                self.assertEqual(
                    json.loads(servers[0].store.get(f"startup_host/{index}")),
                    {
                        "layout": [8, 4],
                        "address": f"{sc.ip}:{sc.start_port}",
                    },
                )
            self.assertFalse(servers[0].store.check(["startup_host/2"]))

    def test_workers_expand_all_eight_addresses_after_store_closes(self):
        configs = self.hosts(local_size=4)
        configs[1].server_config.start_port += 200
        for config in configs:
            config.server_config.worker_info_port_num = 17
        self.prepare(configs)
        expected = []
        for config in configs:
            sc = copy(config.server_config)
            for slot in range(4):
                sc.set_local_rank(slot)
                expected.append((sc.ip, sc.server_port))
        for rank in range(8):
            cfg = configs[rank // 4]
            pc = cfg.parallelism_config
            set_parallelism_config(pc, rank)
            worker = WorkerTopology(cfg)
            worker.validate_pp_partition_consistency(pc)
            world = worker.world_info
            self.assertIs(world, cfg.distribute_config.world_info)
            self.assertEqual(len(world.host_addresses), 2)
            members = world.members(cfg)
            self.assertEqual([(m.ip, m.server_port) for m in members], expected)
            self.assertEqual(members[0].world_rank, 0)
            self.assertEqual(
                (members[rank].world_rank, members[rank].local_rank), (rank, rank % 4)
            )
            self.assertEqual(
                worker.get_nccl_init_port(), configs[0].server_config.start_port - 11
            )

    def test_missing_host_times_out_before_workers_start(self):
        root, _ = self.hosts()
        root.distribute_config.gang_timeout_min = 1 / 60
        with self.assertRaises(DistStoreError):
            prepare_host(root)
        self.assertIsNone(root.distribute_config.world_info)

    def test_registration_error_releases_the_startup_store(self):
        root, _ = self.hosts()
        server = DistributedServer(root)
        with (
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            patch.object(
                server, "bootstrap", side_effect=ValueError("registration failed")
            ),
            self.assertRaisesRegex(ValueError, "registration failed"),
            server.rendezvous(),
        ):
            self.fail("registration failed before the final exchange")
        store.assert_called_once()
        self.assertIsNone(server.store)

    def test_single_host_multi_rank_uses_local_addresses_without_a_store(self):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.world_size = pc.local_world_size = 4
        config.server_config.ip = "127.0.0.1"
        port = config.server_config.start_port
        with (
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            patch("socket.gethostname", return_value="127.0.0.42"),
            DistributedServer(config).rendezvous() as server,
        ):
            store.assert_not_called()
            self.assertEqual(server.world_info.host_addresses, (f"127.0.0.1:{port}",))
            self.assertEqual(server.world_info.collective_master, ("127.0.0.42", port))

    def test_multi_rank_worker_requires_prepared_world(self):
        root, _ = self.hosts()
        with self.assertRaises(RuntimeError):
            WorkerTopology(root)

    def test_single_rank_worker_needs_no_host_rendezvous(self):
        cfg = PyEnvConfigs()
        cfg.server_config.ip = "127.0.0.1"
        pc = cfg.parallelism_config
        pc.local_world_size = 4
        with patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
            self.assertTrue(prepare_host(cfg))
            worker = WorkerTopology(cfg)
            self.assertIs(worker.world_info, cfg.distribute_config.world_info)
            cfg.distribute_config.world_info = None
            self.assertEqual(WorkerTopology(cfg).world_info, worker.world_info)
            store.assert_not_called()
        world = worker.world_info
        (member,) = world.members(cfg)
        self.assertEqual((member.world_rank, member.local_rank), (0, 0))
        self.assertEqual(
            (member.ip, member.server_port), ("127.0.0.1", cfg.server_config.start_port)
        )
        self.assertEqual(world.master, (member.ip, member.server_port))
        worker.validate_pp_partition_consistency(pc)

    def validate_partitions(self, peer_counts):
        configs = self.hosts(local_size=1, pp_size=2)
        self.prepare(configs)
        if peer_counts is None:
            configs[0].distribute_config.dist_comm_timeout = 1
        (port,) = PortsContext.reserve_for_test(self)
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as pool:
            return list(
                pool.map(
                    _validate_worker_partition,
                    configs,
                    (port, port),
                    ([2, 3], peer_counts),
                )
            )

    def test_matching_pp_partitions_on_all_workers(self):
        self.assertEqual(self.validate_partitions([2, 3]), [None, None])

    def test_all_workers_reject_mismatched_pp_partitions(self):
        for error in self.validate_partitions([3, 2]):
            self.assertRegex(error, "inconsistent PP layer partitions")

    def test_missing_pp_worker_times_out_during_partition_validation(self):
        root_error, _ = self.validate_partitions(None)
        self.assertRegex(root_error, "PP partition snapshot from world_rank=1")

    def test_local_view_without_a_backend_uses_local_port_stride(self):
        cfg = self.hosts()[0]
        members = get_local_workers(cfg)
        self.assertEqual(
            [m.server_port for m in members],
            [cfg.server_config.start_port, cfg.server_config.start_port + 10],
        )

    def test_discovery_sources_resolve_original_master(self):
        root, _ = self.hosts()
        dc, pc = root.distribute_config, root.parallelism_config
        data = Path(__file__).with_name("testdata")
        dc.distribute_config_file = str(data / "parallel.json")
        self.assertEqual(get_master(dc, pc), ("11.161.48.116", "10000"))
        dc.distribute_config_file = ""
        self.assertEqual(
            get_master(dc, pc), ("127.0.0.1", str(root.server_config.start_port))
        )
        dc.gang_config_string = ""
        dc.leader_address = "127.0.0.2"
        self.assertEqual(get_master(dc, pc), ("127.0.0.2", ""))
        dc.leader_address = ""
        dc.gang_annocation_path = str(data / "annocation")
        self.assertEqual(get_master(dc, pc), ("33.115.125.211", ""))


if __name__ == "__main__":
    unittest.main()
