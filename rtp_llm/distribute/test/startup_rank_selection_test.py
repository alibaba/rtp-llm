import json
import multiprocessing
import os
import pickle
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from unittest.mock import Mock, call, patch

from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import report_worker
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.distribute.distributed_server import DistributedServer, get_master
from rtp_llm.distribute.startup_rank_selection import (
    STARTUP_RANK_SELECTION_ERROR_QPS,
    STARTUP_RANK_SELECTION_FALLBACK_SUCCESS_QPS,
    RankSelectionError,
    choose_hosts,
    exchange_host_health,
    select_startup_hosts,
)
from rtp_llm.distribute.worker_topology import WorkerTopology
from rtp_llm.metrics import AccMetrics
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.test.utils.port_util import PortsContext

UNHEALTHY_HOST_EVENT = call(
    STARTUP_RANK_SELECTION_ERROR_QPS, 1, {"is_serving": "false"}
)


def fallback_success_event(host_count_change):
    return call(
        STARTUP_RANK_SELECTION_FALLBACK_SUCCESS_QPS,
        1,
        {"is_serving": "false", "host_count_change": host_count_change},
    )


def host_configs(test_case, health, local=4, tp=2):
    # TCPStore binds wildcard, so each loopback host needs a distinct Store port.
    ports = [port + 1 for port in PortsContext.reserve_for_test(test_case, len(health))]
    result = []
    for index, healthy_ids in enumerate(health):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.world_size = pc.ep_size = len(health) * local
        pc.tp_size = tp
        pc.dp_size = pc.world_size // tp
        pc.world_rank = index * local
        pc.local_world_size = local
        sc = config.server_config
        sc.ip = f"127.0.0.{index + 1}"
        sc.start_port = ports[index]
        sc.worker_info_port_num = 10
        dc = config.distribute_config
        dc.enable_startup_rank_selection = True
        dc.gang_config_string = f"name:test_part0,ip:127.0.0.1,port:{ports[0]}"
        dc.gang_timeout_min = 5 / 60
        dc.startup_healthy_gpu_ids = healthy_ids
        result.append(config)
    return result


class SelectionTest(unittest.TestCase):
    def test_each_launcher_logs_its_plan(self):
        pc = PyEnvConfigs().parallelism_config
        pc.world_size = pc.ep_size = 4
        pc.local_world_size = pc.dp_size = 2
        pc.tp_size = 2
        hosts = ["host-a:1000", "host-b:1000"]
        for health, selected in (([True, True], [0, 1]), ([False, True], [1])):
            with (
                self.subTest(health=health),
                patch.object(report_worker, "report_once") as report,
            ):
                plans = []
                for rank in (0, 2):
                    pc.world_rank = rank
                    with self.assertLogs(level="INFO") as logs:
                        self.assertEqual(
                            select_startup_hosts(health, pc, hosts), selected
                        )
                    (record,) = logs.records
                    header, plan = record.getMessage().split("\n", 1)
                    self.assertIn(f"WORLD=4 -> {len(selected) * 2}", header)
                    self.assertIn(f"ORIGINAL_RANK={rank}", header)
                    plans.append(plan)
                self.assertEqual(plans[0], plans[1])
                expected_events = (
                    []
                    if all(health)
                    else [UNHEALTHY_HOST_EVENT, fallback_success_event("[2->1]")]
                )
                self.assertEqual(report.call_args_list, expected_events)

    def test_health_environment_contract(self):
        with patch.dict(os.environ, {}, clear=True):
            dc = setup_args([]).distribute_config
            self.assertFalse(dc.enable_startup_rank_selection)
            self.assertIsNone(dc.startup_healthy_gpu_ids)
        with patch.dict(
            os.environ,
            {"ENABLE_STARTUP_RANK_SELECTION": "1", "STARTUP_HEALTHY_GPU_IDS": ""},
            clear=True,
        ):
            dc = setup_args([]).distribute_config
            self.assertTrue(dc.enable_startup_rank_selection)
            self.assertEqual(dc.startup_healthy_gpu_ids, [])
            self.assertEqual(
                setup_args(
                    ["--startup_healthy_gpu_ids", "0,2"]
                ).distribute_config.startup_healthy_gpu_ids,
                [0, 2],
            )
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                setup_args(["--startup_healthy_gpu_ids", "0,bad"])

    def test_unsupported_shape_fails_only_when_a_host_needs_exclusion(self):
        for local, tp, health, message in (
            (6, 4, [True, False, True, True], "complete local TP groups"),
            (8, 2, [False], "No fully healthy host"),
        ):
            with self.subTest(local=local, tp=tp):
                pc = host_configs(self, health, local=local, tp=tp)[
                    0
                ].parallelism_config
                self.assertEqual(
                    choose_hosts([True] * len(health), pc),
                    list(range(len(health))),
                )
                with self.assertRaisesRegex(RankSelectionError, message):
                    choose_hosts(health, pc)

    def test_host_selection_local_capacity_gate(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        mask = "7,5,3,1,6,4,2,0"
        for enabled, pp, tp, health, should_fail in (
            (True, 2, 2, [None, None], False),  # PP * TP == LOCAL_WORLD_SIZE
            (True, 2, 4, [None, None], True),
            (True, 2, 4, [None, []], True),
            (True, 1, 8, [None, None], True),
            (False, 2, 4, [None, []], False),
            (False, 1, 8, [None, None], False),
        ):
            with (
                self.subTest(enabled=enabled, pp=pp, tp=tp, health=health),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}),
                patch.object(report_worker, "report_once") as report,
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                configs = host_configs(self, health, local=4, tp=tp)
                for config in configs:
                    config.distribute_config.enable_startup_rank_selection = enabled
                    pc = config.parallelism_config
                    pc.pp_size = pp
                    pc.dp_size = pc.world_size // (pp * tp)
                    pc.ep_size = pc.world_size // pp
                futures = [pool.submit(prepare_host, config) for config in configs]
                if should_fail:
                    for future in futures:
                        with self.assertRaisesRegex(
                            RankSelectionError,
                            r"PP_SIZE \* TP_SIZE <= LOCAL_WORLD_SIZE",
                        ):
                            future.result(timeout=10)
                    self.assertEqual(
                        report.call_args_list,
                        [UNHEALTHY_HOST_EVENT] if [] in health else [],
                    )
                    self.assertTrue(
                        all(
                            config.distribute_config.world_info is None
                            for config in configs
                        )
                    )
                    continue
                self.assertEqual(
                    [future.result(timeout=10) for future in futures], [True, True]
                )
                report.assert_not_called()
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)
                for index, config in enumerate(configs):
                    pc = config.parallelism_config
                    self.assertEqual((pc.world_size, pc.world_rank), (8, index * 4))
                    self.assertEqual(
                        (
                            pc.local_world_size,
                            pc.pp_size,
                            pc.tp_size,
                            pc.dp_size,
                            pc.ep_size,
                        ),
                        (4, pp, tp, 8 // (pp * tp), 8 // pp),
                    )
                    sc = config.server_config
                    self.assertEqual(
                        config.distribute_config.world_info.host_addresses[index],
                        f"{sc.ip}:{sc.start_port}",
                    )

    def test_shrink_passes_original_local_slots_and_ports_to_launcher(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        def peer(config):
            # Synchronize peers without writing another host's rank into os.environ.
            with DistributedServer(config).rendezvous() as server:
                exchange_host_health(config, server)

        configs = host_configs(self, [None, [0, 1, 3], None, None])
        excluded, active = configs[1:3]
        excluded_port = excluded.server_config.start_port
        active_port = active.server_config.start_port
        mask = "7,5,3,1,6,4,2,0"
        with (
            tempfile.TemporaryDirectory() as checkpoint,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}),
            ThreadPoolExecutor(max_workers=3) as pool,
        ):
            # Host selection keeps 12 ranks even with metadata from an 8-rank checkpoint.
            Path(checkpoint, "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "is_ft_style_weight": True,
                        "__env__params__": {"TP_SIZE": 2, "DP_SIZE": 4, "EP_SIZE": 8},
                    }
                )
            )
            for config in configs:
                config.model_args.ckpt_path = checkpoint
            os.environ.pop("LOCAL_WORLD_SIZE", None)
            peers = [pool.submit(peer, config) for config in (configs[0], configs[3])]
            excluded_result = pool.submit(prepare_host, excluded)
            self.assertTrue(prepare_host(active))
            self.assertFalse(excluded_result.result())
            for future in peers:
                future.result()
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], mask)
            self.assertNotIn("LOCAL_WORLD_SIZE", os.environ)

        excluded_pc = excluded.parallelism_config
        self.assertEqual((excluded_pc.world_size, excluded_pc.world_rank), (16, 4))
        self.assertEqual(excluded.server_config.start_port, excluded_port)
        pc = active.parallelism_config
        self.assertEqual(
            (pc.world_size, pc.world_rank, pc.dp_size, pc.ep_size), (12, 4, 6, 12)
        )
        self.assertEqual((pc.tp_size, pc.local_world_size), (2, 4))
        members = active.distribute_config.world_info.members(active)
        self.assertEqual(
            [(m.world_rank, m.local_rank, m.ip, m.server_port) for m in members[4:8]],
            [
                (4 + i, i, active.server_config.ip, active_port + 10 * i)
                for i in range(4)
            ],
        )

    def test_ep_one_rejects_host_exclusion(self):
        health = [False, True, True]
        for tp in (4, 2):
            with self.subTest(tp=tp):
                pc = host_configs(self, health, local=4, tp=tp)[0].parallelism_config
                pc.ep_size = 1
                with self.assertRaisesRegex(
                    RankSelectionError, "EP_SIZE different from 1"
                ):
                    choose_hosts(health, pc)


@patch.dict(os.environ)
class RendezvousTest(unittest.TestCase):
    def assert_port_released(self, port):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sock.listen(1)

    def test_one_bad_gpu_removes_its_two_card_host(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        for host_count in (2, 4):
            with self.subTest(host_count=host_count):
                configs = host_configs(
                    self,
                    [[0]] + [None] * (host_count - 1),
                    local=2,
                    tp=1,
                )
                reporting_threads = []
                with (
                    patch.object(
                        report_worker,
                        "report_once",
                        side_effect=lambda *_: reporting_threads.append(
                            threading.get_ident()
                        ),
                    ) as report,
                    ThreadPoolExecutor(max_workers=host_count - 1) as pool,
                ):
                    peers = [pool.submit(prepare_host, cfg) for cfg in configs[1:]]
                    active = [prepare_host(configs[0])] + [
                        future.result() for future in peers
                    ]
                # The original host 0 reports even when it is excluded.
                self.assertEqual(reporting_threads, [threading.get_ident()] * 2)
                self.assertEqual(
                    report.call_args_list,
                    [
                        UNHEALTHY_HOST_EVENT,
                        fallback_success_event(f"[{host_count}->{host_count - 1}]"),
                    ],
                )
                self.assertEqual(active, [False] + [True] * (host_count - 1))
                self.assertIsNone(configs[0].distribute_config.world_info)
                root = configs[1].server_config
                world_size = 2 * (host_count - 1)
                for index, config in enumerate(configs[1:]):
                    pc = config.parallelism_config
                    self.assertEqual(
                        config.distribute_config.world_info.master,
                        (root.ip, root.start_port),
                    )
                    self.assertEqual(
                        (
                            pc.world_rank,
                            pc.world_size,
                            pc.local_world_size,
                            pc.tp_size,
                            pc.dp_size,
                            pc.ep_size,
                        ),
                        (2 * index, world_size, 2, 1, world_size, world_size),
                    )

    def test_each_host_selects_locally_from_the_same_health_vector(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        configs = host_configs(self, [None, [], None])
        local_inputs = []

        def choose_locally(health, pc):
            local_inputs.append((pc.world_rank, tuple(health)))
            return choose_hosts(health, pc)

        with (
            patch(
                "rtp_llm.distribute.startup_rank_selection.choose_hosts",
                side_effect=choose_locally,
            ),
            ThreadPoolExecutor(max_workers=len(configs)) as pool,
        ):
            self.assertEqual(list(pool.map(prepare_host, configs)), [True, False, True])
        self.assertCountEqual(
            local_inputs, [(rank, (True, False, True)) for rank in (0, 4, 8)]
        )

    def test_retained_root_releases_store_after_excluding_middle_host(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        configs = host_configs(self, [None, [], None])
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(prepare_host, configs))

        self.assertEqual(results, [True, False, True])
        self.assert_port_released(configs[0].server_config.start_port - 1)
        expected = [
            (cfg.server_config.ip, cfg.server_config.start_port + local * 10)
            for cfg in (configs[0], configs[2])
            for local in range(4)
        ]
        for cfg in (configs[0], configs[2]):
            worker = WorkerTopology(cfg)
            members = worker.world_info.members(cfg)
            self.assertEqual([(m.ip, m.server_port) for m in members], expected)
            self.assertEqual(
                worker.get_nccl_comm_config().nccl_ip, configs[0].server_config.ip
            )
            self.assertEqual(
                worker.get_nccl_init_port(), configs[0].server_config.start_port - 11
            )

    def test_excluded_discovery_root_releases_store_without_replacement(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        configs = host_configs(self, [[0, 1, 2], None, [0, 1, 2, 3]])
        # The new NCCL master can start even when its base-1 port is occupied.
        with socket.socket() as occupied, ThreadPoolExecutor(max_workers=3) as pool:
            occupied.bind(("", configs[1].server_config.start_port - 1))
            occupied.listen(1)
            results = list(pool.map(prepare_host, configs))

        self.assertEqual(results, [False, True, True])
        self.assert_port_released(configs[0].server_config.start_port - 1)
        addresses = []
        for index in (1, 2):
            cfg = configs[index]
            world = cfg.distribute_config.world_info
            cfg.distribute_config.world_info = pickle.loads(pickle.dumps(world))
            pc = cfg.parallelism_config
            # Original hosts 1 and 2 now own 0..3 and 4..7 respectively.
            self.assertEqual(pc.world_rank, (index - 1) * 4)
            self.assertEqual(
                (
                    pc.world_size,
                    pc.local_world_size,
                    pc.tp_size,
                    pc.dp_size,
                    pc.ep_size,
                ),
                (8, 4, 2, 4, 8),
            )
            set_parallelism_config(pc, pc.world_rank + 2)
            worker = WorkerTopology(cfg)
            world = worker.world_info
            members = world.members(cfg)
            local = members[pc.world_rank]
            addresses.append((local.ip, local.server_port))
            self.assertEqual(len(world.host_addresses), 2)
            new_master = configs[1].server_config
            self.assertEqual(world.master, (new_master.ip, new_master.start_port))
            self.assertEqual(
                get_master(cfg.distribute_config, pc),
                (new_master.ip, str(new_master.start_port)),
            )
            self.assertEqual((members[0].ip, members[0].server_port), world.master)
            self.assertEqual(worker.get_nccl_comm_config().nccl_ip, members[0].ip)
            self.assertEqual(
                worker.get_nccl_init_port(), configs[1].server_config.start_port - 11
            )
            self.assertEqual([m.world_rank for m in members], list(range(8)))
            self.assertEqual([m.local_rank for m in members], [0, 1, 2, 3] * 2)
        self.assertEqual(
            addresses,
            [
                (cfg.server_config.ip, cfg.server_config.start_port + 20)
                for cfg in configs[1:]
            ],
        )

    def test_single_host_workers_use_the_full_topology_with_or_without_selection(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        for enabled in (False, True):
            with self.subTest(selection=enabled):
                (config,) = host_configs(self, [None], local=4)
                config.distribute_config.enable_startup_rank_selection = enabled
                self.assertTrue(prepare_host(config))

                set_parallelism_config(config.parallelism_config, 3)
                world = WorkerTopology(config).world_info
                self.assertEqual(len(world.host_addresses), 1)
                members = world.members(config)
                self.assertEqual([m.world_rank for m in members], [0, 1, 2, 3])
                self.assertEqual(
                    members[config.parallelism_config.world_rank].local_rank, 3
                )

    def test_single_rank_checks_only_its_assigned_gpu(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        config = PyEnvConfigs()
        config.parallelism_config.local_world_size = 4
        dc = config.distribute_config
        dc.enable_startup_rank_selection = True
        dc.startup_healthy_gpu_ids = [0]
        with self.assertLogs(level="INFO") as logs:
            self.assertTrue(prepare_host(config))
        self.assertIn("Startup rank reorder | WORLD=1 -> 1", "\n".join(logs.output))
        self.assertEqual(len(dc.world_info.members(config)), 1)

    def test_feature_off_ignores_fault_input_after_rendezvous(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        configs = host_configs(self, [[], []])
        for config in configs:
            config.distribute_config.enable_startup_rank_selection = False
            config.parallelism_config.pp_size = config.parallelism_config.dp_size = 2
            config.parallelism_config.ep_size = 4
        # A multi-NIC deployment can advertise distinct communication and serving endpoints.
        discovery_ip = "127.0.0.42"
        discovery_base = PortsContext.reserve_for_test(self)[0] + 1
        for config in configs:
            config.distribute_config.gang_config_string = (
                f"name:test_part0,ip:{discovery_ip},port:{discovery_base}"
            )
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(prepare_host, configs))
        self.assertEqual(results, [True, True])
        self.assert_port_released(discovery_base - 1)
        expected_addresses = tuple(
            f"{cfg.server_config.ip}:{cfg.server_config.start_port}" for cfg in configs
        )
        for index, config in enumerate(configs):
            world = config.distribute_config.world_info
            self.assertEqual(world.host_addresses, expected_addresses)
            serving_master = configs[0].server_config
            self.assertEqual(
                world.master, (serving_master.ip, serving_master.start_port)
            )
            self.assertEqual(config.parallelism_config.world_rank, index * 4)
            self.assertEqual(config.parallelism_config.world_size, 8)
            worker = WorkerTopology(config)
            self.assertEqual(
                worker.world_info.members(config)[0].ip, configs[0].server_config.ip
            )
            self.assertEqual(worker.get_nccl_comm_config().nccl_ip, discovery_ip)
            self.assertEqual(worker.get_nccl_init_port(), discovery_base - 11)
        remote = configs[1].server_config
        member = WorkerTopology(configs[0]).world_info.members(configs[0])[5]
        self.assertEqual(
            (member.ip, member.server_port), (remote.ip, remote.start_port + 10)
        )

    def test_unhealthy_hosts_report_once_even_when_selection_fails(self):
        from rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor import KMonitor
        from rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker import (
            ReportWorker,
        )
        from rtp_llm.distribute.host_bootstrap import prepare_host
        from rtp_llm.metrics.kmonitor_metric_reporter import MetricReporter

        with patch.object(ReportWorker, "start"):
            worker = ReportWorker()
        worker.flume = Mock()
        with patch(
            "rtp_llm.aios.kmonitor.python_client.kmonitor.kmonitor.report_worker",
            worker,
        ):
            reporter = MetricReporter(KMonitor())
        reporter.init()
        reporter.report(AccMetrics.QPS_METRIC)
        with (
            patch(
                "rtp_llm.aios.kmonitor.python_client.kmonitor.report_worker.report_worker",
                worker,
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            for health, message in (
                ([[0, 1, 2], [1, 2, 3]], "No fully healthy host"),
                ([None, []], "Removing a host requires PP_SIZE=1"),
            ):
                with self.subTest(health=health):
                    configs = host_configs(self, health)
                    for cfg in configs:
                        # PP=2 fits each host, but cannot shrink after losing one.
                        pc = cfg.parallelism_config
                        pc.pp_size = pc.dp_size = 2
                        pc.ep_size = 4
                    worker.flume.send_batch.reset_mock()
                    with self.assertNoLogs(level="ERROR"):
                        futures = [pool.submit(prepare_host, cfg) for cfg in configs]
                        for future in futures:
                            with self.assertRaisesRegex(RankSelectionError, message):
                                future.result(timeout=10)
                    worker.do_report()
                    worker.flume.send_batch.assert_called_once()
                    (event,) = worker.flume.send_batch.call_args.args[0]
                    name, timestamp, value, *tags = event.body.decode().split()
                    self.assertEqual(name, STARTUP_RANK_SELECTION_ERROR_QPS)
                    self.assertGreater(int(timestamp), 0)
                    self.assertEqual(float(value), 1)
                    self.assertFalse(
                        any(tag.startswith(("outcome=", "reason=")) for tag in tags)
                    )
                    self.assertIn("is_serving=false", tags)

    def test_rendezvous_cannot_return_before_all_original_hosts_register(self):
        from rtp_llm.distribute.host_bootstrap import prepare_host

        configs = host_configs(self, [None, None])
        configs[0].distribute_config.gang_timeout_min = 1 / 60
        with (
            patch.object(report_worker, "report_once") as report,
            self.assertRaises(RuntimeError),
        ):
            prepare_host(configs[0])
        report.assert_not_called()

    def test_root_waits_for_last_reader_before_releasing_store(self):
        from torch.distributed.elastic.utils.store import get_all

        from rtp_llm.distribute.host_bootstrap import prepare_host

        for enabled, final_read in (
            (False, "rtp_llm.distribute.distributed_server.get_all"),
            (True, "torch.distributed.elastic.utils.store.get_all"),
        ):
            with self.subTest(selection=enabled):
                configs = host_configs(self, [None, None])
                for config in configs:
                    config.distribute_config.enable_startup_rank_selection = enabled
                readers = threading.Barrier(3, timeout=5)
                allow_peer_read = threading.Event()

                def delay_peer_read(store, rank, prefix, world_size):
                    # Both hosts have written their record, but the peer has not read yet.
                    readers.wait()
                    if rank == 1:
                        self.assertTrue(allow_peer_read.wait(5))
                    return get_all(store, rank, prefix, world_size)

                with (
                    patch(final_read, delay_peer_read),
                    ThreadPoolExecutor(max_workers=2) as pool,
                ):
                    root, peer = [pool.submit(prepare_host, cfg) for cfg in configs]
                    try:
                        readers.wait()
                        with self.assertRaises(FutureTimeout):
                            root.result(timeout=0.2)
                    finally:
                        allow_peer_read.set()
                    self.assertTrue(root.result(timeout=5))
                    self.assertTrue(peer.result(timeout=5))
                self.assert_port_released(configs[0].server_config.start_port - 1)

    def test_missing_health_report_prevents_root_from_starting_alone(self):
        from torch.distributed import DistStoreError

        from rtp_llm.distribute.host_bootstrap import prepare_host

        def skip_health_exchange(config):
            server = DistributedServer(config)
            server.start()

        configs = host_configs(self, [None, None])
        for config in configs:
            config.distribute_config.gang_timeout_min = 1 / 60
        with (
            patch.object(report_worker, "report_once") as report,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            peer = pool.submit(skip_health_exchange, configs[1])
            with self.assertRaises(DistStoreError):
                prepare_host(configs[0])
            peer.result(timeout=5)
            report.assert_not_called()


class UnhealthyHostTest(unittest.TestCase):
    def test_excluded_host_enters_standby_before_engine_or_frontend_startup(self):
        from rtp_llm import start_server

        config = host_configs(self, [[], None])[0]
        with (
            patch.object(start_server, "setup_args", return_value=config),
            patch.object(start_server, "prepare_host", return_value=False),
            patch.object(start_server, "setup_and_configure_server") as configure,
            patch.object(start_server, "start_server") as start_services,
            patch("rtp_llm.server.unhealthy_server.serve_unhealthy_host") as standby,
        ):
            start_server.main()
            standby.assert_called_once_with(config)
            configure.assert_not_called()
            start_services.assert_not_called()

    def test_excluded_host_serves_only_http_on_original_frontend_ports(self):
        from rtp_llm.server.unhealthy_server import serve_unhealthy_host
        from rtp_llm.start_server import check_server_health

        config = host_configs(self, [None, []])[1]
        base = PortsContext.reserve_for_test(self, 32)[0]
        config.server_config.start_port = base
        server = multiprocessing.get_context("fork").Process(
            target=serve_unhealthy_host, args=(config,)
        )
        server.start()
        self.addCleanup(server.join, 5)
        self.addCleanup(server.terminate)
        deadline = time.monotonic() + 10
        while server.is_alive() and not check_server_health(base, "/liveness"):
            self.assertLess(time.monotonic(), deadline, "HTTP liveness did not start")
            time.sleep(0.05)

        for port in (base, base + 20):
            self.assertTrue(check_server_health(port, "/liveness"))
        for offset in (1, 8, 10, 11, 21, 28, 30, 31):
            with self.subTest(offset=offset), self.assertRaises(ConnectionRefusedError):
                socket.create_connection(
                    ("127.0.0.1", base + offset), timeout=1
                ).close()

        paths = {
            "/liveness": 200,
            "/liveness?probe=1": 200,
            "/liveness/": 500,
            "/": 500,
            "/health": 500,
            "/health_check": 500,
            "/liveness_internal": 500,
            "/frontend_health": 500,
            "/status": 500,
            "/GraphService/cm2_status": 500,
            "/SearchService/cm2_status": 500,
            "/worker_status": 500,
            "/rtp_llm/worker_status": 500,
            "/v1/chat/completions": 500,
            "/unknown?probe=/liveness": 500,
            "/docs": 500,
            "/redoc": 500,
            "/openapi.json": 500,
        }
        for path, expected in paths.items():
            for method in ("GET", "POST", "HEAD", "PUT", "OPTIONS", "PROPFIND"):
                with self.subTest(path=path, method=method):
                    # Read the wire directly; HTTP clients discard stray HEAD bodies.
                    with socket.create_connection(
                        ("127.0.0.1", base), timeout=3
                    ) as conn:
                        conn.sendall(
                            (
                                f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n"
                                "Content-Length: 0\r\nConnection: close\r\n\r\n"
                            ).encode()
                        )
                        with conn.makefile("rb") as stream:
                            headers, _, body = stream.read().partition(b"\r\n\r\n")
                    self.assertEqual(int(headers.split()[1]), expected)
                    if method == "HEAD":
                        self.assertEqual(body, b"")
                    elif expected == 200:
                        self.assertEqual(json.loads(body), "ok")


if __name__ == "__main__":
    unittest.main()
