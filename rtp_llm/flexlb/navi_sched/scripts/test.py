import asyncio
import gc
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from navi_sched import *
from navi_sched.adapters.vllm import (
    NaviSchedAdapterConfig as VLLMNaviSchedAdapterConfig,
    NaviSchedClient as VLLMNaviSchedClient,
    NaviSchedEndpointRouter,
)
from navi_sched.master_process import (
    MasterProcessConfig,
    load_master_config,
    run_master,
)
from navi_sched.util import (
    DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT,
    makeServiceConfig,
    naviSchedModule,
)

navi_sched = naviSchedModule()

class TestNaviSchedPackage(unittest.TestCase):

    def tearDown(self):
        # 强制启动垃圾回收，确保 LSAN 不会因为延迟释放报内存泄漏
        gc.collect()

    def test_navi_sched_role_modules_import_independently(self):
        code = "from navi_sched.client import NaviSchedClient;" \
               "from navi_sched.master import NaviSchedMaster;" \
               "from navi_sched.agent import NaviSchedAgent;" \
               "print(NaviSchedClient.__name__, NaviSchedMaster.__name__," \
               " NaviSchedAgent.__name__)"
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ.copy(),
            cwd="/tmp",
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)
        self.assertIn("NaviSchedClient", result.stdout)

    def test_navi_sched_client_master_agent_interface(self):
        options = {
            "enable_kmon": False,
            "gig": {"grpc_port": 0},
        }
        client = NaviSchedClient("client_biz", "master_biz", **options)
        master = NaviSchedMaster("master_biz", **options)
        agent = NaviSchedAgent(
            "agent_biz", server_port=18000, total_blocks=1024,
            block_size=256, linear_step=1,
            hybrid_tail_required=False, **options)
        client.init()
        master.init()
        agent.init()
        request = NaviSchedRequest(
            prompt_token_count=512,
            block_hashes=[11, 22],
            request_id=1,
        )
        self.assertEqual(1, request.request_id)
        self.assertEqual(512, request.prompt_token_count)
        self.assertEqual([11, 22], request.block_hashes)
        auto_master = NaviSchedMaster("master_biz", **options)
        auto_master.stop()
        agent.stop()
        master.stop()
        client.stop()

    def test_navi_sched_service_config(self):
        config = navi_sched.NaviSchedConfig()
        self.assertTrue(config.wait_no_query)
        config.name = "role"
        config.enable_kmon = False
        config.wait_no_query = False
        gig = navi_sched.NaviSchedGigConfig()
        gig.grpc_port = 12345
        config.gig = gig
        self.assertEqual("role", config.name)
        self.assertEqual("./logs/navi_sched.log", config.log_file_name)
        self.assertEqual("INFO", config.log_level)
        self.assertFalse(config.enable_kmon)
        self.assertEqual(1000, config.kmon_report_interval_ms)
        self.assertEqual(10, config.batch_timeout_ms)
        self.assertEqual(60000, config.closed_batch_keep_ms)
        self.assertEqual(4, config.latency_fit_batch_size)
        self.assertEqual(100, config.latency_cold_start_fit_epochs)
        self.assertEqual(2000, config.latency_history_limit)
        self.assertEqual(256, config.latency_persist_interval)
        self.assertEqual(10000, config.latency_persist_interval_ms)
        self.assertEqual("", config.latency_model_state_file)
        self.assertTrue(config.enable_snapshot)
        self.assertEqual("navi_sched", config.snapshot_dir)
        self.assertEqual(2000, config.snapshot_interval_ms)
        self.assertEqual(200, config.snapshot_keep_count)
        self.assertFalse(config.enable_vis)
        self.assertFalse(config.wait_no_query)
        self.assertEqual(12345, config.gig.grpc_port)

    @mock.patch("navi_sched.master.RoleFacade.__init__", return_value=None)
    @mock.patch("navi_sched.master.makeRuntimeConfig")
    @mock.patch("navi_sched.master.makeServiceConfig")
    @mock.patch("navi_sched.master.naviSchedModule")
    def test_navi_sched_master_snapshot_keep_count(
        self,
        navi_sched_module,
        make_service_config,
        make_runtime_config,
        role_facade_init,
    ):
        native_module = navi_sched_module.return_value
        native_module.NaviSchedMaster.return_value = mock.sentinel.master
        service_options = []
        make_service_config.side_effect = (
            lambda module, name, options: service_options.append(dict(options))
            or mock.sentinel.service_config
        )

        NaviSchedMaster("master_biz")
        options = service_options[-1]
        self.assertEqual(
            DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT,
            options.get("snapshot_keep_count"),
            options,
        )

        NaviSchedMaster("master_biz", snapshot_keep_count=17)
        options = service_options[-1]
        self.assertEqual(17, options.get("snapshot_keep_count"), options)

    def test_navi_sched_all_service_config_options(self):
        options = {
            "log_file_name": "role.log",
            "log_level": "DEBUG",
            "enable_kmon": False,
            "kmon_service_name": "role_service",
            "kmon_metrics_path": "/metrics",
            "kmon_tags": {"zone": "test"},
            "kmon_monitor_name_prefix": "prefix",
            "kmon_report_interval_ms": 1500,
            "gig": {
                "heartbeat_thread_num": 7,
                "heartbeat_queue_size": 2048,
                "log_prefix": "gig",
                "snapshot_log_dir": "/gig_snapshot",
                "grpc_port": 12345,
                "grpc_io_thread_num": 9,
            },
            "wait_no_query": False,
            "batch_timeout_ms": 11,
            "closed_batch_keep_ms": 12000,
            "latency_fit_batch_size": 8,
            "latency_cold_start_fit_epochs": 20,
            "latency_history_limit": 3000,
            "latency_persist_interval": 32,
            "latency_persist_interval_ms": 4000,
            "latency_model_state_file": "/latency/model.state",
            "enable_snapshot": False,
            "snapshot_dir": "/snapshot",
            "snapshot_interval_ms": 3000,
            "snapshot_keep_count": 12,
            "enable_vis": True,
        }
        config = makeServiceConfig(navi_sched, "role", options)
        self.assertEqual("role", config.name)
        gig_options = options.pop("gig")
        for name, value in options.items():
            self.assertEqual(value, getattr(config, name))
        for name, value in gig_options.items():
            self.assertEqual(value, getattr(config.gig, name))

    def test_navi_sched_role_snapshots(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            current_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                options = {
                    "enable_kmon": False,
                    "gig": {"grpc_port": 0},
                    "enable_snapshot": True,
                    "snapshot_dir": "navi_sched",
                    "snapshot_interval_ms": 60000,
                }
                client = NaviSchedClient(
                    "client_biz", "master_biz", name="host_0", **options)
                master = NaviSchedMaster(
                    "master_biz", name="host_1", **options)
                agent = NaviSchedAgent(
                    "agent_biz", name="host_2", total_blocks=1024,
                    block_size=256, linear_step=1,
                    hybrid_tail_required=False, **options)
                client.init()
                master.init()
                agent.init()
                snapshot_root = os.path.join(temp_dir, "navi_sched")
                for _ in range(100):
                    if all(os.path.isdir(os.path.join(snapshot_root, name))
                           for name in ("host_0", "host_1", "host_2")):
                        break
                    time.sleep(0.01)
                agent.stop()
                master.stop()
                client.stop()
            finally:
                os.chdir(current_dir)

            expected = {
                "host_0": (
                    "  type: client",
                    "  master_biz_name: master_biz",
                    "  schedule_count: 0",
                    "  schedule_success_count: 0",
                ),
                "host_1": (
                    "  type: master",
                    "  biz_name: master_biz",
                    "  schedule_graph_running: true",
                    "  max_wait_schedule_count: 30",
                    "  max_wait_schedule_time_ms: 10",
                    "Agents\n  count: 0",
                ),
                "host_2": (
                    "  type: agent",
                    "  inflight_requests: 0",
                    "  cached_block_count: 0",
                ),
            }
            for role_name, expected_lines in expected.items():
                snapshot_dir = os.path.join(
                    temp_dir, "navi_sched", role_name)
                files = [
                    name for name in os.listdir(snapshot_dir)
                    if name.endswith("_snapshot")
                ]
                self.assertEqual(1, len(files))
                with open(os.path.join(snapshot_dir, files[0])) as stream:
                    snapshot = stream.read()
                self.assertIn("Role\n", snapshot)
                self.assertIn(f"  name: {role_name}\n", snapshot)
                self.assertIn("  started: true\n", snapshot)
                self.assertRegex(snapshot, r"  grpc_port: [1-9][0-9]*")
                for expected_line in expected_lines:
                    self.assertIn(expected_line, snapshot)

    def test_navi_sched_protocol_bindings(self):
        self.assertEqual(
            "nonlinear_prefill", navi_sched.get_latency_formula_name(1))
        target = navi_sched.NaviSchedTarget()
        target.biz_name = "model_biz"
        target.address = "127.0.0.1:1234"
        self.assertEqual("model_biz", target.biz_name)

    def test_navi_sched_master_process_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "master.json")
            expected = {
                "thread_num": 64,
                "local_nodes": [["agent", "127.0.0.1", 18000]],
            }
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump(expected, stream)
            config = MasterProcessConfig.from_env(
                {
                    "NAVI_SCHED_MASTER_BIZ_NAME": "master",
                    "NAVI_SCHED_MASTER_CONFIG_FILE": config_path,
                    "NAVI_SCHED_MASTER_GRPC_PORT": "19000",
                    "NAVI_SCHED_MASTER_ENABLE_KMON": "false",
                }
            )

            config.validate()
            self.assertEqual("master", config.master_biz_name)
            self.assertEqual("master_snapshot", config.name)
            self.assertEqual(".", config.snapshot_dir)
            self.assertEqual(19000, config.grpc_port)
            self.assertFalse(config.enable_kmon)
            self.assertEqual(expected, load_master_config(config_path))

    @mock.patch("navi_sched.master_process.wait_for_stop")
    @mock.patch("navi_sched.master_process.NaviSchedMaster")
    def test_navi_sched_master_process_uses_runtime_directory(
        self, master_class, wait_for_stop
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "master.json")
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump({"thread_num": 128}, stream)
            config = MasterProcessConfig(
                master_biz_name="master",
                config_file=config_path,
                snapshot_dir="runtime",
                enable_kmon=False,
            )
            master = master_class.return_value.__enter__.return_value
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            try:
                run_master(config)
            finally:
                os.chdir(original_dir)
            options = master_class.call_args.kwargs
            self.assertEqual(
                config.snapshot_dir, options["gig"]["snapshot_log_dir"]
            )
            master.init.assert_called_once_with()
            wait_for_stop.assert_called_once_with()

    def test_vllm_endpoint_router_preserves_request_semantics(self):
        client = mock.Mock()
        decision = object()
        client.schedule.return_value = decision
        router = NaviSchedEndpointRouter(
            client,
            "model_biz",
            timeout_ms=1200,
        )
        result = router.route(
            [1, 2, 3],
            request_id=1,
            block_hashes=[11, 22],
        )
        self.assertIs(decision, result)
        client.schedule.assert_called_once_with(
            3,
            block_hashes=[11, 22],
            block_size=16,
            forward_payload=b"",
            target_biz_name="model_biz",
            timeout_ms=1200,
            request_id=1,
        )
        client.schedule.reset_mock()
        self.assertIs(
            decision,
            asyncio.run(router.route_async([4], request_id=2)),
        )

    def test_vllm_endpoint_router_maps_generate_request_safely(self):
        client = mock.Mock()
        decision = object()
        client.schedule.return_value = decision
        router = NaviSchedEndpointRouter(client, "model_biz")
        request = type(
            "GenerateRequest",
            (),
            {
                "token_ids": [1, 2, 3],
                "request_id": 1,
            },
        )()

        result = router.route_generate_request(
            request,
            request_id=3,
        )

        self.assertIs(decision, result)
        client.schedule.assert_called_once_with(
            3,
            block_hashes=[],
            block_size=16,
            forward_payload=b"",
            target_biz_name="model_biz",
            timeout_ms=1000,
            request_id=3,
        )

        exact_decision = object()
        client.schedule.return_value = exact_decision
        result = router.route_generate_request(
            request,
            block_hashes=[101],
        )
        self.assertIs(exact_decision, result)
        self.assertEqual(2, client.schedule.call_count)
        client.schedule.assert_called_with(
            3,
            block_hashes=[101],
            block_size=16,
            forward_payload=b"",
            target_biz_name="model_biz",
            timeout_ms=1000,
            request_id=1,
        )

    def test_vllm_adapter_config_loads_client_settings(self):
        config = VLLMNaviSchedAdapterConfig.from_env(
            {
                "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": "model",
                "VLLM_NAVI_SCHED_MASTER_ADDRESSES": (
                    "127.0.0.1:19000,127.0.0.2:19001"
                ),
                "VLLM_NAVI_SCHED_TIMEOUT_MS": "800",
            }
        )

        self.assertEqual(
            ("127.0.0.1:19000", "127.0.0.2:19001"),
            config.master_addresses,
        )
        self.assertEqual(800, config.timeout_ms)

    def test_vllm_client_routes_request(self):
        native_client = mock.Mock()
        native_client.schedule.return_value = object()
        config = VLLMNaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="model",
            master_addresses=("127.0.0.1:19000",),
        )
        client = VLLMNaviSchedClient(
            config,
            native_client=native_client,
        )

        result = client.route([1, 2], request_id=1)

        self.assertIs(native_client.schedule.return_value, result)
        native_client.schedule.assert_called_once_with(
            2,
            block_hashes=[],
            block_size=16,
            forward_payload=b"",
            target_biz_name="model",
            timeout_ms=1000,
            request_id=1,
        )

    def test_vllm_client_accepts_role_name(self):
        config = VLLMNaviSchedAdapterConfig(
            master_biz_name="master",
            target_biz_name="model",
            master_addresses=("127.0.0.1:19000",),
        )
        client_module = importlib.import_module(
            "navi_sched.adapters.vllm.NaviSchedClient"
        )

        with mock.patch.object(
            client_module, "CoreNaviSchedClient"
        ) as client_class:
            client = VLLMNaviSchedClient(
                config, name="dashserving_client"
            )

            self.assertEqual(
                "dashserving_client",
                client_class.call_args.kwargs["name"],
            )
            client.close()



if __name__ == "__main__":
    unittest.main()
