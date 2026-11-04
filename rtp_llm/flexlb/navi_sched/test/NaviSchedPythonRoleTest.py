import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import json
from unittest import mock
from urllib.request import Request, urlopen
import navi_sched
from navi_sched import (
    NaviSchedAgent,
    NaviSchedClient,
    NaviSchedMaster,
    NaviSchedRequest,
)
from navi_sched.alog import NAVI_SCHED_ALOG_CONFIG
from navi_sched.master_process import (
    MasterProcessConfig,
    load_master_config,
    make_master_kmon_tags,
    make_runtime_config,
    make_scheduler_config,
    run_master,
)
from navi_sched.adapters.rtp_llm.RtpLLMMaster import (
    make_rtp_llm_master_options,
)
from navi_sched.adapters.rtp_llm.RtpLLMAdapterUtils import (
    make_kmon_options,
)
from navi_sched.adapters.rtp_llm.ServiceDiscovery import (
    RtpLLMServiceRoute,
)
from navi_sched.route_control import (
    RouteAddressSubscriber,
)
from navi_sched.util import DEFAULT_MASTER_SNAPSHOT_KEEP_COUNT


class NaviSchedPythonRoleTest(unittest.TestCase):
    @mock.patch("navi_sched.master.RoleFacade.__init__", return_value=None)
    @mock.patch("navi_sched.master.makeRuntimeConfig")
    @mock.patch("navi_sched.master.makeServiceConfig")
    @mock.patch("navi_sched.master.naviSchedModule")
    def test_master_collector_queue_options_forwarded_to_native_config(
        self, native_module, make_service_config, make_runtime_config, role_init
    ):
        config = make_runtime_config.return_value
        NaviSchedMaster("master_biz")
        self.assertEqual(0, config.collector_thread_num)
        self.assertEqual(0, config.collector_queue_count)
        NaviSchedMaster("master_biz", collector_thread_num=64, collector_queue_count=4)
        self.assertEqual(64, config.collector_thread_num)
        self.assertEqual(4, config.collector_queue_count)
        options = make_service_config.call_args.args[2]
        self.assertNotIn("collector_thread_num", options)
        self.assertNotIn("collector_queue_count", options)

    @mock.patch("navi_sched.client.RoleFacade.__init__", return_value=None)
    @mock.patch("navi_sched.client.makeRuntimeConfig")
    @mock.patch("navi_sched.client.makeServiceConfig")
    @mock.patch("navi_sched.client.naviSchedModule")
    def test_client_sample_collection_config_defaults_on(
        self, native_module, make_service_config, make_runtime_config, role_init
    ):
        config = make_runtime_config.return_value
        NaviSchedClient("client_biz")
        self.assertIs(True, config.enable_sample_collection)
        NaviSchedClient("client_biz", enable_sample_collection=False)
        self.assertIs(False, config.enable_sample_collection)
        self.assertNotIn("enable_sample_collection", make_service_config.call_args.args[2])

    @mock.patch("navi_sched.master.RoleFacade.__init__", return_value=None)
    @mock.patch("navi_sched.master.makeRuntimeConfig")
    @mock.patch("navi_sched.master.makeServiceConfig")
    @mock.patch("navi_sched.master.naviSchedModule")
    def test_master_snapshot_keep_count_default_and_override(
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

    def test_vllm_bridge_capacity_split_argument_contract(self):
        agent = NaviSchedAgent(
            "capacity_split_bridge_agent",
            total_blocks=1,
            block_size=16,
            linear_step=1,
            hybrid_tail_required=False,
            pd_separated=False,
            latency_prediction_mode="unified",
            enable_kmon=False,
            wait_no_query=False,
            gig={"grpc_port": 0},
        )
        try:
            agent.init()
            bridge = navi_sched._navi_sched_core._VLLMAgentBridge(agent)
            workload = {
                "event_time_us": 123456,
                "batch_count": 2,
                "request_count": 4,
                "remaining_prefill_request_count": 3,
                "running_request_count": 2,
                "running_elapsed_time_ms": 12.5,
                "running_decode_request_count": 1,
                "running_decode_generated_token_count": 7,
                "running_decode_remaining_token_budget": 13,
                "waiting_request_count": 2,
                "remaining_prefill_context_token_count": 24,
                "remaining_prefill_token_count": 36,
                "remaining_prefill_token_square_sum": 650.0,
                "context_remaining_prefill_token_product_sum": 434.0,
                "max_running_request_count": 3,
                "running_capacity_slot_count": 3,
                "running_capacity_prefill_request_count": 2,
                "running_capacity_prefill_context_token_count": 11,
                "running_capacity_prefill_token_count": 17,
                "running_capacity_prefill_token_square_sum": 289.0,
                "running_capacity_context_remaining_prefill_token_product_sum": 187.0,
                "waiting_capacity_prefill_request_count": 1,
                "waiting_capacity_prefill_context_token_count": 13,
                "waiting_capacity_prefill_token_count": 19,
                "waiting_capacity_prefill_token_square_sum": 361.0,
                "waiting_capacity_context_remaining_prefill_token_product_sum": 247.0,
                "remaining_total_context_token_count": 40,
                "remaining_total_token_count": 50,
                "remaining_total_token_square_sum": 2500.0,
                "context_remaining_total_token_product_sum": 2000.0,
                "total_prompt_token_count": 60,
                "in_flight_scheduled_token_count": 7,
                "admitted_schedule_batch_id": 0,
                "admitted_local_batch_id": 0,
                "admitted_request_count": 0,
                "admitted_pending_cache_tracked": False,
                "admitted_request_ids": [],
            }
            self.assertTrue(bridge.report_workload(**workload))

            invalid = dict(workload)
            invalid["waiting_capacity_prefill_request_count"] = 2
            self.assertFalse(bridge.report_workload(**invalid))

            invalid = dict(workload)
            invalid["running_decode_request_count"] = 3
            self.assertFalse(bridge.report_workload(**invalid))

            invalid = dict(workload)
            invalid["running_decode_remaining_token_budget"] = -1
            self.assertFalse(bridge.report_workload(**invalid))

            self.assertFalse(
                bridge.finish_batch_with_execution(
                    reporter=None,
                    values=[(1, 60, 24)],
                    wall_time_ms=5.0,
                    execute_model_time_ms=4.0,
                    execution_step_count=2,
                    prefill_token_count=36,
                    decode_request_step_count=1,
                    decode_token_count=1,
                    unified_latency_ms=7.0,
                    preceding_request_count=5,
                    sample_batch_count=2,
                    running_request_count=2,
                    running_cache_hit_token_count=11,
                    running_cache_miss_token_count=17,
                    running_cache_miss_token_square_sum=289.0,
                    running_cache_hit_miss_token_product_sum=187.0,
                    waiting_request_count=1,
                    waiting_cache_hit_token_count=13,
                    waiting_cache_miss_token_count=19,
                    waiting_cache_miss_token_square_sum=361.0,
                    waiting_cache_hit_miss_token_product_sum=247.0,
                    running_elapsed_time_ms=12.5,
                    running_decode_request_count=1,
                    running_decode_generated_token_count=7,
                    running_decode_remaining_token_budget=13,
                    admission_to_first_execution_ms=3.5,
                    native_preparation_ms=1.0,
                )
            )
        finally:
            agent.stop()

    def test_agent_latency_prediction_mode_validation(self):
        options = {
            "total_blocks": 1,
            "block_size": 16,
            "linear_step": 1,
            "hybrid_tail_required": False,
        }
        with self.assertRaisesRegex(ValueError, "must be 'split' or 'unified'"):
            NaviSchedAgent(
                "agent", latency_prediction_mode="invalid", **options
            )
        with self.assertRaisesRegex(
            ValueError,
            "pd_separated=False requires latency_prediction_mode='unified'",
        ):
            NaviSchedAgent(
                "agent",
                pd_separated=False,
                latency_prediction_mode="split",
                **options,
            )

    def test_master_config_json_takes_precedence_over_file(self):
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as stream:
            json.dump({"scheduler_name": "file"}, stream)
            stream.flush()
            config = MasterProcessConfig.from_env({
                "NAVI_SCHED_MASTER_BIZ_NAME": "master",
                "NAVI_SCHED_MASTER_CONFIG_FILE": stream.name,
                "NAVI_SCHED_MASTER_CONFIG_JSON": json.dumps({
                    "scheduler_name": "json",
                }),
            })
            config.validate()
            self.assertEqual("master_snapshot", config.name)
            self.assertEqual(".", config.snapshot_dir)
            self.assertEqual(
                {"scheduler_name": "json"},
                load_master_config(config.config_file, config.config_json),
            )

    def test_master_config_json_rejects_non_object(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            load_master_config(config_json="[]")

    def test_standalone_master_scheduler_config_is_an_object(self):
        options = {
            "scheduler_config": {
                "latency_weight_concavity": 0.0,
                "latency_token_normalization_power": 0.0,
            },
        }
        self.assertEqual(
            '{"latency_token_normalization_power":0.0,'
            '"latency_weight_concavity":0.0}',
            make_scheduler_config(options),
        )
        self.assertEqual({}, options)

        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            make_scheduler_config({"scheduler_config": "{}"})

    def test_standalone_master_starts_with_nonempty_scheduler_config(self):
        script = """
import navi_sched.master_process as master_process

master_process.wait_for_stop = lambda: None
master_process.run_master(
    master_process.MasterProcessConfig(
        master_biz_name="scheduler_config_object_test",
        config_file="",
        config_json="{}",
        enable_kmon=False,
    ),
    {
        "scheduler_name": "cost",
        "scheduler_config": {
            "latency_weight_concavity": 0.0,
            "latency_token_normalization_power": 0.0,
        },
        "runtime_config": {"thread_num": 2},
        "enable_snapshot": False,
    },
)
"""
        with tempfile.TemporaryDirectory() as work_dir:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_role_alog_paths(self):
        self.assertIn(
            "alog.appender.NaviSchedAlogAppender.fileName="
            "./logs/navi_sched_alog.log",
            NAVI_SCHED_ALOG_CONFIG,
        )
        self.assertIn(
            "alog.logger.multi_call=INFO, GigAppender",
            NAVI_SCHED_ALOG_CONFIG,
        )
        self.assertIn(
            "alog.appender.GigAppender.fileName=./logs/gig.log",
            NAVI_SCHED_ALOG_CONFIG,
        )

    def test_default_role_runtime_artifact_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_dir = os.getcwd()
            roles = []
            os.chdir(temp_dir)
            try:
                options = {
                    "enable_kmon": False,
                    "enable_snapshot": True,
                    "snapshot_interval_ms": 60000,
                    "gig": {"grpc_port": 0},
                    "wait_no_query": False,
                }
                roles = [
                    NaviSchedClient(
                        "client_biz", "master_biz", **options
                    ),
                    NaviSchedMaster("master_biz", **options),
                    NaviSchedAgent(
                        "agent_biz",
                        total_blocks=1,
                        block_size=16,
                        linear_step=1,
                        hybrid_tail_required=False,
                        **options,
                    ),
                ]
                for role in roles:
                    role.init()
                snapshot_dirs = (
                    "navi_sched/sched_client",
                    "master_snapshot",
                    "navi_sched/sched_agent",
                )
                deadline = time.monotonic() + 2
                while not all(os.path.isdir(path) for path in snapshot_dirs):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
            finally:
                for role in reversed(roles):
                    role.stop()
                os.chdir(previous_dir)

            artifacts = set()
            for root, dirs, files in os.walk(temp_dir):
                for path in dirs + files:
                    artifacts.add(os.path.relpath(
                        os.path.join(root, path), temp_dir
                    ))
            expected = {
                "logs/navi_sched_client_default.log",
                "logs/navi_master_snapshot_default.log",
                "logs/navi_sched_agent_default.log",
                "navi.sched_client.snapshot.vis",
                "navi.master_snapshot.snapshot.vis",
                "navi.sched_agent.snapshot.vis",
                "navi.sched_client.snapshot_summary",
                "navi.master_snapshot.snapshot_summary",
                "navi.sched_agent.snapshot_summary",
                "sched_client.navi_history",
                "master_snapshot.navi_history",
                "sched_agent.navi_history",
                "navi_sched/sched_client",
                "master_snapshot",
                "navi_sched/sched_agent",
            }
            self.assertTrue(expected.issubset(artifacts), expected - artifacts)
            for artifact in artifacts:
                self.assertNotIn("navi_navi_sched", artifact)
                self.assertNotIn(str(os.getpid()), artifact)

    def test_standalone_master_adds_dashboard_kmon_tags(self):
        options = {"kmon_tags": {"custom": "value"}}
        self.assertEqual(
            {
                "custom": "value",
                "role": "master",
                "deployment_name": "cn_MI308X_2TP_spot",
                "hippo_app": "whale_prod_Qwen3",
                "hippo_role": "Qwen3_cn_MI308X.navi_sched_master",
                "hippo_group": "Qwen3_cn_MI308X_2TP_spot",
                "slot_id": "330485",
                "host": "215.77.99.80",
                "host_ip": "215.77.99.80",
                "container_ip": "215.77.99.80",
                "physical_host_ip": "215.77.18.161",
            },
            make_master_kmon_tags(
                options,
                {
                    "DEPLOYMENT_NAME": "cn_MI308X_2TP_spot",
                    "HIPPO_APP": "whale_prod_Qwen3",
                    "HIPPO_ROLE": "Qwen3_cn_MI308X.navi_sched_master",
                    "C2_GROUP": "Qwen3_cn_MI308X_2TP_spot",
                    "HIPPO_SLOT_ID": "330485",
                    "HIPPO_SLAVE_IP": "215.77.18.161",
                    "POD_IP": "215.77.99.80",
                },
            ),
        )
        self.assertEqual({}, options)

    def test_standalone_master_uses_hippo_role_as_deployment_fallback(self):
        self.assertEqual(
            "cn_MI308X_2TP_spot",
            make_master_kmon_tags(
                {}, {"HIPPO_ROLE": "cn_MI308X_2TP_spot"}
            )["deployment_name"],
        )

    def test_rtp_master_subscribes_to_prefill_vip(self):
        environ = {
            "BIZ_NAME": "generation.model",
            "MODEL_SERVICE_CONFIG": json.dumps({
                "service_id": "generation.model",
                "master_endpoint": {
                    "address": "model.master.vip",
                    "protocol": "http",
                },
                "role_endpoints": [{
                    "group": "default",
                    "prefill_endpoint": {
                        "address": "model.prefill.vip",
                        "protocol": "http",
                    },
                }],
            }),
            "com.alibaba.vipserver.jmenv": "jmenv.test",
        }
        config = MasterProcessConfig(
            master_biz_name="",
            config_file="",
            config_json=json.dumps({"scheduler_name": "random"}),
        )
        config, options = make_rtp_llm_master_options(config, environ)
        self.assertEqual("rtp_llm_navi_sched_master", config.master_biz_name)
        self.assertEqual(7003, config.grpc_port)
        self.assertEqual({
            "jmenv_domain": "jmenv.test",
            "services": ["model.prefill.vip"],
            "grpc_port_diff": 3,
        }, options["runtime_config"]["vip_subscription"])

    def test_rtp_master_does_not_require_its_own_endpoint(self):
        environ = {
            "MODEL_SERVICE_CONFIG": json.dumps({
                "role_endpoints": [{
                    "prefill_endpoint": {
                        "address": "model.prefill.group",
                        "protocol": "http",
                    },
                }],
            }),
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json="{}"
        )
        config, _ = make_rtp_llm_master_options(config, environ)
        self.assertEqual("rtp_llm_navi_sched_master", config.master_biz_name)
        self.assertEqual(7001, config.http_port)
        self.assertEqual(7002, config.arpc_port)
        self.assertEqual(7003, config.grpc_port)

    def test_rtp_master_keeps_default_biz_when_endpoint_cannot_be_derived(self):
        environ = {
            "MODEL_SERVICE_CONFIG": json.dumps({
                "role_endpoints": [{
                    "prefill_endpoint": {
                        "address": "custom-service-name",
                        "protocol": "http",
                    },
                }],
            }),
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json="{}"
        )
        config, _ = make_rtp_llm_master_options(config, environ)
        self.assertEqual("rtp_llm_navi_sched_master", config.master_biz_name)

    def test_rtp_master_matches_flexlb_port_layout(self):
        environ = {
            "START_PORT": "7101",
            "MODEL_SERVICE_CONFIG": json.dumps({
                "role_endpoints": [{
                    "prefill_endpoint": {
                        "address": "model.prefill.group",
                        "protocol": "http",
                    },
                }],
            }),
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json="{}"
        )
        config, _ = make_rtp_llm_master_options(config, environ)
        self.assertEqual(7101, config.http_port)
        self.assertEqual(7102, config.arpc_port)
        self.assertEqual(7103, config.grpc_port)

    def test_rtp_master_uses_unified_navi_sched_config(self):
        environ = {
            "MODEL_SERVICE_CONFIG": json.dumps({
                "master_endpoint": {
                    "address": "model.master.vip",
                    "protocol": "http",
                },
                "role_endpoints": [{
                    "prefill_endpoint": {
                        "address": "model.prefill.vip",
                        "protocol": "http",
                    },
                }],
            }),
            "RTP_LLM_NAVI_SCHED_CONFIG_JSON": json.dumps({
                "log_path": "/home/admin/logs",
                "master": {
                    "grpc_port": 7013,
                    "enable_kmon": False,
                    "scheduler_name": "cost",
                    "scheduler_config": {
                        "lambda": 0.0,
                        "alpha": 512.0,
                    },
                    "max_wait_schedule_time_ms": 30,
                },
            }),
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json=""
        )
        config, options = make_rtp_llm_master_options(config, environ)
        self.assertEqual(7013, config.grpc_port)
        self.assertTrue(config.enable_kmon)
        self.assertEqual("/home/admin/logs/navi_sched.log", config.log_file_name)
        self.assertEqual("navi_sched", config.snapshot_dir)
        self.assertEqual("cost", options["scheduler_name"])
        self.assertEqual(
            '{"alpha":512.0,"lambda":0.0}', options["scheduler_config"]
        )
        self.assertEqual(30, options["max_wait_schedule_time_ms"])

    def test_rtp_master_uses_cost_defaults_without_unified_config(self):
        environ = {
            "MODEL_SERVICE_CONFIG": json.dumps({
                "master_endpoint": {
                    "address": "model.master.vip",
                    "protocol": "http",
                },
                "role_endpoints": [{
                    "prefill_endpoint": {
                        "address": "model.prefill.vip",
                        "protocol": "http",
                    },
                }],
            }),
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json=""
        )
        config, options = make_rtp_llm_master_options(config, environ)
        config.validate()
        self.assertEqual(7003, config.grpc_port)
        self.assertTrue(config.enable_kmon)
        self.assertEqual("cost", options["scheduler_name"])
        self.assertEqual(30, options["max_wait_schedule_count"])
        self.assertEqual(30, options["max_wait_schedule_time_ms"])
        self.assertEqual(64, options["runtime_config"]["thread_num"])
        self.assertEqual(
            {
                "alpha": 512.0,
                "alpha_decay": 1.0,
                "lambda": 0.0,
                "max_loop_count": 10,
                "min_alpha": 0.0,
                "parallelism": 0,
            },
            json.loads(options["scheduler_config"]),
        )

    def test_rtp_roles_always_use_shared_kmon_service_name(self):
        environ = {
            "kmonitorServiceName": "deepseek_v4_flash",
            "DEPLOYMENT_NAME": "na130_navi_sched",
        }
        for role in ("master", "client"):
            self.assertEqual({
                "enable_kmon": True,
                "kmon_service_name": "deepseek_v4_flash",
                "kmon_metrics_path": "",
                "kmon_tags": {
                    "role": role,
                    "deployment_name": "na130_navi_sched",
                },
            }, make_kmon_options(role, environ))
        self.assertEqual(
            "navi_sched",
            make_kmon_options("client", {})["kmon_service_name"],
        )
        self.assertEqual(
            {"role": "client"},
            make_kmon_options("client", {})["kmon_tags"],
        )

    def test_unified_navi_sched_config_rejects_invalid_section(self):
        environ = {
            "MODEL_SERVICE_CONFIG": "{}",
            "RTP_LLM_NAVI_SCHED_CONFIG_JSON": '{"master":[]}',
        }
        config = MasterProcessConfig(
            master_biz_name="", config_file="", config_json=""
        )
        with self.assertRaisesRegex(ValueError, "master must be a JSON object"):
            make_rtp_llm_master_options(config, environ)

    def test_rtp_service_route_rejects_ambiguous_prefill_domains(self):
        environ = {"MODEL_SERVICE_CONFIG": json.dumps({
            "role_endpoints": [
                {"prefill_endpoint": {"address": "prefill-a"}},
                {"prefill_endpoint": {"address": "prefill-b"}},
            ],
        })}
        with self.assertRaisesRegex(ValueError, "same address and protocol"):
            RtpLLMServiceRoute.from_env(environ)

    def test_route_address_control_pushes_updates_to_subscriber(self):
        grpc_port = self._reserve_adjacent_ports()
        master = NaviSchedMaster(
            "route_master_biz",
            name="route_master",
            runtime_config=self._make_runtime_config(thread_num=4),
            gig={"grpc_port": grpc_port},
            enable_kmon=False,
            enable_snapshot=False,
            wait_no_query=False,
        ).init()
        subscriber = RouteAddressSubscriber(
            f"127.0.0.1:{grpc_port}", reconnect_delay_s=0.01
        )
        try:
            self.assertEqual(grpc_port, master.grpc_port)
            self.assertEqual(grpc_port + 1, master.http_port)
            self.assertTrue(subscriber.wait_for(""))
            request = Request(
                f"http://127.0.0.1:{master.http_port}/route-mode",
                method="POST",
                data=b'{"update":true,"masterAddress":"127.0.0.1:17001"}',
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertTrue(payload["accepted"])
            self.assertEqual("127.0.0.1:17001", payload["masterAddress"])
            self.assertTrue(subscriber.wait_for("127.0.0.1:17001"))

            request = Request(
                f"http://127.0.0.1:{master.http_port}/route-mode",
                method="POST",
                data=b'{"update":true,"masterAddress":"invalid"}',
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=2) as response:
                rejected = json.load(response)
            self.assertFalse(rejected.get("accepted", False))
            self.assertEqual("127.0.0.1:17001", rejected["masterAddress"])
            self.assertEqual("127.0.0.1:17001", subscriber.master_address)

            request = Request(
                f"http://127.0.0.1:{master.http_port}/route-mode",
                method="POST",
                data=b'{"update":true,"masterAddress":""}',
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertTrue(payload["accepted"])
            self.assertEqual("", payload.get("masterAddress", ""))
            self.assertTrue(subscriber.wait_for(""))
            self.assertGreaterEqual(subscriber.generation, 3)
        finally:
            subscriber.stop()
            master.stop()

    def test_route_address_subscriber_reports_observed_generation(self):
        grpc_port = self._reserve_adjacent_ports()
        master = NaviSchedMaster(
            "route_state_master_biz",
            name="route_state_master",
            runtime_config=self._make_runtime_config(thread_num=4),
            gig={"grpc_port": grpc_port},
            enable_kmon=False,
            enable_snapshot=False,
            wait_no_query=False,
        ).init()
        with tempfile.TemporaryDirectory() as state_dir:
            previous = os.environ.get(
                "RTP_LLM_NAVI_SCHED_ROUTE_STATE_DIR"
            )
            os.environ["RTP_LLM_NAVI_SCHED_ROUTE_STATE_DIR"] = state_dir
            subscriber = RouteAddressSubscriber(
                f"127.0.0.1:{grpc_port}", reconnect_delay_s=0.01
            )
            try:
                self.assertTrue(subscriber.wait_for(""))
                state_file = os.path.join(
                    state_dir, "route_state.json"
                )
                deadline = time.monotonic() + 2
                while not os.path.exists(state_file):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                with open(state_file, encoding="utf-8") as file:
                    state = json.load(file)
                self.assertEqual(os.getpid(), state["pid"])
                self.assertEqual("", state["masterAddress"])
                self.assertEqual(subscriber.generation, state["generation"])
            finally:
                subscriber.stop()
                master.stop()
                if previous is None:
                    os.environ.pop(
                        "RTP_LLM_NAVI_SCHED_ROUTE_STATE_DIR", None
                    )
                else:
                    os.environ[
                        "RTP_LLM_NAVI_SCHED_ROUTE_STATE_DIR"
                    ] = previous

    def test_scheduler_config_http_update_and_query(self):
        grpc_port = self._reserve_adjacent_ports()
        agent = self._create_agents(1)[0]
        master = NaviSchedMaster(
            "config_master_biz",
            name="config_master",
            runtime_config=self._make_runtime_config(
                thread_num=4,
                local_nodes=[("agent_biz", agent.grpc_port, True)],
            ),
            gig={"grpc_port": grpc_port},
            scheduler_name="cost",
            scheduler_config=json.dumps({"lambda": 0.5, "alpha": 512.0}),
            max_wait_schedule_count=30,
            max_wait_schedule_time_ms=30,
            enable_kmon=False,
            enable_snapshot=False,
            wait_no_query=False,
        ).init()
        url = f"http://127.0.0.1:{master.http_port}/scheduler-config"
        try:
            initial = self._post_json(url, {})
            self.assertTrue(initial["accepted"])
            self.assertGreater(initial["generation"], 0)

            updated = self._post_json(url, {
                "update": True,
                "configJson": json.dumps({
                    "lambda": 0.0,
                    "max_wait_schedule_count": 16,
                    "max_wait_schedule_time_ms": 20,
                }),
            })
            self.assertTrue(updated["accepted"])
            self.assertEqual(initial["generation"] + 1, updated["generation"])
            config = json.loads(updated["configJson"])
            self.assertEqual(0.0, config["lambda"])
            self.assertEqual(16, config["max_wait_schedule_count"])
            self.assertEqual(20, config["max_wait_schedule_time_ms"])
            self.assertEqual(16000, config["optimization_time_budget_us"])

            queried = self._post_json(url, {})
            self.assertTrue(queried["accepted"])
            self.assertEqual(updated["generation"], queried["generation"])
            self.assertEqual(updated["configJson"], queried["configJson"])

            rejected = self._post_json(url, {
                "update": True,
                "configJson": '{"unknown":1}',
            })
            self.assertFalse(rejected.get("accepted", False))
            self.assertEqual(updated["generation"], rejected["generation"])
            self.assertIn("unknown", rejected["error"])

            with NaviSchedClient(
                "config_client_biz",
                name="config_client",
                runtime_config=self._make_runtime_config(
                    thread_num=2,
                    local_nodes=[
                        ("config_master_biz", master.grpc_port, False)
                    ],
                ),
                enable_agent_forward=False,
                enable_kmon=False,
                enable_snapshot=False,
                wait_no_query=False,
            ) as client:
                client.init()
                decision = client.schedule(
                    NaviSchedRequest(
                        request_id=1001,
                        prompt_token_count=256,
                        block_hashes=[1],
                    ),
                    "agent_biz",
                    master_biz_name="config_master_biz",
                    timeout_ms=1000,
                    collect_replay=True,
                )
                replay = json.loads(decision.replay.to_json())
                self.assertEqual(
                    "cold_start_inflight_selected", decision.reason
                )
                scheduler_info = replay["scheduler_info"]
                self.assertEqual(1, scheduler_info["audit_version"])
                self.assertTrue(scheduler_info["cold_start_evaluated"])
                self.assertEqual(1, len(scheduler_info["replay_node_runtime"]))
        finally:
            master.stop()
            agent.stop()

    def test_replay_control_embedded_cluster(self):
        replay_snapshot = tempfile.TemporaryDirectory()
        self.addCleanup(replay_snapshot.cleanup)
        agent = self._create_agents(1)[0]
        master_port = self._reserve_adjacent_ports()
        options = {
            "enable_kmon": False,
            "enable_snapshot": False,
            "snapshot_dir": replay_snapshot.name,
            "wait_no_query": False,
        }
        master = NaviSchedMaster(
            "replay_master_biz",
            name="replay_master",
            runtime_config=self._make_runtime_config(
                thread_num=8,
                local_nodes=[("agent_biz", agent.grpc_port, True)],
            ),
            gig={"grpc_port": master_port},
            **options,
        ).init()
        clients = [
            NaviSchedClient(
                f"replay_client_{suffix}",
                "replay_master_biz",
                name=f"replay_client_{suffix}",
                runtime_config=self._make_runtime_config(
                    thread_num=4,
                    local_nodes=[
                        ("replay_master_biz", master.grpc_port, False)
                    ],
                ),
                enable_agent_forward=False,
                **options,
            ).init()
            for suffix in ("a", "b")
        ]
        url = f"http://127.0.0.1:{master.http_port}/replay-control"

        def query_clients(expected_count, timeout=5.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                payload = self._post_json(url, {"action": "query"})
                if len(payload.get("clients", [])) == expected_count:
                    return payload
                time.sleep(0.01)
            self.fail(f"Replay clients did not reach {expected_count}: {payload}")

        def schedule(client, request_id):
            return self._wait_for_decision(
                client,
                NaviSchedRequest(
                    request_id=request_id,
                    prompt_token_count=256,
                    block_hashes=[],
                ),
                "replay_master_biz",
            )

        try:
            initial = query_clients(2)
            self.assertTrue(initial["accepted"])
            self.assertEqual("stop", initial["action"])
            self.assertEqual(0, initial["eligibleClientCount"])
            self.assertEqual(0.0, initial["perClientQps"])
            for item in initial["clients"]:
                self.assertEqual(0, item["earliestEventTimeUs"])
                self.assertEqual(0, item["latestEventTimeUs"])
                self.assertEqual(0, item["sampleCount"])
                self.assertFalse(item["replayable"])
                self.assertEqual(0, item["selectionGeneration"])
                self.assertEqual(0, item["selectedSampleCount"])

            rejected = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "totalQps": 10.0,
                "loopCount": 1,
            })
            self.assertIn("accepted", rejected)
            self.assertFalse(rejected["accepted"])
            self.assertIn("start_time_us", rejected["error"])

            schedule(clients[0], 1001)
            schedule(clients[1], 1002)
            time.sleep(0.01)
            schedule(clients[0], 1003)
            schedule(clients[1], 1004)
            populated = query_clients(2)
            self.assertTrue(all(
                item["replayable"] and item["sampleCount"] >= 2
                for item in populated["clients"]
            ))
            start_time_us = max(
                item["earliestEventTimeUs"] for item in populated["clients"]
            )
            end_time_us = min(
                item["latestEventTimeUs"] for item in populated["clients"]
            )
            self.assertGreater(end_time_us, start_time_us)
            before_counts = {
                item["clientId"]: item["sampleCount"]
                for item in populated["clients"]
            }

            recent_fixed = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "historyMode": "recent_count",
                "recentQueryCount": 2,
                "durationMs": 1000,
                "totalQps": 20.0,
                "loopCount": 1,
            })
            self.assertTrue(recent_fixed.get("accepted"), recent_fixed)
            self.assertEqual(2, recent_fixed["eligibleClientCount"])
            self.assertAlmostEqual(10.0, recent_fixed["perClientQps"])
            self.assertTrue(self._post_json(url, {"action": "stop"})["accepted"])

            recent_realtime = self._post_json(url, {
                "action": "start",
                "mode": "realtime",
                "historyMode": "recent_count",
                "recentQueryCount": 2,
                "totalQps": 0.0,
                "loopCount": 1,
            })
            self.assertTrue(recent_realtime.get("accepted"), recent_realtime)
            self.assertEqual(2, recent_realtime["eligibleClientCount"])
            self.assertEqual(0.0, recent_realtime.get("perClientQps", 0.0))
            self.assertTrue(self._post_json(url, {"action": "stop"})["accepted"])

            fixed = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "startTimeUs": start_time_us,
                "endTimeUs": end_time_us,
                "durationMs": 1000,
                "totalQps": 10.0,
                "loopCount": 0,
            })
            self.assertTrue(fixed.get("accepted"), fixed)
            self.assertEqual("start", fixed["action"])
            self.assertEqual(2, fixed["eligibleClientCount"])
            self.assertAlmostEqual(5.0, fixed["perClientQps"])

            for total_qps, per_client_qps in (
                (20.0, 10.0),
                (6.0, 3.0),
                (40.0, 20.0),
                (4.0, 2.0),
            ):
                updated = self._post_json(url, {
                    "action": "update",
                    "totalQps": total_qps,
                })
                self.assertTrue(updated.get("accepted"), updated)
                self.assertAlmostEqual(
                    per_client_qps, updated["perClientQps"]
                )

            replay_deadline = time.monotonic() + 2.0
            while any(client.request_received_count <= 2 for client in clients):
                self.assertLess(
                    time.monotonic(), replay_deadline,
                    "fixed_qps command did not produce requests on every client",
                )
                time.sleep(0.01)

            schedule(clients[0], 1005)
            time.sleep(0.1)
            concurrent = query_clients(2)
            after_counts = {
                item["clientId"]: item["sampleCount"]
                for item in concurrent["clients"]
            }
            changed = [
                client_id for client_id, count in before_counts.items()
                if after_counts[client_id] == count + 1
            ]
            unchanged = [
                client_id for client_id, count in before_counts.items()
                if after_counts[client_id] == count
            ]
            self.assertEqual(1, len(changed), after_counts)
            self.assertEqual(1, len(unchanged), after_counts)

            stopped = self._post_json(url, {"action": "stop"})
            self.assertTrue(stopped["accepted"])
            self.assertEqual("stop", stopped["action"])
            self.assertEqual(0, stopped["eligibleClientCount"])
            self.assertEqual(0.0, stopped["perClientQps"])
            self.assertTrue(all(
                item["selectionGeneration"] == 0
                and item["selectedSampleCount"] == 0
                for item in stopped["clients"]
            ))

            realtime = self._post_json(url, {
                "action": "start",
                "mode": "realtime",
                "startTimeUs": start_time_us,
                "endTimeUs": end_time_us,
                "totalQps": 0.0,
                "loopCount": 2,
            })
            self.assertTrue(realtime["accepted"], realtime)
            self.assertEqual(2, realtime["eligibleClientCount"])
            self.assertEqual(0.0, realtime.get("perClientQps", 0.0))
            self.assertTrue(self._post_json(url, {"action": "stop"})["accepted"])

            fixed = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "historyMode": "recent_count",
                "recentQueryCount": 2,
                "durationMs": 10000,
                "totalQps": 10.0,
                "loopCount": 0,
            })
            self.assertEqual(2, fixed["eligibleClientCount"])
            stopped_client = clients.pop()
            stopped_client.stop()
            reallocated = query_clients(1)
            self.assertEqual(1, reallocated["eligibleClientCount"])
            self.assertAlmostEqual(10.0, reallocated["perClientQps"])

            restarted_client = NaviSchedClient(
                "replay_client_b",
                "replay_master_biz",
                name="replay_client_b",
                runtime_config=self._make_runtime_config(
                    thread_num=4,
                    local_nodes=[
                        ("replay_master_biz", master.grpc_port, False)
                    ],
                ),
                enable_agent_forward=False,
                **options,
            ).init()
            clients.append(restarted_client)
            rejoin_deadline = time.monotonic() + 5.0
            while True:
                rejoined = query_clients(2)
                restored = next(
                    item for item in rejoined["clients"]
                    if item["clientId"].startswith("replay_client_b@")
                )
                if (
                    rejoined["eligibleClientCount"] == 2
                    and restored["selectionGeneration"] > 0
                    and restored["selectedSampleCount"] == 2
                ):
                    break
                self.assertLess(
                    time.monotonic(), rejoin_deadline,
                    f"Restarted replay client did not rejoin: {rejoined}",
                )
                time.sleep(0.01)
            self.assertGreaterEqual(restored["sampleCount"], 2)
            self.assertTrue(restored["replayable"])
            self.assertAlmostEqual(5.0, rejoined["perClientQps"])
            self.assertTrue(self._post_json(url, {"action": "stop"})["accepted"])

            master.stop()
            for client in reversed(clients):
                stop_started = time.monotonic()
                client.stop()
                self.assertLess(time.monotonic() - stop_started, 2.0)
        finally:
            for client in reversed(clients):
                client.stop()
            master.stop()
            agent.stop()

    def test_replay_time_range_excludes_client_with_empty_window(self):
        replay_snapshot = tempfile.TemporaryDirectory()
        self.addCleanup(replay_snapshot.cleanup)
        agent = self._create_agents(1)[0]
        master_port = self._reserve_adjacent_ports()
        options = {
            "enable_kmon": False,
            "enable_snapshot": False,
            "snapshot_dir": replay_snapshot.name,
            "wait_no_query": False,
        }
        master = NaviSchedMaster(
            "replay_range_master_biz",
            name="replay_range_master",
            runtime_config=self._make_runtime_config(
                thread_num=8,
                local_nodes=[("agent_biz", agent.grpc_port, True)],
            ),
            gig={"grpc_port": master_port},
            **options,
        ).init()
        clients = {
            suffix: NaviSchedClient(
                f"replay_range_client_{suffix}",
                "replay_range_master_biz",
                name=f"replay_range_client_{suffix}",
                runtime_config=self._make_runtime_config(
                    thread_num=4,
                    local_nodes=[
                        ("replay_range_master_biz", master.grpc_port, False)
                    ],
                ),
                enable_agent_forward=False,
                **options,
            ).init()
            for suffix in ("sparse", "inside")
        }
        url = f"http://127.0.0.1:{master.http_port}/replay-control"

        def query_clients(timeout=5.0):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                payload = self._post_json(url, {"action": "query"})
                if len(payload.get("clients", [])) == 2:
                    return payload
                time.sleep(0.01)
            self.fail(f"Replay clients did not reach 2: {payload}")

        def schedule(client, request_id):
            return self._wait_for_decision(
                client,
                NaviSchedRequest(
                    request_id=request_id,
                    prompt_token_count=256,
                    block_hashes=[],
                ),
                "replay_range_master_biz",
            )

        try:
            self.assertTrue(query_clients()["accepted"])
            schedule(clients["sparse"], 2001)
            time.sleep(0.01)
            schedule(clients["inside"], 2002)
            time.sleep(0.01)
            schedule(clients["inside"], 2003)
            time.sleep(0.01)
            schedule(clients["sparse"], 2004)

            populated = query_clients()
            by_name = {
                item["clientId"].split("@", 1)[0]: item
                for item in populated["clients"]
            }
            sparse = by_name["replay_range_client_sparse"]
            inside = by_name["replay_range_client_inside"]
            self.assertLess(
                sparse["earliestEventTimeUs"], inside["earliestEventTimeUs"]
            )
            self.assertGreater(
                sparse["latestEventTimeUs"], inside["latestEventTimeUs"]
            )

            rejected = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "startTimeUs": sparse["latestEventTimeUs"] + 1,
                "endTimeUs": sparse["latestEventTimeUs"] + 1000,
                "durationMs": 1000,
                "totalQps": 40.0,
                "loopCount": 1,
            })
            self.assertFalse(rejected.get("accepted"), rejected)
            self.assertEqual(
                "no replayable samples in requested selection", rejected["error"]
            )
            self.assertEqual("stop", rejected["action"])
            self.assertEqual(0, rejected["eligibleClientCount"])

            started = self._post_json(url, {
                "action": "start",
                "mode": "fixed_qps",
                "startTimeUs": inside["earliestEventTimeUs"],
                "endTimeUs": inside["latestEventTimeUs"],
                "durationMs": 1000,
                "totalQps": 40.0,
                "loopCount": 1,
            })
            self.assertTrue(started.get("accepted"), started)
            self.assertEqual(2, started["eligibleClientCount"])
            self.assertAlmostEqual(20.0, started["perClientQps"])

            deadline = time.monotonic() + 5.0
            while True:
                selected = query_clients()
                selected_by_name = {
                    item["clientId"].split("@", 1)[0]: item
                    for item in selected["clients"]
                }
                if (
                    selected["eligibleClientCount"] == 1
                    and selected_by_name["replay_range_client_sparse"][
                        "selectionGeneration"
                    ] > 0
                    and selected_by_name["replay_range_client_inside"][
                        "selectionGeneration"
                    ] > 0
                ):
                    break
                self.assertLess(
                    time.monotonic(), deadline,
                    f"Replay selection did not converge: {selected}",
                )
                time.sleep(0.01)

            self.assertAlmostEqual(40.0, selected["perClientQps"])
            self.assertEqual(
                0,
                selected_by_name["replay_range_client_sparse"].get(
                    "selectedSampleCount", 0
                ),
            )
            self.assertEqual(
                2,
                selected_by_name["replay_range_client_inside"][
                    "selectedSampleCount"
                ],
            )
            stopped = self._post_json(url, {"action": "stop"})
            self.assertTrue(stopped["accepted"])
            self.assertEqual("stop", stopped["action"])
        finally:
            for client in reversed(list(clients.values())):
                client.stop()
            master.stop()
            agent.stop()

    def test_master_custom_ports_and_health_alias(self):
        start_port = self._reserve_consecutive_ports(3)
        master = NaviSchedMaster(
            "health_master_biz",
            name="health_master",
            runtime_config=self._make_runtime_config(thread_num=2),
            gig={
                "http_port": start_port,
                "arpc_port": start_port + 1,
                "grpc_port": start_port + 2,
            },
            scheduler_name="cost",
            enable_kmon=False,
            enable_snapshot=False,
            wait_no_query=False,
        ).init()
        try:
            self.assertEqual(start_port, master.http_port)
            self.assertEqual(start_port + 2, master.grpc_port)
            with socket.create_connection(
                ("127.0.0.1", start_port + 1), timeout=2
            ):
                pass
            with urlopen(
                f"http://127.0.0.1:{start_port}/health", timeout=2
            ) as response:
                payload = json.load(response)
            self.assertTrue(payload["accepted"])
        finally:
            master.stop()

    @staticmethod
    def _reserve_adjacent_ports():
        for _ in range(100):
            with socket.socket() as first:
                first.bind(("127.0.0.1", 0))
                port = first.getsockname()[1]
                if port >= 65535:
                    continue
                with socket.socket() as second:
                    try:
                        second.bind(("127.0.0.1", port + 1))
                    except OSError:
                        continue
                    return port
        raise RuntimeError("failed to reserve adjacent ports")

    @staticmethod
    def _reserve_consecutive_ports(count):
        for _ in range(100):
            sockets = []
            try:
                first = socket.socket()
                sockets.append(first)
                first.bind(("127.0.0.1", 0))
                port = first.getsockname()[1]
                if port + count > 65535:
                    continue
                for offset in range(1, count):
                    current = socket.socket()
                    sockets.append(current)
                    current.bind(("127.0.0.1", port + offset))
                return port
            except OSError:
                continue
            finally:
                for current in sockets:
                    current.close()
        raise RuntimeError("failed to reserve consecutive ports")

    @staticmethod
    def _post_json(url, payload):
        request = Request(
            url,
            method="POST",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=2) as response:
            return json.load(response)

    @mock.patch("navi_sched.master_process.wait_for_stop")
    @mock.patch("navi_sched.master_process.NaviSchedMaster")
    def test_master_process_preserves_gig_io_threads(self, master_class, wait_for_stop):
        options = {"gig": {"grpc_io_thread_num": 10, "grpc_port": 9999}}
        run_master(MasterProcessConfig(
            master_biz_name="master", config_file="unused", grpc_port=7003,
        ), options)
        gig = master_class.call_args.kwargs["gig"]
        self.assertEqual(10, gig["grpc_io_thread_num"])
        self.assertEqual(7003, gig["grpc_port"])
        self.assertEqual({"grpc_io_thread_num": 10, "grpc_port": 9999}, options["gig"])
        master_class.return_value.__enter__.return_value.init.assert_called_once_with()
        wait_for_stop.assert_called_once_with()

    def test_master_process_splits_runtime_options(self):
        options = {
            "scheduler_name": "random",
            "max_wait_schedule_count": 100,
            "thread_num": 16,
            "local_nodes": [{"biz_name": "agent_biz"}],
        }
        runtime = make_runtime_config(options)
        self.assertEqual(16, runtime["thread_num"])
        self.assertEqual("agent_biz", runtime["local_nodes"][0]["biz_name"])
        self.assertEqual({
            "scheduler_name": "random",
            "max_wait_schedule_count": 100,
        }, options)

        options = {
            "scheduler_name": "random",
            "runtime_config": {"thread_num": 8},
        }
        self.assertEqual({"thread_num": 8}, make_runtime_config(options))
        self.assertEqual({"scheduler_name": "random"}, options)

    def test_master_process_uses_deployment_subscription_and_template_workers(self):
        template = {"runtime_config": {"thread_num": 128}, "scheduler_config": {"parallelism": 64}}
        for domain in ["cluster_a.publish", "cluster_b.publish"]:
            options = dict(template)
            subscription = {"services": [domain], "jmenv_domain": "discovery.example"}
            runtime = make_runtime_config(options, {
                "NAVI_SCHED_MASTER_VIP_SUBSCRIPTION_JSON": json.dumps(subscription),
            })
            self.assertEqual({"thread_num": 128, "vip_subscription": subscription}, runtime)
            self.assertEqual({"scheduler_config": {"parallelism": 64}}, options)
        self.assertEqual({"thread_num": 128}, template["runtime_config"])

    def test_master_process_rejects_invalid_deployment_subscription(self):
        for value in ["invalid", "[]", "null", "1", '"domain"']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "NAVI_SCHED_MASTER_VIP_SUBSCRIPTION_JSON"):
                make_runtime_config({"runtime_config": {"thread_num": 128}}, {
                    "NAVI_SCHED_MASTER_VIP_SUBSCRIPTION_JSON": value,
                })
        self.assertEqual({"thread_num": 128}, make_runtime_config(
            {"runtime_config": {"thread_num": 128}}, {},
        ))

    @mock.patch.dict(os.environ, {
        "NAVI_SCHED_MASTER_VIP_SUBSCRIPTION_JSON": '{"services":["cluster_b.publish"]}',
    })
    @mock.patch("navi_sched.master_process.wait_for_stop")
    @mock.patch("navi_sched.master_process.NaviSchedMaster")
    def test_master_process_applies_subscription_environment(self, master_class, wait_for_stop):
        options = {
            "runtime_config": {
                "thread_num": 128,
                "vip_subscription": {"services": ["cluster_a.publish"]},
            },
            "scheduler_config": {"parallelism": 64},
            "gig": {"grpc_io_thread_num": 30},
        }
        run_master(MasterProcessConfig(master_biz_name="master", config_file="unused"), options)
        args = master_class.call_args.kwargs
        self.assertEqual({
            "thread_num": 128, "vip_subscription": {"services": ["cluster_b.publish"]},
        }, args["runtime_config"])
        self.assertEqual({"parallelism": 64}, json.loads(args["scheduler_config"]))
        self.assertEqual(30, args["gig"]["grpc_io_thread_num"])
        self.assertEqual(["cluster_a.publish"], options["runtime_config"]["vip_subscription"]["services"])

    def test_navi_sched_python_module_boundary(self):
        core = navi_sched._navi_sched_core
        self.assertTrue(hasattr(core, "navi"))
        self.assertTrue(hasattr(core, "NaviSchedClient"))
        self.assertTrue(hasattr(core, "_configure_alog"))
        self.assertEqual({
            "NaviSchedAgent",
            "NaviSchedClient",
            "NaviSchedDecision",
            "NaviSchedMaster",
            "NaviSchedRequest",
            "NaviSchedTarget",
        }, set(navi_sched.__all__))

    def test_all_roles_configure_alog_files(self):
        role_expressions = (
            "NaviSchedMaster('master')",
            "NaviSchedClient('client', 'master')",
            "NaviSchedAgent('agent', total_blocks=1, block_size=16, "
            "linear_step=1, hybrid_tail_required=False)",
        )
        for expression in role_expressions:
            with self.subTest(expression=expression):
                with tempfile.TemporaryDirectory() as work_dir:
                    env = dict(os.environ)
                    env["PYTHONPATH"] = os.pathsep.join(sys.path)
                    result = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            "from navi_sched import NaviSchedAgent, "
                            "NaviSchedClient, NaviSchedMaster; " + expression,
                        ],
                        cwd=work_dir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertTrue(os.path.isfile(os.path.join(
                        work_dir, "logs", "navi_sched_alog.log")))
                    self.assertTrue(os.path.isfile(os.path.join(
                        work_dir, "logs", "gig.log")))

    def _assert_master_vip_sigterm_exit(self, enable_kmon):
        script = """
import signal
import threading
import navi_sched.master_process as master_process

def wait_for_stop():
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stopped.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.set())
    open("master.ready", "w").close()
    stopped.wait()

master_process.wait_for_stop = wait_for_stop
master_process.run_master(
    master_process.MasterProcessConfig(
        master_biz_name="navi_sched_exit_test_agent",
        config_file="",
        config_json="{}",
        enable_kmon=%s,
    ),
    {
        "scheduler_name": "random",
        "thread_num": 2,
        "vip_subscription": {
            "jmenv_domain": "jmenv.tbsite.net",
            "services": ["navi_sched.exit.test"],
            "grpc_port_diff": 2,
        },
        "enable_snapshot": False,
    },
)
""" % ("True" if enable_kmon else "False")
        with tempfile.TemporaryDirectory() as work_dir:
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(sys.path)
            process = subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            try:
                deadline = time.monotonic() + 15.0
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        stdout, stderr = process.communicate()
                        self.fail(
                            "Master exited before becoming healthy: "
                            f"stdout={stdout} stderr={stderr}"
                        )
                    if os.path.exists(os.path.join(work_dir, "master.ready")):
                        break
                    time.sleep(0.05)
                else:
                    self.fail("Master did not install its signal handlers")

                process.terminate()
                stdout, stderr = process.communicate(timeout=15)
                self.assertEqual(
                    0,
                    process.returncode,
                    f"stdout={stdout} stderr={stderr}",
                )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

    def test_master_vip_stops_before_process_loggers(self):
        for enable_kmon in (False, True):
            with self.subTest(enable_kmon=enable_kmon):
                self._assert_master_vip_sigterm_exit(enable_kmon)

    def test_python_wraps_cpp_roles(self):
        block_hash = (1 << 64) - 1
        agent_count = int(os.environ.get(
            "NAVI_SCHED_TEST_AGENT_COUNT", "10"))
        self.assertGreater(agent_count, 0)
        agents = self._create_agents(agent_count)
        self.assertEqual(agent_count, len(agents))
        self.assertIsInstance(
            agents[0], navi_sched._navi_sched_core.NaviSchedAgent)
        options = {
            "enable_kmon": False,
            "enable_snapshot": False,
            "wait_no_query": False,
        }
        try:
            with NaviSchedMaster(
                "master_biz",
                name="master",
                runtime_config=self._make_runtime_config(
                    thread_num=16,
                    local_nodes=[
                        ("agent_biz", agent.grpc_port, True)
                        for agent in agents
                    ],
                ),
                max_wait_schedule_count=20,
                max_wait_schedule_time_ms=10,
                **options,
            ) as master:
                master.init()
                with NaviSchedClient(
                    "client_biz",
                    name="client",
                    runtime_config=self._make_runtime_config(
                        thread_num=4,
                        local_nodes=[("master_biz", master.grpc_port, False)],
                    ),
                    enable_agent_forward=False,
                    **options,
                ) as client:
                    client.init()
                    request = NaviSchedRequest(
                        request_id=1,
                        prompt_token_count=256,
                        block_hashes=[block_hash],
                    )
                    with self.assertRaisesRegex(
                            ValueError, "master_biz_name is required"):
                        client.schedule(request, "agent_biz")
                    decision = self._wait_for_decision(
                        client, request, "master_biz")
                    self.assertIsNotNone(decision.target)
                    self.assertIsInstance(
                        decision.predicted_cache_hit_tokens, int
                    )
                    replay_decision = client.schedule(
                        NaviSchedRequest(
                            request_id=2,
                            prompt_token_count=256,
                            block_hashes=[block_hash],
                        ),
                        "agent_biz",
                        master_biz_name="master_biz",
                        timeout_ms=1000,
                        collect_replay=True,
                    )
                    replay = json.loads(replay_decision.replay.to_json())
                    self.assertEqual("agent_biz", replay["biz_name"])
                    replay_request_id = replay["requests"][0]["request"][
                        "request_id"
                    ]
                    self.assertGreater(replay_request_id, 0)
                    self.assertLess(replay_request_id, 1 << 64)
                    self.assertNotEqual(2, replay_request_id)
                    self.assertEqual(agent_count, len(replay["nodes"]))
                    agents[0].stop()
        finally:
            for agent in reversed(agents):
                agent.stop()

    def test_master_reconnects_restarted_agent_at_same_address(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            grpc_port = probe.getsockname()[1]
        options = {
            "enable_kmon": False,
            "enable_snapshot": False,
            "wait_no_query": False,
        }
        agent = None
        replacement = None
        try:
            agent = NaviSchedAgent(
                "restart_agent_biz",
                server_port=grpc_port,
                total_blocks=1024,
                block_size=256,
                linear_step=1,
                hybrid_tail_required=False,
                gig={"grpc_port": grpc_port},
                name="restart_agent_first",
                runtime_config=self._make_runtime_config(thread_num=4),
                **options,
            ).init()
            with NaviSchedMaster(
                "restart_master_biz",
                name="restart_master",
                runtime_config=self._make_runtime_config(
                    thread_num=8,
                    local_nodes=[("restart_agent_biz", grpc_port, True)],
                ),
                **options,
            ) as master:
                master.init()
                with NaviSchedClient(
                    "restart_client_biz",
                    "restart_master_biz",
                    name="restart_client",
                    runtime_config=self._make_runtime_config(
                        thread_num=4,
                        local_nodes=[
                            ("restart_master_biz", master.grpc_port, False)],
                    ),
                    **options,
                ) as client:
                    client.init()
                    first = self._wait_for_decision(
                        client,
                        NaviSchedRequest(
                            request_id=101,
                            prompt_token_count=256,
                            block_hashes=[]),
                        "restart_master_biz",
                        "restart_agent_biz")
                    self.assertEqual(
                        f"127.0.0.1:{grpc_port}", first.target.address)

                    agent.stop()
                    agent = None
                    self._wait_for_no_decision(
                        client,
                        NaviSchedRequest(
                            request_id=102,
                            prompt_token_count=256,
                            block_hashes=[]),
                        "restart_master_biz")
                    replacement = NaviSchedAgent(
                        "restart_agent_biz",
                        server_port=grpc_port,
                        total_blocks=1024,
                        block_size=256,
                        linear_step=1,
                        hybrid_tail_required=False,
                        gig={"grpc_port": grpc_port},
                        name="restart_agent_second",
                        runtime_config=self._make_runtime_config(thread_num=4),
                        **options,
                    ).init()
                    second = self._wait_for_decision(
                        client,
                        NaviSchedRequest(
                            request_id=103,
                            prompt_token_count=256,
                            block_hashes=[]),
                        "restart_master_biz",
                        "restart_agent_biz")
                    self.assertEqual(
                        f"127.0.0.1:{grpc_port}", second.target.address)
        finally:
            if replacement is not None:
                replacement.stop()
            if agent is not None:
                agent.stop()

    @staticmethod
    def _create_agents(agent_count):
        agents = []
        try:
            for index in range(agent_count):
                grpc_port = 18000 + index
                agent = NaviSchedAgent(
                    "agent_biz",
                    server_port=grpc_port,
                    total_blocks=1024,
                    block_size=256,
                    linear_step=1,
                    hybrid_tail_required=False,
                    name=f"agent_{grpc_port}",
                    gig={"grpc_port": grpc_port},
                    runtime_config=NaviSchedPythonRoleTest._make_runtime_config(
                        thread_num=16),
                    enable_kmon=False,
                    enable_snapshot=False,
                    wait_no_query=False,
                )
                agent.init()
                agents.append(agent)
            return agents
        except Exception:
            for agent in reversed(agents):
                agent.stop()
            raise

    @staticmethod
    def _wait_for_decision(
            client, request, master_biz_name, target_biz_name="agent_biz"):
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            decision = client.schedule(
                request,
                target_biz_name,
                master_biz_name=master_biz_name,
                timeout_ms=1000,
            )
            if decision.target is not None:
                return decision
            time.sleep(0.01)
        raise TimeoutError("agents were not discovered")

    @staticmethod
    def _wait_for_no_decision(client, request, master_biz_name):
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            decision = client.schedule(
                request,
                "restart_agent_biz",
                master_biz_name=master_biz_name,
                timeout_ms=1000,
            )
            if decision.target is None:
                return
            time.sleep(0.01)
        raise TimeoutError("stopped Agent remained schedulable")

    @staticmethod
    def _make_runtime_config(thread_num=64, local_nodes=()):
        return {
            "thread_num": thread_num,
            "local_nodes": [
                {
                    "biz_name": biz_name,
                    "grpc_port": grpc_port,
                    "support_heartbeat": support_heartbeat,
                }
                for biz_name, grpc_port, support_heartbeat in local_nodes
            ],
        }

if __name__ == "__main__":
    unittest.main()
