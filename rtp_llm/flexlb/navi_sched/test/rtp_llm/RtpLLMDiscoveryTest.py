import asyncio
import enum
import json
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class RoleType(enum.IntEnum):
    FRONTEND = 0
    PREFILL = 1
    DECODE = 2


class RoleAddr:
    def __init__(self, role, ip, http_port, grpc_port):
        self.role = role
        self.ip = ip
        self.http_port = http_port
        self.grpc_port = grpc_port


class FlexlbResponse:
    @classmethod
    def ok(cls, role_addrs, enqueued_by_master=False):
        return SimpleNamespace(
            role_addrs=role_addrs,
            enqueued_by_master=enqueued_by_master,
        )

    @classmethod
    def error_response(cls, error_code, error_message=None):
        return SimpleNamespace(
            error_code=error_code,
            error_message=error_message,
        )


generate_config = types.ModuleType("rtp_llm.config.generate_config")
generate_config.RoleAddr = RoleAddr
generate_config.RoleType = RoleType
exceptions = types.ModuleType("rtp_llm.config.exceptions")
exceptions.ExceptionType = SimpleNamespace(MASTER_NO_AVAILABLE_WORKER=8001)
master_client = types.ModuleType("rtp_llm.server.master_client")
master_client.FlexlbResponse = FlexlbResponse
sys.modules[generate_config.__name__] = generate_config
sys.modules[exceptions.__name__] = exceptions
sys.modules[master_client.__name__] = master_client

from navi_sched.adapters.rtp_llm.NaviSchedAdapterConfig import (  # noqa: E402
    NaviSchedAdapterConfig,
)
from navi_sched.adapters.rtp_llm.NaviSchedClient import (  # noqa: E402
    NaviSchedClient,
    apply_frontend_request_overrides,
    create_navi_sched_client,
)
from navi_sched.adapters.rtp_llm.SwitchableMasterClient import (  # noqa: E402
    AutoDetectMasterClient,
    SwitchableMasterClient,
    _install_navi_sched_master_health_fallback,
    _probe_navi_sched_master,
    create_auto_detect_master_client,
)


class FakeNativeClient:
    def __init__(self):
        self.stopped = False

    def init(self):
        return self

    def stop(self):
        self.stopped = True


class FakeNaviSchedModule:
    def __init__(self):
        self.master_biz_name = None
        self.options = None
        self.clients = []

    def NaviSchedClient(self, _biz_name, master_biz_name, **options):
        self.master_biz_name = master_biz_name
        self.options = options
        client = FakeNativeClient()
        self.clients.append(client)
        return client


class FakeVipServer:
    def __init__(self, domain, hosts):
        self.domain = domain
        self.use_local = False
        self.hosts = hosts

    def get_hosts(self, refresh=False):
        return self.hosts


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def model_service_config(use_local=False):
    return json.dumps({
        "service_id": "generation.model",
        "use_local": use_local,
        "master_endpoint": {
            "address": (
                "127.0.0.1:7001" if use_local else "model.master.vip"
            ),
            "protocol": "http",
        },
        "role_endpoints": [{
            "group": "default",
            "prefill_endpoint": {
                "address": (
                    "127.0.0.1:8000" if use_local else "model.prefill.vip"
                ),
                "protocol": "http",
            },
        }],
    })


class RtpLLMDiscoveryTest(unittest.TestCase):
    def test_switchable_client_preserves_forward_payload_for_navi_sched(self):
        class CapturingClient:
            def __init__(self):
                self.calls = []

            async def get_backend_role_addrs(self, **kwargs):
                self.calls.append(kwargs)
                return "scheduled"

        navi_client = CapturingClient()
        client = SwitchableMasterClient.__new__(SwitchableMasterClient)
        client._route_subscriber = SimpleNamespace(master_address="")
        client._navi_sched_client = navi_client
        client._flexlb_client = CapturingClient()
        input_pb = object()

        result = asyncio.run(client.get_backend_role_addrs(input_pb=input_pb))

        self.assertEqual("scheduled", result)
        self.assertIs(input_pb, navi_client.calls[0]["input_pb"])

    def test_probe_accepts_default_navi_master_route(self):
        client_module = sys.modules[_probe_navi_sched_master.__module__]
        with patch.object(client_module, "build_opener") as build_opener:
            build_opener.return_value.open.return_value = FakeHttpResponse({
                "accepted": True,
                "generation": 1,
            })
            detected = _probe_navi_sched_master("127.0.0.1:7001", 0.2)
        self.assertTrue(detected)

    def test_probe_rejects_non_navi_response(self):
        client_module = sys.modules[_probe_navi_sched_master.__module__]
        with patch.object(client_module, "build_opener") as build_opener:
            build_opener.return_value.open.return_value = FakeHttpResponse({
                "generation": 1,
            })
            detected = _probe_navi_sched_master("127.0.0.1:7001", 0.2)
        self.assertFalse(detected)

    def test_auto_detect_uses_raw_vip_host_when_flexlb_health_check_fails(self):
        client = AutoDetectMasterClient.__new__(AutoDetectMasterClient)
        client._host_service = SimpleNamespace(
            get_master_addr=lambda: None,
            master_vip=FakeVipServer(
                "model.master.vip",
                [SimpleNamespace(ip="10.0.0.7", port=7001)],
            ),
        )
        self.assertEqual("10.0.0.7:7001", client._master_probe_address())

    def test_health_fallback_preserves_flexlb_master_info(self):
        expected = {
            "real_master_host": "10.0.0.7:7001",
            "queue_length": 19,
        }
        master_service = SimpleNamespace(
            _probe_single_host=lambda host: expected,
        )
        host_service = SimpleNamespace(master_service=master_service)
        client_module = sys.modules[_probe_navi_sched_master.__module__]
        with patch.object(client_module, "_probe_navi_sched_master") as probe:
            _install_navi_sched_master_health_fallback(host_service)
            actual = master_service._probe_single_host(
                SimpleNamespace(ip="10.0.0.7", port=7001)
            )
        self.assertIs(expected, actual)
        probe.assert_not_called()

    def test_health_fallback_recognizes_navi_master(self):
        master_service = SimpleNamespace(
            _probe_single_host=lambda host: None,
        )
        host_service = SimpleNamespace(master_service=master_service)
        client_module = sys.modules[_probe_navi_sched_master.__module__]
        with patch.object(
            client_module, "_probe_navi_sched_master", return_value=True
        ) as probe:
            _install_navi_sched_master_health_fallback(host_service)
            actual = master_service._probe_single_host(
                SimpleNamespace(ip="10.0.0.7", port=7001)
            )
        self.assertEqual(
            {
                "real_master_host": "10.0.0.7:7001",
                "queue_length": 0,
            },
            actual,
        )
        probe.assert_called_once_with("10.0.0.7:7001", 0.2)

    def test_prefill_role_does_not_create_client(self):
        config = NaviSchedAdapterConfig(enabled=True, biz_name="generation.model")
        client_module = sys.modules[create_navi_sched_client.__module__]
        with patch.dict(os.environ, {"ROLE_TYPE": "PREFILL"}, clear=True):
            with patch.object(client_module, "NaviSchedClient") as client_class:
                client = create_navi_sched_client(object(), config=config)
        self.assertIsNone(client)
        client_class.assert_not_called()

    def test_frontend_role_creates_client(self):
        config = NaviSchedAdapterConfig(enabled=True, biz_name="generation.model")
        client_module = sys.modules[create_navi_sched_client.__module__]
        host_service = object()
        with patch.dict(os.environ, {"ROLE_TYPE": "FRONTEND"}, clear=True):
            with patch.object(client_module, "NaviSchedClient") as client_class:
                client = create_navi_sched_client(host_service, config=config)
        self.assertIs(client_class.return_value, client)
        client_class.assert_called_once_with(
            config,
            host_service=host_service,
            enable_agent_forward=True,
        )

    def test_prefill_role_does_not_create_auto_detect_client(self):
        flexlb_client = object()
        client_module = sys.modules[create_auto_detect_master_client.__module__]
        with patch.dict(os.environ, {"ROLE_TYPE": "PREFILL"}, clear=True):
            with patch.object(
                client_module, "AutoDetectMasterClient"
            ) as client_class:
                client = create_auto_detect_master_client(
                    flexlb_client, object()
                )
        self.assertIs(flexlb_client, client)
        client_class.assert_not_called()

    def test_frontend_role_creates_auto_detect_client(self):
        config = NaviSchedAdapterConfig(enabled=True, biz_name="generation.model")
        flexlb_client = object()
        host_service = object()
        client_module = sys.modules[create_auto_detect_master_client.__module__]
        with patch.dict(os.environ, {"ROLE_TYPE": "FRONTEND"}, clear=True):
            with patch.object(
                client_module.NaviSchedAdapterConfig,
                "from_env",
                return_value=config,
            ), patch.object(
                client_module, "AutoDetectMasterClient"
            ) as client_class:
                client = create_auto_detect_master_client(
                    flexlb_client, host_service
                )
        self.assertIs(client_class.return_value, client)
        client_class.assert_called_once_with(
            flexlb_client, host_service, config
        )

    def test_config_uses_fixed_master_and_derived_prefill_biz(self):
        config = NaviSchedAdapterConfig.from_env({
            "RTP_LLM_MASTER_TYPE": "navi_sched",
            "MODEL_SERVICE_CONFIG": model_service_config(),
        })
        self.assertEqual("rtp_llm_navi_sched_master", config.master_biz_name)
        self.assertEqual("model.prefill.vip", config.biz_name)

    def test_unified_config_enables_client_and_uses_rtp_biz_name(self):
        config = NaviSchedAdapterConfig.from_env({
            "BIZ_NAME": "generation.model",
            "MODEL_SERVICE_CONFIG": json.dumps({
                "decode_endpoint": {
                    "address": "model.decode.vip",
                    "protocol": "http",
                },
            }),
            "RTP_LLM_NAVI_SCHED_CONFIG_JSON": json.dumps({
                "client": {"target_role": "PREFILL"},
            }),
        })
        self.assertTrue(config.enabled)
        self.assertEqual("generation.model", config.biz_name)

    def test_unified_config_forces_one_output_token_in_flexlb_mode(self):
        environ = {
            "RTP_LLM_MASTER_TYPE": "flexlb",
            "RTP_LLM_NAVI_SCHED_CONFIG_JSON": json.dumps({
                "client": {"force_max_new_tokens": 1},
            }),
        }
        request = SimpleNamespace(
            request_id=7,
            generate_config=SimpleNamespace(max_new_tokens=128),
        )

        with patch.dict(os.environ, environ, clear=True):
            apply_frontend_request_overrides(request)

        self.assertEqual(1, request.generate_config.max_new_tokens)

    def test_vip_master_uses_gig_subscription(self):
        environ = {
            "RTP_LLM_MASTER_TYPE": "navi_sched",
            "MODEL_SERVICE_CONFIG": model_service_config(),
            "com.alibaba.vipserver.jmenv": "jmenv.test",
        }
        module = FakeNaviSchedModule()
        host_service = SimpleNamespace(master_vip=FakeVipServer(
            "model.master.vip",
            [SimpleNamespace(ip="10.0.0.7", port=7001)],
        ))
        with patch.dict(os.environ, environ, clear=True):
            client = NaviSchedClient(
                NaviSchedAdapterConfig.from_env(),
                host_service=host_service,
                navi_sched_module=module,
            )
        self.assertIsNotNone(client)
        self.assertEqual("rtp_llm_navi_sched_master", module.master_biz_name)
        self.assertEqual("sched_client", module.options["name"])
        self.assertFalse(module.options["expect_agent_response_stream"])
        self.assertEqual({
            "jmenv_domain": "jmenv.test",
            "services": ["model.master.vip"],
            "grpc_port_diff": 2,
        }, module.options["runtime_config"]["vip_subscription"])

    def test_replay_filename_does_not_include_process_id(self):
        with tempfile.TemporaryDirectory() as replay_dir:
            client = object.__new__(NaviSchedClient)
            client.config = NaviSchedAdapterConfig(
                enabled=True,
                biz_name="generation.model",
                replay_dir=replay_dir,
            )
            decision = SimpleNamespace(
                replay=SimpleNamespace(to_json=lambda: "{}"),
                request_timing=None,
                request_id=19,
                schedule_id=7,
                schedule_loss=0.0,
                schedule_latency_us=1,
                batch_id=3,
                expected_batch_size=1,
                predicted_cache_hit_tokens=0,
                predicted_queue_latency_ms=0.0,
                predicted_execution_latency_ms=1.0,
                target=SimpleNamespace(address="127.0.0.1:7001"),
            )

            client._write_replay(decision, 8, [1])

            self.assertEqual(
                ["schedule_7_request_19.json"], os.listdir(replay_dir)
            )

    def test_local_master_http_port_maps_to_navi_grpc_p_plus_two(self):
        environ = {
            "RTP_LLM_MASTER_TYPE": "navi_sched",
            "MODEL_SERVICE_CONFIG": model_service_config(use_local=True),
        }
        module = FakeNaviSchedModule()
        with patch.dict(os.environ, environ, clear=True):
            NaviSchedClient(
                NaviSchedAdapterConfig.from_env(),
                navi_sched_module=module,
            )
        self.assertEqual([{
            "biz_name": "rtp_llm_navi_sched_master",
            "ip": "127.0.0.1",
            "grpc_port": 7003,
        }], module.options["runtime_config"]["local_nodes"])

    def test_vip_master_topology_is_owned_by_gig(self):
        environ = {
            "RTP_LLM_MASTER_TYPE": "navi_sched",
            "MODEL_SERVICE_CONFIG": model_service_config(),
        }
        module = FakeNaviSchedModule()
        vip = FakeVipServer(
            "model.master.vip",
            [SimpleNamespace(ip="10.0.0.7", port=7001)],
        )
        host_service = SimpleNamespace(master_vip=vip)
        with patch.dict(os.environ, environ, clear=True):
            client = NaviSchedClient(
                NaviSchedAdapterConfig.from_env(),
                host_service=host_service,
                navi_sched_module=module,
            )
        self.assertEqual(1, len(module.clients))
        self.assertIs(module.clients[0], client._client)
        self.assertEqual(["model.master.vip"],
                         module.options["runtime_config"]["vip_subscription"]["services"])

    def test_service_route_rejects_string_use_local(self):
        value = json.loads(model_service_config())
        value["use_local"] = "false"
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            NaviSchedAdapterConfig.from_env({
                "RTP_LLM_MASTER_TYPE": "navi_sched",
                "MODEL_SERVICE_CONFIG": json.dumps(value),
            })


if __name__ == "__main__":
    unittest.main()
