import enum
import importlib
import os
import shutil
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class RoleType(enum.IntEnum):
    FRONTEND = 0
    PREFILL = 1
    DECODE = 2
    VIT = 3


class RoleAddr:
    def __init__(self, role, ip, http_port, grpc_port):
        self.role = role
        self.ip = ip
        self.http_port = http_port
        self.grpc_port = grpc_port


class FlexlbResponse:
    def __init__(self, role_addrs=None, error_code=None, error_message=None):
        self.role_addrs = role_addrs
        self.error_code = error_code
        self.error_message = error_message

    @property
    def is_ok(self):
        return self.role_addrs is not None

    @classmethod
    def ok(cls, role_addrs, enqueued_by_master=False):
        return cls(role_addrs=role_addrs)

    @classmethod
    def error_response(cls, error_code, error_message=None):
        return cls(error_code=error_code, error_message=error_message)


class ExceptionType(enum.IntEnum):
    MASTER_NO_AVAILABLE_WORKER = 8001


generate_config_module = types.ModuleType("rtp_llm.config.generate_config")
generate_config_module.RoleAddr = RoleAddr
generate_config_module.RoleType = RoleType
exceptions_module = types.ModuleType("rtp_llm.config.exceptions")
exceptions_module.ExceptionType = ExceptionType
exceptions_module.FtRuntimeException = Exception
master_client_module = types.ModuleType("rtp_llm.server.master_client")
master_client_module.FlexlbResponse = FlexlbResponse
sys.modules[generate_config_module.__name__] = generate_config_module
sys.modules[exceptions_module.__name__] = exceptions_module
sys.modules[master_client_module.__name__] = master_client_module

integration = importlib.import_module("rtp_llm.server.navi_sched_adapter")
adapter = importlib.import_module("navi_sched.adapters.rtp_llm")
NaviSchedAdapterConfig = adapter.NaviSchedAdapterConfig
NaviSchedClient = adapter.NaviSchedClient
apply_frontend_request_overrides = adapter.apply_frontend_request_overrides
AutoDetectMasterClient = adapter.AutoDetectMasterClient
split_address = adapter.split_address
get_schedule_timeout_ms = adapter.get_schedule_timeout_ms


class _FakeTokenIds:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value

    def numel(self):
        value = (
            self._value[0]
            if self._value and isinstance(self._value[0], list)
            else self._value
        )
        return len(value)


class _FakeNativeClient:
    def __init__(
        self,
        batch_id="",
        expected_batch_size=0,
        targets=None,
    ):
        self.batch_id = batch_id
        self.expected_batch_size = expected_batch_size
        self.targets = (
            [SimpleNamespace(role=1, address="127.0.0.1:9001")]
            if targets is None
            else targets
        )
        self.calls = []
        self.closed = False

    def init(self):
        return self

    def schedule(self, prompt_token_count, **kwargs):
        self.calls.append((prompt_token_count, kwargs))
        return SimpleNamespace(
            reason="" if self.targets else "no target",
            batch_id=self.batch_id,
            expected_batch_size=self.expected_batch_size,
            target=self.targets[0] if self.targets else None,
            predicted_cache_hit_tokens=0,
            predicted_queue_latency_ms=0.0,
            predicted_execution_latency_ms=0.0,
            schedule_id=1,
            schedule_loss=0.0,
            schedule_latency_us=0,
            schedule_time_us=0,
            request_id=1,
            request_timing=None,
            replay=None,
        )

    def stop(self):
        self.closed = True


class _FakeNaviSchedModule:
    def __init__(self):
        self.agent = None
        self.client = None
        self.navi = SimpleNamespace(
            NaviSchedStrategy=SimpleNamespace(
                DEFAULT=0,
                CUSTOM=1,
                RANDOM=2,
            ),
            NaviSchedStrategyParams=self._make_strategy_params,
        )

    @staticmethod
    def _make_strategy_params():
        return SimpleNamespace(score_scale=0.0)

    def NaviSchedClient(self, _client_biz_name, master_biz_name, **kwargs):
        self.client = _FakeNativeClient()
        self.client.master_biz_name = master_biz_name
        self.client.kwargs = kwargs
        return self.client


class _FakeHostService:
    def __init__(self, domain, use_local):
        self.master_vip = SimpleNamespace(domain=domain, use_local=use_local)

    def get_master_addr(self):
        return self.master_vip.domain or None


class _FakeRoutingClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.closed = False

    def get_latest_queue_length(self):
        return 0

    async def get_backend_role_addrs(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def close(self):
        self.closed = True


class NaviSchedAdapterConfigTest(unittest.TestCase):
    def test_from_env_is_disabled_by_default(self):
        config = NaviSchedAdapterConfig.from_env({})

        self.assertFalse(config.enabled)

    def test_master_type_selects_navi_sched(self):
        config = NaviSchedAdapterConfig.from_env({"RTP_LLM_MASTER_TYPE": "navi_sched"})

        self.assertTrue(config.enabled)

    def test_obsolete_enable_flag_is_ignored(self):
        config = NaviSchedAdapterConfig.from_env({"RTP_LLM_NAVI_SCHED_ENABLED": "true"})

        self.assertFalse(config.enabled)

    def test_unified_config_reads_force_max_new_tokens_in_flexlb_mode(self):
        config = NaviSchedAdapterConfig.from_env(
            {
                "RTP_LLM_MASTER_TYPE": "flexlb",
                "RTP_LLM_NAVI_SCHED_CONFIG_JSON": (
                    '{"client":{"force_max_new_tokens":1}}'
                ),
            }
        )

        self.assertEqual(config.force_max_new_tokens, 1)

    def test_negative_force_max_new_tokens_is_rejected(self):
        config = NaviSchedAdapterConfig(
            biz_name="model",
            force_max_new_tokens=-1,
        )

        with self.assertRaisesRegex(ValueError, "force_max_new_tokens"):
            config.validate_client()

    def test_frontend_request_override_forces_output_length(self):
        request = SimpleNamespace(
            request_id=7,
            generate_config=SimpleNamespace(max_new_tokens=128),
        )

        apply_frontend_request_overrides(
            request,
            NaviSchedAdapterConfig(force_max_new_tokens=1),
        )

        self.assertEqual(request.generate_config.max_new_tokens, 1)

    def test_invalid_master_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "RTP_LLM_MASTER_TYPE"):
            NaviSchedAdapterConfig.from_env({"RTP_LLM_MASTER_TYPE": "unknown"})

    def test_from_env_reads_all_role_settings(self):
        config = NaviSchedAdapterConfig.from_env(
            {
                "RTP_LLM_MASTER_TYPE": "navi_sched",
                "RTP_LLM_NAVI_SCHED_MASTER_BIZ_NAME": "master",
                "RTP_LLM_NAVI_SCHED_BIZ_NAME": "model",
                "RTP_LLM_NAVI_SCHED_TARGET_ROLE": "decode",
            }
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.master_biz_name, "master")
        self.assertEqual(config.biz_name, "model")
        self.assertEqual(config.get_target_role(), RoleType.DECODE)

    def test_split_address_supports_ipv4_and_ipv6(self):
        self.assertEqual(split_address("127.0.0.1:9000"), ("127.0.0.1", 9000))
        self.assertEqual(split_address("[::1]:9001"), ("::1", 9001))

    def test_timeout_uses_positive_request_deadline(self):
        config = SimpleNamespace(ttft_timeout_ms=-1, timeout_ms=800)

        self.assertEqual(get_schedule_timeout_ms(config), 800)

    def test_timeout_uses_default_when_deadlines_are_disabled(self):
        config = SimpleNamespace(ttft_timeout_ms=-1, timeout_ms=-1)

        self.assertEqual(get_schedule_timeout_ms(config), 1000)


class NaviSchedIntegrationTest(unittest.TestCase):
    def test_flexlb_mode_does_not_load_navi_sched_wheel(self):
        with patch.dict(
            "os.environ", {"RTP_LLM_MASTER_TYPE": "flexlb"}, clear=True
        ), patch.object(integration, "_load_adapter") as load_adapter:
            client = integration.create_navi_sched_client(
                _FakeHostService("flexlb.master.domain", use_local=False)
            )

        self.assertIsNone(client)
        load_adapter.assert_not_called()

    def test_navi_sched_mode_loads_wheel_adapter(self):
        wheel_adapter = SimpleNamespace(
            create_navi_sched_client=lambda host_service: "client",
        )
        with patch.dict(
            "os.environ", {"RTP_LLM_MASTER_TYPE": "navi_sched"}, clear=True
        ), patch.object(integration, "_load_adapter", return_value=wheel_adapter):
            client = integration.create_navi_sched_client(
                _FakeHostService("navi.master.domain", use_local=False)
            )

        self.assertEqual("client", client)

    def test_unified_config_loads_wheel_adapter_by_default(self):
        wheel_adapter = SimpleNamespace(
            create_navi_sched_client=lambda host_service: "client",
        )
        with patch.dict(
            "os.environ",
            {"RTP_LLM_NAVI_SCHED_CONFIG_JSON": '{"client": {}}'},
            clear=True,
        ), patch.object(integration, "_load_adapter", return_value=wheel_adapter):
            client = integration.create_navi_sched_client(
                _FakeHostService("navi.master.domain", use_local=False)
            )

        self.assertEqual("client", client)

    def test_explicit_flexlb_overrides_unified_config(self):
        with patch.dict(
            "os.environ",
            {
                "RTP_LLM_MASTER_TYPE": "flexlb",
                "RTP_LLM_NAVI_SCHED_CONFIG_JSON": '{"client": {}}',
            },
            clear=True,
        ), patch.object(integration, "_load_adapter") as load_adapter:
            client = integration.create_navi_sched_client(
                _FakeHostService("flexlb.master.domain", use_local=False)
            )

        self.assertIsNone(client)
        load_adapter.assert_not_called()

    def test_request_override_runs_in_flexlb_mode(self):
        request = SimpleNamespace()
        wheel_adapter = SimpleNamespace(
            apply_frontend_request_overrides=lambda value: setattr(
                value, "overridden", True
            )
        )
        with patch.dict(
            "os.environ",
            {
                "RTP_LLM_MASTER_TYPE": "flexlb",
                "RTP_LLM_NAVI_SCHED_CONFIG_JSON": (
                    '{"client":{"force_max_new_tokens":1}}'
                ),
            },
            clear=True,
        ), patch.object(integration, "_load_adapter", return_value=wheel_adapter):
            integration.apply_navi_sched_request_overrides(request)

        self.assertTrue(request.overridden)

    def test_request_override_does_not_load_adapter_without_config(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(
            integration, "_load_adapter"
        ) as load_adapter:
            integration.apply_navi_sched_request_overrides(SimpleNamespace())

        load_adapter.assert_not_called()

    def test_mode_helper_preserves_flexlb_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(integration.navi_sched_enabled())

        with patch.dict(
            "os.environ", {"RTP_LLM_MASTER_TYPE": "navi_sched"}, clear=True
        ):
            self.assertTrue(integration.navi_sched_enabled())

    def test_backend_roles_use_navi_sched_target_role(self):
        default_roles = [RoleType.PREFILL, RoleType.DECODE]
        client = SimpleNamespace(
            config=SimpleNamespace(get_target_role=lambda: RoleType.PREFILL)
        )

        self.assertEqual(
            integration.get_navi_sched_backend_roles(default_roles, client),
            [RoleType.PREFILL],
        )

    def test_backend_roles_preserve_flexlb_defaults(self):
        default_roles = [RoleType.PREFILL, RoleType.DECODE]

        self.assertIs(
            integration.get_navi_sched_backend_roles(default_roles, None),
            default_roles,
        )

    def test_backend_roles_preserve_defaults_for_flexlb_client(self):
        default_roles = [RoleType.PREFILL, RoleType.DECODE]

        self.assertIs(
            integration.get_navi_sched_backend_roles(default_roles, object()),
            default_roles,
        )


class AutoDetectMasterClientTest(unittest.IsolatedAsyncioTestCase):
    def _config(self):
        return NaviSchedAdapterConfig(
            enabled=True,
            master_biz_name="master",
            biz_name="model",
            target_role="PREFILL",
        )

    @staticmethod
    def _response(ok):
        return SimpleNamespace(is_ok=ok)

    def _client(self, probe_values, flexlb_responses, navi_responses):
        values = iter(probe_values)
        probe = lambda _address, _timeout: next(values)
        flexlb = _FakeRoutingClient(*flexlb_responses)
        navi = _FakeRoutingClient(*navi_responses)
        client = AutoDetectMasterClient(
            flexlb,
            _FakeHostService("master:1234", use_local=True),
            config=self._config(),
            navi_sched_client=navi,
            probe=probe,
            probe_interval_s=60,
        )
        return client, flexlb, navi

    async def test_initial_probe_selects_flexlb(self):
        success = self._response(True)
        client, flexlb, navi = self._client([False], [success], [])

        await client.prepare_input()
        result = await client.get_backend_role_addrs(request_id=1)

        self.assertIs(success, result)
        self.assertFalse(client.using_navi_sched())
        self.assertEqual(1, len(flexlb.calls))
        self.assertFalse(navi.calls)

    async def test_initial_probe_selects_navi_sched(self):
        success = self._response(True)
        client, flexlb, navi = self._client([True], [], [success])

        await client.prepare_input()
        result = await client.get_backend_role_addrs(request_id=1)

        self.assertIs(success, result)
        self.assertTrue(client.using_navi_sched())
        self.assertFalse(flexlb.calls)
        self.assertEqual(1, len(navi.calls))

    async def test_same_address_can_switch_after_master_restart(self):
        client, flexlb, navi = self._client([False, True], [], [])

        await client.prepare_input()
        self.assertFalse(client.using_navi_sched())
        client._last_probe_time = 0
        await client.prepare_input()

        self.assertTrue(client.using_navi_sched())
        self.assertFalse(flexlb.calls)
        self.assertFalse(navi.calls)

    async def test_failed_route_retries_only_after_mode_changes(self):
        failed = self._response(False)
        success = self._response(True)
        client, flexlb, navi = self._client([False, True], [failed], [success])

        result = await client.get_backend_role_addrs(request_id=1)

        self.assertIs(success, result)
        self.assertEqual(1, len(flexlb.calls))
        self.assertEqual(1, len(navi.calls))

    async def test_navi_business_error_does_not_fallback(self):
        failed = self._response(False)
        client, flexlb, navi = self._client([True, True], [], [failed])

        result = await client.get_backend_role_addrs(request_id=1)

        self.assertIs(failed, result)
        self.assertTrue(client.using_navi_sched())
        self.assertFalse(flexlb.calls)
        self.assertEqual(1, len(navi.calls))

    async def test_close_closes_both_clients(self):
        client, flexlb, navi = self._client([], [], [])

        await client.close()

        self.assertTrue(flexlb.closed)
        self.assertTrue(navi.closed)


class NaviSchedClientTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._log_dir = os.path.abspath(".navi_sched/test_adapter")
        shutil.rmtree(self._log_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self._log_dir, ignore_errors=True)

    def _config(self):
        return NaviSchedAdapterConfig(
            enabled=True,
            master_biz_name="master",
            biz_name="model",
            target_role="PREFILL",
        )

    async def test_schedule_passes_block_hashes_and_returns_rtp_address(self):
        native_client = _FakeNativeClient()
        client = NaviSchedClient(self._config(), native_client=native_client)
        request = SimpleNamespace(
            token_ids=_FakeTokenIds([[1, 2, 3]]),
            group_id=-1,
            group_size=1,
            generate_config=SimpleNamespace(
                ttft_timeout_ms=250,
                timeout_ms=1000,
                max_new_tokens=32,
                role_addrs=[],
            ),
        )

        response = await client.get_backend_role_addrs([11, 22], 16, request, 123)

        self.assertTrue(response.is_ok)
        self.assertFalse(client.uses_prompt_token_ids)
        self.assertEqual(native_client.calls[0][0], 3)
        self.assertEqual(native_client.calls[0][1]["block_hashes"], [11, 22])
        self.assertEqual(native_client.calls[0][1]["target_biz_name"], "model")
        self.assertEqual(native_client.calls[0][1]["timeout_ms"], 250)
        self.assertEqual(response.role_addrs[0].role, RoleType.PREFILL)
        self.assertEqual(response.role_addrs[0].grpc_port, 9001)
        self.assertEqual(response.role_addrs[0].http_port, 9000)
        self.assertEqual(request.group_id, -1)
        self.assertEqual(request.group_size, 1)

    async def test_schedule_applies_rtp_batch_fields(self):
        native_client = _FakeNativeClient(
            batch_id="42",
            expected_batch_size=3,
        )
        client = NaviSchedClient(self._config(), native_client=native_client)
        request = SimpleNamespace(
            token_ids=_FakeTokenIds([1, 2, 3]),
            group_id=-1,
            group_size=1,
            generate_config=SimpleNamespace(
                ttft_timeout_ms=250,
                timeout_ms=1000,
                max_new_tokens=32,
                role_addrs=[],
            ),
        )

        response = await client.get_backend_role_addrs([], 16, request, 123)

        self.assertTrue(response.is_ok)
        self.assertEqual(request.group_id, 42)
        self.assertEqual(request.group_size, 3)

    async def test_schedule_preserves_other_role_addresses(self):
        native_client = _FakeNativeClient()
        client = NaviSchedClient(self._config(), native_client=native_client)
        decode = RoleAddr(
            role=RoleType.DECODE,
            ip="127.0.0.2",
            http_port=9100,
            grpc_port=9101,
        )
        request = SimpleNamespace(
            token_ids=_FakeTokenIds([1, 2, 3]),
            group_id=-1,
            group_size=1,
            generate_config=SimpleNamespace(
                ttft_timeout_ms=250,
                timeout_ms=1000,
                max_new_tokens=32,
                role_addrs=[decode],
            ),
        )

        response = await client.get_backend_role_addrs([], 16, request, 123)

        self.assertTrue(response.is_ok)
        self.assertEqual(2, len(response.role_addrs))
        self.assertEqual(RoleType.PREFILL, response.role_addrs[0].role)
        self.assertEqual(9001, response.role_addrs[0].grpc_port)
        self.assertIs(decode, response.role_addrs[1])

    async def test_schedule_rejects_non_integer_rtp_batch_id(self):
        native_client = _FakeNativeClient(
            batch_id="batch_42",
            expected_batch_size=3,
        )
        client = NaviSchedClient(self._config(), native_client=native_client)
        request = SimpleNamespace(
            token_ids=_FakeTokenIds([1, 2, 3]),
            group_id=-1,
            group_size=1,
            generate_config=SimpleNamespace(
                ttft_timeout_ms=250,
                timeout_ms=1000,
                max_new_tokens=32,
                role_addrs=[],
            ),
        )

        with self.assertRaisesRegex(ValueError, "batch_id"):
            await client.get_backend_role_addrs([], 16, request, 123)

    async def test_schedule_returns_error_when_no_target_is_available(self):
        client = NaviSchedClient(
            self._config(), native_client=_FakeNativeClient(targets=[])
        )
        request = SimpleNamespace(
            token_ids=_FakeTokenIds([1, 2]),
            group_id=-1,
            group_size=1,
            generate_config=SimpleNamespace(
                ttft_timeout_ms=100,
                timeout_ms=1000,
                max_new_tokens=8,
                role_addrs=[],
            ),
        )

        response = await client.get_backend_role_addrs([], 16, request, 123)

        self.assertFalse(response.is_ok)
        self.assertEqual(response.error_message, "no target")

    async def test_local_master_uses_direct_gig_subscription(self):
        module = _FakeNaviSchedModule()
        host_service = _FakeHostService("127.0.0.1:8000,127.0.0.2:8001", use_local=True)

        with patch.dict(
            "os.environ",
            {"LOG_PATH": self._log_dir},
            clear=True,
        ):
            client = NaviSchedClient(
                self._config(),
                host_service=host_service,
                navi_sched_module=module,
            )

        self.assertEqual(module.client.master_biz_name, "master")
        self.assertEqual(
            module.client.kwargs["name"],
            "sched_client",
        )
        self.assertEqual(
            module.client.kwargs["log_file_name"],
            os.path.join(self._log_dir, "navi_sched.log"),
        )
        self.assertEqual(
            module.client.kwargs["snapshot_dir"],
            os.path.join(self._log_dir, "navi_sched"),
        )
        self.assertEqual(
            module.client.kwargs["gig"]["snapshot_log_dir"],
            os.path.join(self._log_dir, "navi_sched"),
        )
        runtime_config = module.client.kwargs["runtime_config"]
        self.assertEqual(len(runtime_config["local_nodes"]), 2)
        self.assertEqual(
            runtime_config["local_nodes"][0],
            {"biz_name": "master", "ip": "127.0.0.1", "grpc_port": 8000},
        )
        self.assertEqual(
            runtime_config["local_nodes"][1],
            {"biz_name": "master", "ip": "127.0.0.2", "grpc_port": 8001},
        )
        await client.close()

    async def test_vip_master_uses_gig_vip_subscription(self):
        module = _FakeNaviSchedModule()
        host_service = _FakeHostService("navi.master.domain", use_local=False)
        config = NaviSchedAdapterConfig(
            enabled=True, master_biz_name="master", biz_name="model"
        )

        with patch.dict("os.environ", {"com.alibaba.vipserver.jmenv": "jmenv.example"}):
            client = NaviSchedClient(
                config,
                host_service=host_service,
                navi_sched_module=module,
            )

        self.assertEqual(module.client.master_biz_name, "master")
        runtime_config = module.client.kwargs["runtime_config"]
        self.assertEqual(
            runtime_config["vip_subscription"]["services"], ["navi.master.domain"]
        )
        self.assertEqual(
            runtime_config["vip_subscription"]["jmenv_domain"], "jmenv.example"
        )
        await client.close()


if __name__ == "__main__":
    unittest.main()
