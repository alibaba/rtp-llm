import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path


def _load_master_client_module():
    rtp_root = Path(__file__).resolve().parents[2]
    for name, path in (
        ("rtp_llm", rtp_root),
        ("rtp_llm.server", rtp_root / "server"),
        ("rtp_llm.config", rtp_root / "config"),
        ("rtp_llm.utils", rtp_root / "utils"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    exceptions = types.ModuleType("rtp_llm.config.exceptions")

    class ExceptionType(IntEnum):
        MASTER_NO_AVAILABLE_WORKER = 8400

    class FtRuntimeException(Exception):
        def __init__(self, exception_type, message):
            self.exception_type = exception_type
            super().__init__(message)

    exceptions.ExceptionType = ExceptionType
    exceptions.FtRuntimeException = FtRuntimeException
    sys.modules[exceptions.__name__] = exceptions

    generate_config = types.ModuleType("rtp_llm.config.generate_config")

    class RoleType(Enum):
        PREFILL = "PREFILL"

    @dataclass
    class RoleAddr:
        role: RoleType
        ip: str
        http_port: int
        grpc_port: int

    generate_config.RoleType = RoleType
    generate_config.RoleAddr = RoleAddr
    sys.modules[generate_config.__name__] = generate_config

    host_service = types.ModuleType("rtp_llm.server.host_service")
    host_service.HostService = object
    host_service.VipServerWrapper = object
    sys.modules[host_service.__name__] = host_service

    worker_status = types.ModuleType("rtp_llm.server.worker_status")
    worker_status.ScheduleMeta = object
    sys.modules[worker_status.__name__] = worker_status

    datatypes = types.ModuleType("rtp_llm.utils.base_model_datatypes")
    datatypes.GenerateInput = object
    sys.modules[datatypes.__name__] = datatypes

    module_name = "rtp_llm.server.master_client"
    spec = importlib.util.spec_from_file_location(
        module_name, rtp_root / "server" / "master_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


master_client = _load_master_client_module()


class _HostService:
    class _Vip:
        domain = ""

    master_vip = _Vip()

    def __init__(self, master=None, slave=None):
        self.master = master
        self.slave = slave

    def get_master_addr(self):
        return self.master

    def get_slave_addr(self):
        return self.slave


class _KvcmClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0
        self.last_kwargs = None
        self.closed = False

    async def query_and_select(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.result

    async def close(self):
        self.closed = True


def _config(enabled=True):
    return types.SimpleNamespace(
        master_max_connect_pool_size=10,
        master_session_timeout_s=-1,
        master_default_timeout_ms=1_000,
        master_kvcm_fallback_enabled=enabled,
        master_flexlb_transport_timeout_ms=50,
    )


def _input():
    return types.SimpleNamespace(
        prompt_length=12,
        input_ids=list(range(12)),
        generate_config=types.SimpleNamespace(
            ttft_timeout_ms=1_000,
            timeout_ms=1_000,
            traffic_reject_priority=7,
        ),
    )


def _selected_result():
    selected = types.SimpleNamespace(
        host_ip="10.0.0.2",
        http_port=8080,
        grpc_port=8001,
        host_ip_port="10.0.0.2:8080",
        local_blocks=5,
        p2p_fetch_blocks=0,
        p2p_total_match_blocks=5,
    )
    return types.SimpleNamespace(
        selected=selected,
        outcome="selected",
        candidate_count=2,
        block_count=3,
        latency_us=10,
    )


class MasterClientKvcmFallbackTest(unittest.IsolatedAsyncioTestCase):
    async def test_local_worker_is_added_to_kvcm_candidate_pool(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService(),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )
        local = master_client.RoleAddr(
            role=master_client.RoleType.PREFILL,
            ip="10.0.0.9",
            http_port=8000,
            grpc_port=8001,
        )
        try:
            result = await client.get_backend_role_addrs(
                [], _input(), "request-local", local_fallback_addr=local
            )
            self.assertTrue(result.is_ok)
            candidate = kvcm_client.last_kwargs["local_candidate"]
            self.assertEqual("10.0.0.9", candidate.host_ip)
            self.assertEqual(8001, candidate.worker_status_port)
            self.assertEqual(0, candidate.local_blocks)
        finally:
            await client.close()

    async def test_no_master_uses_kvcm_max_affinity_route(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService(),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )
        try:
            result = await client.get_backend_role_addrs([], _input(), "request-1")
            self.assertTrue(result.is_ok)
            self.assertEqual("KVCM", result.route_source)
            self.assertEqual("10.0.0.2", result.role_addrs[0].ip)
            self.assertEqual(8001, result.role_addrs[0].grpc_port)
            self.assertEqual(5, result.cache_match["local_blocks"])
            self.assertEqual(1, kvcm_client.calls)
        finally:
            await client.close()
        self.assertTrue(kvcm_client.closed)

    async def test_switch_off_preserves_connection_failure_behavior(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService(),
            master_config=_config(enabled=False),
            kvcm_fallback_client=kvcm_client,
        )
        result = await client.get_backend_role_addrs([], _input(), "request-2")
        self.assertTrue(result.connection_failed)
        self.assertEqual(0, kvcm_client.calls)
        await client.close()

    async def test_master_and_slave_transport_fail_before_kvcm(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService("master:7001", "slave:7001"),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )
        attempts = []

        async def fail_transport(addr, payload, timeout_ms, request_id):
            attempts.append((addr, timeout_ms))
            return master_client.FlexlbResponse.connection_failed_response()

        client._send_schedule_request = fail_transport
        result = await client.get_backend_role_addrs([], _input(), "request-3")
        self.assertEqual([("master:7001", 50), ("slave:7001", 50)], attempts)
        self.assertEqual("KVCM", result.route_source)
        self.assertEqual(1, kvcm_client.calls)
        await client.close()

    async def test_flexlb_business_error_does_not_query_kvcm(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService("master:7001"),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )

        async def business_error(addr, payload, timeout_ms, request_id):
            return master_client.FlexlbResponse.error_response(8400)

        client._send_schedule_request = business_error
        result = await client.get_backend_role_addrs([], _input(), "request-4")
        self.assertEqual(8400, result.error_code)
        self.assertEqual(0, kvcm_client.calls)
        await client.close()

    async def test_explicit_8600_does_not_query_kvcm(self):
        kvcm_client = _KvcmClient(result=_selected_result())
        client = master_client.MasterClient(
            host_service=_HostService("master:7001"),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )

        async def explicit_fallback(addr, payload, timeout_ms, request_id):
            return master_client.FlexlbResponse.ok_with_result({"code": 8600})

        client._send_schedule_request = explicit_fallback
        result = await client.get_backend_role_addrs([], _input(), "request-5")
        self.assertTrue(result.fallback)
        self.assertEqual(0, kvcm_client.calls)
        await client.close()

    async def test_kvcm_failure_preserves_final_caller_fallback(self):
        kvcm_client = _KvcmClient(error=RuntimeError("kvcm unavailable"))
        client = master_client.MasterClient(
            host_service=_HostService(),
            master_config=_config(),
            kvcm_fallback_client=kvcm_client,
        )
        result = await client.get_backend_role_addrs([], _input(), "request-6")
        self.assertTrue(result.connection_failed)
        self.assertIsNone(result.role_addrs)
        self.assertEqual(1, kvcm_client.calls)
        await client.close()


if __name__ == "__main__":
    unittest.main()
