import asyncio
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path


def _module(name, **attributes):
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    sys.modules[name] = module
    return module


def _load_visitor_module():
    rtp_root = Path(__file__).resolve().parents[2]
    for name, path in (
        ("rtp_llm", rtp_root),
        ("rtp_llm.server", rtp_root / "server"),
        ("rtp_llm.config", rtp_root / "config"),
        ("rtp_llm.cpp", rtp_root / "cpp"),
        ("rtp_llm.cpp.model_rpc", rtp_root / "cpp" / "model_rpc"),
        ("rtp_llm.metrics", rtp_root / "metrics"),
        ("rtp_llm.utils", rtp_root / "utils"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package

    class ExceptionType(IntEnum):
        TRAFFIC_LIMIT_ERROR = 1
        ROUTE_ERROR = 2

    class FtRuntimeException(Exception):
        pass

    class RoleType(Enum):
        PREFILL = "PREFILL"
        DECODE = "DECODE"
        PDFUSION = "PDFUSION"
        FRONTEND = "FRONTEND"
        VIT = "VIT"

    @dataclass
    class RoleAddr:
        role: RoleType
        ip: str
        http_port: int
        grpc_port: int

    @dataclass
    class FlexlbResponse:
        role_addrs: list | None = None
        connection_failed: bool = False
        fallback: bool = False
        error_code: int | None = None
        error_message: str | None = None

        @property
        def is_ok(self):
            return self.role_addrs is not None

    class _Kmonitor:
        @staticmethod
        def report(*_args, **_kwargs):
            return None

    class _Metrics:
        MASTER_ROUTE_ERROR_QPS_METRIC = "master_error"
        MASTER_ROUTE_QPS_METRIC = "master"
        DOMAIN_ROUTE_QPS_METRIC = "domain"
        MASTER_QUEUE_REJECT_QPS_METRIC = "reject"
        MASTER_ROUTE_RT_METRIC = "master_rt"
        DOMAIN_ROUTE_RT_METRIC = "domain_rt"
        ROUTE_RT_METRIC = "route_rt"

    class _Timer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        @staticmethod
        def cost_ms():
            return 0

    _module("torch", inference_mode=lambda: lambda function: function)
    _module(
        "rtp_llm.config.exceptions",
        ExceptionType=ExceptionType,
        FtRuntimeException=FtRuntimeException,
    )
    _module(
        "rtp_llm.config.generate_config",
        RoleAddr=RoleAddr,
        RoleType=RoleType,
    )
    _module("rtp_llm.config.model_config", ModelConfig=object)
    _module("rtp_llm.cpp.model_rpc.model_rpc_client", ModelRpcClient=object)
    sys.modules["rtp_llm.metrics"].kmonitor = _Kmonitor()
    _module(
        "rtp_llm.metrics.kmonitor_metric_reporter",
        AccMetrics=_Metrics,
        GaugeMetrics=_Metrics,
    )
    _module(
        "rtp_llm.ops",
        SpeculativeExecutionConfig=object,
        VitSeparation=types.SimpleNamespace(VIT_SEPARATION_REMOTE="remote"),
        get_block_cache_keys=lambda token_ids, block_size: [len(token_ids), block_size],
    )
    _module(
        "rtp_llm.server.host_service",
        HostService=object,
        HostServiceArgs=object,
    )
    _module(
        "rtp_llm.server.master_client",
        FlexlbResponse=FlexlbResponse,
        MasterClient=object,
    )
    _module("rtp_llm.server.misc", format_exception=lambda error: {"error": str(error)})
    _module(
        "rtp_llm.utils.base_model_datatypes",
        GenerateInput=object,
        GenerateOutputs=object,
    )
    _module("rtp_llm.utils.time_util", Timer=_Timer)

    module_name = "rtp_llm.server.backend_rpc_server_visitor"
    spec = importlib.util.spec_from_file_location(
        module_name,
        rtp_root / "server" / "backend_rpc_server_visitor.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, RoleAddr, RoleType, FlexlbResponse


visitor_module, RoleAddr, RoleType, FlexlbResponse = _load_visitor_module()


class _TokenIds:
    shape = (1, 4)

    @staticmethod
    def size(_dimension):
        return 1

    @staticmethod
    def tolist():
        return [[1, 2, 3, 4]]


class _HostService:
    def __init__(self):
        self.domain_calls = 0

    @staticmethod
    def get_queue_length():
        return 0

    @staticmethod
    def get_master_addr():
        return None

    def get_backend_role_addrs(self, _roles):
        self.domain_calls += 1
        return [RoleAddr(RoleType.PREFILL, "10.0.0.9", 8080, 8001)]


class _MasterClient:
    def __init__(self, enabled):
        self.kvcm_fallback_enabled = enabled
        self.calls = 0

    async def get_backend_role_addrs(self, **_kwargs):
        self.calls += 1
        return FlexlbResponse(
            role_addrs=[RoleAddr(RoleType.PREFILL, "10.0.0.2", 8080, 8001)]
        )


def _visitor(kvcm_enabled):
    visitor = visitor_module.BackendRPCServerVisitor.__new__(
        visitor_module.BackendRPCServerVisitor
    )
    visitor.seq_size_per_block = 4
    visitor.backend_role_list = [RoleType.PREFILL]
    visitor.master_config = types.SimpleNamespace(master_queue_reject_threshold=100)
    visitor.host_service = _HostService()
    visitor.master_client = _MasterClient(kvcm_enabled)
    return visitor


def _input():
    return types.SimpleNamespace(
        request_id="request-1",
        token_ids=_TokenIds(),
        generate_config=types.SimpleNamespace(role_addrs=[]),
    )


class BackendVisitorKvcmEntryTest(unittest.IsolatedAsyncioTestCase):
    async def test_no_master_address_still_enters_master_client_when_kvcm_enabled(self):
        visitor = _visitor(True)
        generate_input = _input()

        await visitor.route_ips(generate_input)

        self.assertEqual(1, visitor.master_client.calls)
        self.assertEqual(0, visitor.host_service.domain_calls)
        self.assertEqual("10.0.0.2", generate_input.generate_config.role_addrs[0].ip)

    async def test_switch_off_preserves_direct_domain_fallback(self):
        visitor = _visitor(False)
        generate_input = _input()

        await visitor.route_ips(generate_input)

        self.assertEqual(0, visitor.master_client.calls)
        self.assertEqual(1, visitor.host_service.domain_calls)
        self.assertEqual("10.0.0.9", generate_input.generate_config.role_addrs[0].ip)


if __name__ == "__main__":
    unittest.main()
