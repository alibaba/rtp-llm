from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from flexlb_load_client import LoadClient
from online_eval.mock_engine import MockEngineCluster, MockEngineState
from online_eval.rt_model import PerformanceModel


class LoadClientDiscoveryTest(unittest.TestCase):
    def load_endpoints(self, data: dict) -> LoadClient:
        client = LoadClient.__new__(LoadClient)
        client._fallback_prefill_addrs = []
        client._fallback_decode_addrs = []
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "endpoints.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                client._load_fallback_endpoints(str(path))
        return client

    def test_fallback_reads_model_service_hosts_from_mock_discovery(self) -> None:
        cluster = MockEngineCluster(None, None, PerformanceModel({}))
        for role, http_port in [("prefill", 8100), ("decode", 8200), ("prefill", 8300)]:
            cluster.states.append(
                MockEngineState(
                    pb2=None,
                    name=f"{role}-{http_port}",
                    role=role,
                    host="127.0.0.1",
                    grpc_port=http_port + 1,
                    http_port=http_port,
                    performance=cluster.performance,
                    cache_capacity_blocks=4,
                    total_kv_tokens=4096,
                    block_size=1024,
                    cluster=cluster,
                )
            )
        env = cluster.service_discovery_env("mock.prefill", "mock.decode")
        self.assertEqual({"MODEL_SERVICE_CONFIG"}, set(env))

        client = self.load_endpoints(
            {
                "prefill_domain": "mock.prefill",
                "decode_domain": "mock.decode",
                "env": env,
                "engines": [
                    {"role": "prefill", "grpc_addr": "127.0.0.2:9001"},
                    {"role": "decode", "grpc_addr": "127.0.0.2:9101"},
                ],
            }
        )

        self.assertEqual(
            ["127.0.0.1:8101", "127.0.0.1:8301"], client._fallback_prefill_addrs
        )
        self.assertEqual(["127.0.0.1:8201"], client._fallback_decode_addrs)

    def test_fallback_uses_engine_metadata_when_role_hosts_are_absent(self) -> None:
        for env in (
            {},
            {"MODEL_SERVICE_CONFIG": json.dumps({"hosts": {"mock.prefill": []}})},
        ):
            with self.subTest(env=env):
                client = self.load_endpoints(
                    {
                        "prefill_domain": "mock.prefill",
                        "decode_domain": "mock.decode",
                        "env": env,
                        "engines": [
                            {"role": "prefill", "grpc_addr": "127.0.0.1:8101"},
                            {"role": "decode", "grpc_addr": "127.0.0.1:8201"},
                        ],
                    }
                )

                self.assertEqual(["127.0.0.1:8101"], client._fallback_prefill_addrs)
                self.assertEqual(["127.0.0.1:8201"], client._fallback_decode_addrs)


if __name__ == "__main__":
    unittest.main()


class RealEngineStreamRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_master_admitted_real_request_attaches_to_decode_generate_stream(
        self,
    ):
        client = LoadClient.__new__(LoadClient)
        client.args = SimpleNamespace(engine_protocol="rtp", timeout_ms=1000)
        client._role_addr = Mock(
            side_effect=lambda response, role: {
                "PREFILL": "prefill:1",
                "DECODE": "decode:2",
            }.get(role, "")
        )
        client._channel = AsyncMock(return_value="decode-channel")

        async def empty_stream():
            if False:
                yield None

        stub = Mock()
        stub.GenerateStreamCall.return_value = empty_stream()
        client.pb2_grpc = SimpleNamespace(RpcServiceStub=Mock(return_value=stub))
        input_pb = SimpleNamespace(request_id=123)
        await client._read_engine_stream(
            input_pb, SimpleNamespace(enqueued_by_master=True)
        )
        client._channel.assert_awaited_once_with("decode:2")
        stub.GenerateStreamCall.assert_called_once_with(input_pb, timeout=1.0)
        stub.FetchResponse.assert_not_called()

    async def test_auto_protocol_uses_empty_mock_topology_and_caches_result(self):
        client = LoadClient.__new__(LoadClient)
        client._channel = AsyncMock(return_value="prefill-channel")
        stub = SimpleNamespace(
            GetPeerInfo=AsyncMock(return_value=SimpleNamespace(tp_size=0))
        )
        client.pb2 = SimpleNamespace(GetPeerInfoRequestPB=Mock(return_value="probe"))
        client.pb2_grpc = SimpleNamespace(RpcServiceStub=Mock(return_value=stub))
        self.assertTrue(await client._uses_mock_protocol("prefill:1"))
        self.assertTrue(await client._uses_mock_protocol("prefill:1"))
        stub.GetPeerInfo.assert_awaited_once_with("probe", timeout=1.0)

    async def test_failed_protocol_probe_keeps_decode_route_and_can_retry(self):
        import grpc

        client = LoadClient.__new__(LoadClient)
        client._channel = AsyncMock(return_value="prefill-channel")
        stub = SimpleNamespace(
            GetPeerInfo=AsyncMock(
                side_effect=[grpc.RpcError(), SimpleNamespace(tp_size=4)]
            )
        )
        client.pb2 = SimpleNamespace(GetPeerInfoRequestPB=Mock(return_value="probe"))
        client.pb2_grpc = SimpleNamespace(RpcServiceStub=Mock(return_value=stub))
        self.assertFalse(await client._uses_mock_protocol("prefill:1"))
        self.assertFalse(await client._uses_mock_protocol("prefill:1"))
        self.assertEqual(2, stub.GetPeerInfo.await_count)
