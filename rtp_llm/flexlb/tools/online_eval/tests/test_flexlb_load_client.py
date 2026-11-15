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


class LoadClientStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_master_admission_routes_to_real_decode_or_mock_prefill(self):
        async def outputs():
            yield SimpleNamespace(flatten_output=SimpleNamespace(finished=[True]))

        for tp_size in (0, 4):
            with self.subTest(tp_size=tp_size):
                client = LoadClient.__new__(LoadClient)
                client.args = SimpleNamespace(timeout_ms=1000)
                client.pb2 = SimpleNamespace(
                    GetPeerInfoRequestPB=Mock(return_value=object()),
                    FetchRequestPB=Mock(return_value=object()),
                )
                prefill = SimpleNamespace(
                    GetPeerInfo=AsyncMock(
                        return_value=SimpleNamespace(tp_size=tp_size)
                    ),
                    FetchResponse=Mock(side_effect=lambda *args, **kwargs: outputs()),
                )
                decode = SimpleNamespace(
                    GenerateStreamCall=Mock(
                        side_effect=lambda *args, **kwargs: outputs()
                    ),
                )
                stubs = {"prefill:9001": prefill, "decode:9001": decode}
                client.pb2_grpc = SimpleNamespace(RpcServiceStub=stubs.__getitem__)
                client._channel = AsyncMock(side_effect=lambda address: address)
                client._mock_prefill_probes = {}
                response = SimpleNamespace(
                    enqueued_by_master=True,
                    server_status=[
                        SimpleNamespace(
                            role="PREFILL", server_ip="prefill", grpc_port=9001
                        ),
                        SimpleNamespace(
                            role="DECODE", server_ip="decode", grpc_port=9001
                        ),
                    ],
                )
                inputs = [SimpleNamespace(request_id=i) for i in (10, 11)]
                for item in inputs:
                    await client._read_engine_stream(item, response)

                prefill.GetPeerInfo.assert_awaited_once()
                if tp_size:
                    self.assertEqual(decode.GenerateStreamCall.call_count, 2)
                    self.assertTrue(all(item.enqueued_by_master for item in inputs))
                    prefill.FetchResponse.assert_not_called()
                else:
                    self.assertEqual(prefill.FetchResponse.call_count, 2)
                    decode.GenerateStreamCall.assert_not_called()


if __name__ == "__main__":
    unittest.main()
