from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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
            cluster.states.append(MockEngineState(
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
            ))
        env = cluster.service_discovery_env("mock.prefill", "mock.decode")
        self.assertEqual({"MODEL_SERVICE_CONFIG"}, set(env))

        client = self.load_endpoints({
            "prefill_domain": "mock.prefill",
            "decode_domain": "mock.decode",
            "env": env,
            "engines": [
                {"role": "prefill", "grpc_addr": "127.0.0.2:9001"},
                {"role": "decode", "grpc_addr": "127.0.0.2:9101"},
            ],
        })

        self.assertEqual(["127.0.0.1:8101", "127.0.0.1:8301"], client._fallback_prefill_addrs)
        self.assertEqual(["127.0.0.1:8201"], client._fallback_decode_addrs)

    def test_fallback_uses_engine_metadata_when_role_hosts_are_absent(self) -> None:
        for env in ({}, {"MODEL_SERVICE_CONFIG": json.dumps({"hosts": {"mock.prefill": []}})}):
            with self.subTest(env=env):
                client = self.load_endpoints({
                    "prefill_domain": "mock.prefill",
                    "decode_domain": "mock.decode",
                    "env": env,
                    "engines": [
                        {"role": "prefill", "grpc_addr": "127.0.0.1:8101"},
                        {"role": "decode", "grpc_addr": "127.0.0.1:8201"},
                    ],
                })

                self.assertEqual(["127.0.0.1:8101"], client._fallback_prefill_addrs)
                self.assertEqual(["127.0.0.1:8201"], client._fallback_decode_addrs)


if __name__ == "__main__":
    unittest.main()
