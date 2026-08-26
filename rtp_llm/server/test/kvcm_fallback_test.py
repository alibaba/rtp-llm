import importlib
import sys
import types
import unittest
from pathlib import Path

import grpc


def _install_lightweight_rtp_namespace() -> None:
    rtp_root = Path(__file__).resolve().parents[2]
    if "rtp_llm" not in sys.modules:
        package = types.ModuleType("rtp_llm")
        package.__path__ = [str(rtp_root)]
        sys.modules["rtp_llm"] = package
    if "rtp_llm.server" not in sys.modules:
        package = types.ModuleType("rtp_llm.server")
        package.__path__ = [str(rtp_root / "server")]
        sys.modules["rtp_llm.server"] = package


_install_lightweight_rtp_namespace()
kvcm = importlib.import_module("rtp_llm.server.kvcm_fallback")
kvcm_pb2 = importlib.import_module("rtp_llm.server.kvcm_proto.kvcm_meta_service_pb2")
kvcm_pb2_grpc = importlib.import_module(
    "rtp_llm.server.kvcm_proto.kvcm_meta_service_pb2_grpc"
)


class _FakeMetaService(kvcm_pb2_grpc.MetaServiceServicer):
    def __init__(self) -> None:
        self.port = 0
        self.hosts = []
        self.last_cache_request = None
        self.cluster_query_count = 0

    async def GetClusterInfo(self, request, context):
        self.cluster_query_count += 1
        return kvcm_pb2.GetClusterInfoResponse(
            header=kvcm_pb2.CommonResponseHeader(
                status=kvcm_pb2.Status(code=kvcm_pb2.OK)
            ),
            leader_endpoint=kvcm_pb2.MetaNodeEndpoint(
                host="127.0.0.1",
                meta_rpc_port=self.port,
            ),
        )

    async def GetHostCacheState(self, request, context):
        self.last_cache_request = request
        return kvcm_pb2.GetHostCacheStateResponse(
            header=kvcm_pb2.CommonResponseHeader(
                status=kvcm_pb2.Status(code=kvcm_pb2.OK)
            ),
            hosts=self.hosts,
        )


class VllmBlockHashTest(unittest.TestCase):
    def test_matches_flexlb_and_vllm_golden_vectors(self):
        self.assertEqual(
            [2164874634404590027],
            kvcm.calculate_vllm_block_cache_keys([1, 2, 3, 4], 4),
        )
        self.assertEqual(
            [-7527834946346035334, -7860823284622341314],
            kvcm.calculate_vllm_block_cache_keys(list(range(128)), 64),
        )
        self.assertEqual(
            [2771287707320467766, -4525836348354197114],
            kvcm.calculate_vllm_block_cache_keys(list(range(1, 10)), 4, 1),
        )

    def test_drops_partial_final_block(self):
        self.assertEqual(
            kvcm.calculate_vllm_block_cache_keys(list(range(128)), 64),
            kvcm.calculate_vllm_block_cache_keys(list(range(130)), 64),
        )


class KvcmFallbackClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.service = _FakeMetaService()
        self.server = grpc.aio.server()
        kvcm_pb2_grpc.add_MetaServiceServicer_to_server(self.service, self.server)
        self.service.port = self.server.add_insecure_port("127.0.0.1:0")
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop(None)

    def _client(self):
        return kvcm.KvcmFallbackClient(
            kvcm.KvcmFallbackConfig(
                instance_id="prefill-test_4",
                block_size=4,
                worker_grpc_port_override=8001,
            ),
            lambda: [f"127.0.0.1:{self.service.port}"],
        )

    async def test_queries_leader_and_selects_maximum_local_affinity(self):
        self.service.hosts = [
            kvcm_pb2.HostCacheMatch(
                host_ip_port="10.0.0.1:8080",
                local=2,
                p2p_1_total_match=8,
            ),
            kvcm_pb2.HostCacheMatch(
                host_ip_port="10.0.0.2:8080",
                local=5,
                p2p_1_total_match=5,
            ),
            kvcm_pb2.HostCacheMatch(host_ip_port="not-an-endpoint", local=99),
        ]
        client = self._client()
        try:
            result = await client.query_and_select(
                request_id="request-1",
                block_cache_keys=[],
                input_ids=list(range(12)),
            )
            self.assertEqual("selected", result.outcome)
            self.assertIsNotNone(result.selected)
            self.assertEqual("10.0.0.2", result.selected.host_ip)
            self.assertEqual(5, result.selected.local_blocks)
            self.assertEqual(8001, result.selected.grpc_port)
            self.assertEqual(2, result.candidate_count)
            self.assertEqual(3, result.block_count)
            self.assertEqual(
                kvcm.calculate_vllm_block_cache_keys(list(range(12)), 4),
                list(self.service.last_cache_request.block_cache_keys),
            )
            self.assertEqual(
                "prefill-test_4", self.service.last_cache_request.instance_id
            )
            self.assertEqual(
                kvcm_pb2.QT_PREFIX_MATCH, self.service.last_cache_request.query_type
            )
            # Bootstrap and leader are the same target, so one channel is reused.
            self.assertEqual(1, len(client._channels))
        finally:
            await client.close()

    async def test_no_positive_match_returns_no_route(self):
        self.service.hosts = [
            kvcm_pb2.HostCacheMatch(host_ip_port="10.0.0.1:8080", local=0)
        ]
        client = self._client()
        try:
            result = await client.query_and_select(
                request_id="request-2",
                block_cache_keys=[1, 2],
            )
            self.assertEqual("no_positive_match", result.outcome)
            self.assertIsNone(result.selected)
        finally:
            await client.close()


if __name__ == "__main__":
    unittest.main()
