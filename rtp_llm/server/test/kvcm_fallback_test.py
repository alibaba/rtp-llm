import importlib
import sys
import types
import unittest
from pathlib import Path

import grpc
import asyncio


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
status_pb2 = importlib.import_module(
    "rtp_llm.server.worker_status_proto.worker_status_service_pb2"
)
status_pb2_grpc = importlib.import_module(
    "rtp_llm.server.worker_status_proto.worker_status_service_pb2_grpc"
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


class _FakeWorkerStatusService(status_pb2_grpc.RpcServiceServicer):
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.delay_s = 0.0

    async def GetWorkerStatus(self, request, context):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            return status_pb2.WorkerStatusPB(
                role="RoleType.PREFILL",
                alive=True,
                status_version=self.calls,
            )
        finally:
            self.active -= 1


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
        self.worker_service = _FakeWorkerStatusService()
        self.worker_server = grpc.aio.server()
        status_pb2_grpc.add_RpcServiceServicer_to_server(
            self.worker_service, self.worker_server
        )
        self.worker_port = self.worker_server.add_insecure_port("127.0.0.1:0")
        await self.worker_server.start()

    async def asyncTearDown(self):
        await self.server.stop(None)
        await self.worker_server.stop(None)

    def _client(self):
        return kvcm.KvcmFallbackClient(
            kvcm.KvcmFallbackConfig(
                instance_id="prefill-test_4",
                block_size=4,
                worker_grpc_port_override=8001,
                worker_status_port_override=self.worker_port,
                hot_candidate_pool_size=4,
                candidate_pool_size=4,
            ),
            lambda: [f"127.0.0.1:{self.service.port}"],
        )

    async def test_queries_leader_and_selects_maximum_local_affinity(self):
        self.service.hosts = [
            kvcm_pb2.HostCacheMatch(
                host_ip_port="127.0.0.1:8080",
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
                input_ids=list(range(40)),
            )
            self.assertEqual("selected", result.outcome)
            self.assertIsNotNone(result.selected)
            self.assertEqual("127.0.0.1", result.selected.host_ip)
            self.assertEqual(5, result.selected.local_blocks)
            self.assertEqual(8001, result.selected.grpc_port)
            self.assertEqual(1, result.candidate_count)
            self.assertEqual(10, result.block_count)
            self.assertEqual(
                kvcm.calculate_vllm_block_cache_keys(list(range(40)), 4),
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
            kvcm_pb2.HostCacheMatch(host_ip_port="127.0.0.1:8080", local=0)
        ]
        client = self._client()
        try:
            result = await client.query_and_select(
                request_id="request-2",
                block_cache_keys=[1, 2],
            )
            self.assertEqual("no_candidates", result.outcome)
            self.assertEqual("no_positive_match", result.cache_query_outcome)
            self.assertIsNone(result.selected)
        finally:
            await client.close()

    async def test_worker_status_is_fresh_concurrency_bounded_and_channels_reused(self):
        self.service.hosts = [
            kvcm_pb2.HostCacheMatch(
                host_ip_port="127.0.0.1:8080",
                local=4,
            )
        ]
        self.worker_service.delay_s = 0.01
        client = kvcm.KvcmFallbackClient(
            kvcm.KvcmFallbackConfig(
                instance_id="prefill-test_4",
                block_size=4,
                worker_grpc_port_override=8001,
                worker_status_port_override=self.worker_port,
                hot_candidate_pool_size=4,
                candidate_pool_size=4,
                worker_status_concurrency=2,
            ),
            lambda: [f"127.0.0.1:{self.service.port}"],
        )
        try:
            for request_id in ("request-a", "request-b"):
                result = await client.query_and_select(
                    request_id=request_id,
                    block_cache_keys=[1, 2],
                )
                self.assertEqual("selected", result.outcome)
                self.assertEqual(1, result.status_success_count)
            self.assertEqual(2, self.worker_service.calls)
            self.assertEqual(1, len(client._worker_status_channels))

            extra_servers = []
            extra_candidates = []
            for index in range(4):
                server = grpc.aio.server()
                status_pb2_grpc.add_RpcServiceServicer_to_server(
                    self.worker_service, server
                )
                port = server.add_insecure_port("127.0.0.1:0")
                await server.start()
                extra_servers.append(server)
                extra_candidates.append(
                    kvcm.KvcmCacheCandidate(
                        host_ip="127.0.0.1",
                        http_port=8080 + index,
                        grpc_port=8001 + index,
                        worker_status_port=port,
                        local_blocks=1,
                        p2p_fetch_blocks=0,
                        p2p_total_match_blocks=1,
                    )
                )
            try:
                semaphore = asyncio.Semaphore(2)
                snapshots = await asyncio.gather(
                    *(
                        client._probe_worker(candidate, semaphore)
                        for candidate in extra_candidates
                    )
                )
                self.assertTrue(all(snapshot is not None for snapshot in snapshots))
                self.assertLessEqual(self.worker_service.max_active, 2)
            finally:
                for server in extra_servers:
                    await server.stop(None)
        finally:
            await client.close()


class CandidatePoolAndStrategyTest(unittest.TestCase):
    def _candidate(
        self,
        ip: str,
        local_blocks: int = 0,
        p2p_total_match_blocks: int = 0,
    ):
        return kvcm.KvcmCacheCandidate(
            host_ip=ip,
            http_port=8080,
            grpc_port=8001,
            worker_status_port=18002,
            local_blocks=local_blocks,
            p2p_fetch_blocks=0,
            p2p_total_match_blocks=p2p_total_match_blocks,
        )

    def _config(self, **kwargs):
        return kvcm.KvcmFallbackConfig(
            instance_id="test_4",
            block_size=4,
            **kwargs,
        )

    def _snapshot(self, candidate, outstanding=0, waiting=0, alive=True):
        return kvcm.WorkerLoadSnapshot(
            candidate=candidate,
            role="RoleType.PREFILL",
            alive=alive,
            waiting_task_count=waiting,
            outstanding_uncached_tokens=outstanding,
            status_version=1,
        )

    def test_pool_keeps_local_and_merges_its_real_cache_match(self):
        local = self._candidate("10.0.0.1")
        hits = [
            self._candidate("10.0.0.1", 7),
            self._candidate("10.0.0.2", 9),
            self._candidate("10.0.0.3", 8),
        ]
        pool = kvcm.build_candidate_pool(
            hits,
            local,
            self._config(candidate_pool_size=2, hot_candidate_pool_size=2),
        )
        self.assertEqual(["10.0.0.1", "10.0.0.2"], [item.host_ip for item in pool])
        self.assertEqual(7, pool[0].local_blocks)

    def test_worker_snapshot_uses_remaining_running_tokens(self):
        candidate = self._candidate("10.0.0.1")
        status = status_pb2.WorkerStatusPB(
            role="RoleType.PREFILL",
            alive=True,
            running_task_info=[
                status_pb2.TaskInfoPB(
                    request_id="waiting",
                    input_length=100,
                    prefix_length=40,
                    is_waiting=True,
                ),
                status_pb2.TaskInfoPB(
                    request_id="running",
                    input_length=200,
                    prefix_length=20,
                    is_waiting=False,
                    remaining_prefill_tokens=30,
                ),
            ],
        )
        snapshot = kvcm.worker_load_snapshot(candidate, status)
        self.assertEqual(1, snapshot.waiting_task_count)
        self.assertEqual(90, snapshot.outstanding_uncached_tokens)

    def test_cache_leader_respects_extra_work_bound(self):
        hot = self._candidate("10.0.0.1", 20)
        cold = self._candidate("10.0.0.2", 0)
        snapshots = [
            self._snapshot(hot, outstanding=200),
            self._snapshot(cold, outstanding=0),
        ]
        reject = kvcm.select_cache_affinity_first(
            snapshots,
            seq_len=100,
            config=self._config(
                cache_affinity_first_max_extra_work_tokens=50,
                cache_affinity_first_min_hit_rate=5,
            ),
        )
        self.assertEqual("10.0.0.2", reject.selected.candidate.host_ip)
        self.assertEqual("SHORTEST_TTFT", reject.reason)
        accept = kvcm.select_cache_affinity_first(
            snapshots,
            seq_len=100,
            config=self._config(
                cache_affinity_first_max_extra_work_tokens=200,
                cache_affinity_first_min_hit_rate=5,
            ),
        )
        self.assertEqual("10.0.0.1", accept.selected.candidate.host_ip)
        self.assertEqual("CACHE_LEADER", accept.reason)

    def test_low_hit_and_outstanding_guard_match_flexlb_fallbacks(self):
        hot = self._candidate("10.0.0.1", 1)
        cold = self._candidate("10.0.0.2", 0)
        snapshots = [self._snapshot(hot, 80), self._snapshot(cold, 0)]
        low_hit = kvcm.select_cache_affinity_first(
            snapshots,
            seq_len=100,
            config=self._config(cache_affinity_first_min_hit_rate=10),
        )
        self.assertEqual("SHORTEST_TTFT_LOW_CACHE_HIT", low_hit.reason)
        all_guarded = kvcm.select_cache_affinity_first(
            snapshots,
            seq_len=100,
            config=self._config(outstanding_uncached_tokens_threshold=10),
        )
        self.assertEqual("SHORTEST_TTFT_OUTSTANDING_GUARD_FALLBACK", all_guarded.reason)


if __name__ == "__main__":
    unittest.main()
