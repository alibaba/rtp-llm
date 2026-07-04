from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from online_eval.mock_engine import LruBlockCache, MockEngineCluster, encode_unique_key
from online_eval.proto_utils import ensure_proto_modules
from online_eval.rt_model import PerformanceModel


class LruBlockCacheTest(unittest.TestCase):
    def test_lru_admit_and_prefix_hits(self) -> None:
        cache = LruBlockCache(capacity_blocks=2, block_size=1024)

        self.assertTrue(cache.admit([1, 2]))
        self.assertEqual([1, 2], cache.keys)
        self.assertEqual(2, cache.prefix_hit_blocks([1, 2, 3]))
        self.assertEqual([1, 2], cache.keys)

        self.assertTrue(cache.admit([3]))
        self.assertEqual([2, 3], cache.keys)
        self.assertEqual(1, cache.evictions)
        self.assertEqual(0, cache.prefix_hit_blocks([1, 2, 3]))


class MockEngineGrpcTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        perf = PerformanceModel(
            {
                "sleep_scale": 0.0,
                "prefill": {"fixed_ms": 1.0},
                "decode": {"step_ms_by_batch": [[1, 1.0]], "scale": 1.0},
            }
        )
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, perf)
        self.prefill = await self.cluster.add_engine(
            name="prefill-0",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        self.decode = await self.cluster.add_engine(
            name="decode-0",
            role="decode",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        import grpc

        self.channel = grpc.aio.insecure_channel(self.prefill.ip_port)
        self.decode_channel = grpc.aio.insecure_channel(self.decode.ip_port)

    async def asyncTearDown(self) -> None:
        await self.channel.close()
        await self.decode_channel.close()
        await self.cluster.stop()

    async def test_enqueue_fetch_status_and_cache(self) -> None:
        stub = self.pb2_grpc.RpcServiceStub(self.channel)
        decode_stub = self.pb2_grpc.RpcServiceStub(self.decode_channel)

        input_pb = self._input_pb(request_id=123)
        batch = self.pb2.EnqueueBatchRequestPB(batch_id=7)
        slot = batch.dp_slots.add(dp_rank=0)
        slot.requests.add().input.CopyFrom(input_pb)

        response = await stub.EnqueueBatch(batch)
        self.assertEqual(7, response.batch_id)
        self.assertEqual([123], [item.request_id for item in response.successes])

        finished = []
        async for output in stub.FetchResponse(self.pb2.FetchRequestPB(request_id=123)):
            finished.extend(bool(x) for x in output.flatten_output.finished)
        self.assertEqual([False, True], finished)

        status = await decode_stub.GetWorkerStatus(
            self.pb2.StatusVersionPB(latest_finished_version=-1)
        )
        self.assertEqual(1, len(status.finished_task_list))
        self.assertEqual(123, status.finished_task_list[0].request_id)

        cache = await decode_stub.GetCacheStatus(
            self.pb2.CacheVersionPB(need_cache_keys=True)
        )
        self.assertIn(11, cache.cache_keys)
        self.assertIn(12, cache.cache_keys)

    def _input_pb(self, request_id: int):
        config = self.pb2.GenerateConfigPB(
            max_new_tokens=2,
            num_return_sequences=1,
            return_incremental=True,
            is_streaming=True,
            timeout_ms=1000,
            unique_key=encode_unique_key(
                {
                    "rid": str(request_id),
                    "input_len": 2048,
                    "output_len": 2,
                    "block_cache_keys": [11, 12],
                }
            ),
        )
        config.role_addrs.add(
            role=self.pb2.ROLE_TYPE_DECODE,
            ip=self.decode.host,
            http_port=self.decode.http_port,
            grpc_port=self.decode.grpc_port,
        )
        return self.pb2.GenerateInputPB(
            request_id=request_id,
            token_ids=[0] * 2048,
            generate_config=config,
            client_id="mock_engine_test",
            start_time=0,
        )


if __name__ == "__main__":
    unittest.main()
