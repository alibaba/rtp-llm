from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parents[1]
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from online_eval.mock_engine import (
    LruBlockCache,
    MockEngineCluster,
    encode_unique_key,
    generate_aggregated_prometheus_metrics,
)
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

        self.channel = grpc.aio.insecure_channel(
            f"{self.prefill.host}:{self.prefill.grpc_port}"
        )
        self.decode_channel = grpc.aio.insecure_channel(
            f"{self.decode.host}:{self.decode.grpc_port}"
        )

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
            role="DECODE",
            role_type=self.pb2.ROLE_TYPE_DECODE,
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


class TestAggregatedMetrics(unittest.TestCase):
    """Unit tests for generate_aggregated_prometheus_metrics."""

    def _make_snapshot(
        self,
        role="prefill",
        stopped=False,
        running=0,
        waiting=0,
        accepted=0,
        completed=0,
        cancelled_count=0,
        cache_keys=0,
        cache_evictions=0,
        active_kv_tokens=0,
        available_kv_tokens=0,
        rpc_counts=None,
        prefill_ms_avg=0.0,
        prefill_ms_p99=0.0,
        prefill_ms_count=0,
        decode_ms_avg=0.0,
        decode_ms_p99=0.0,
        decode_ms_count=0,
    ) -> dict:
        return {
            "role": role,
            "stopped": stopped,
            "running": running,
            "waiting": waiting,
            "accepted": accepted,
            "completed": completed,
            "cancelled_count": cancelled_count,
            "cache_keys": cache_keys,
            "cache_evictions": cache_evictions,
            "active_kv_tokens": active_kv_tokens,
            "available_kv_tokens": available_kv_tokens,
            "rpc_counts": rpc_counts or {},
            "prefill_ms_avg": prefill_ms_avg,
            "prefill_ms_p99": prefill_ms_p99,
            "prefill_ms_count": prefill_ms_count,
            "decode_ms_avg": decode_ms_avg,
            "decode_ms_p99": decode_ms_p99,
            "decode_ms_count": decode_ms_count,
        }

    def _extract_value(
        self, text: str, metric_name: str, label_fragment: str
    ) -> str | None:
        """Extract the numeric value from a Prometheus metric data line.

        Skips comment lines (HELP/TYPE).  Returns the value as a string,
        or None if no matching data line is found.
        """
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if metric_name in line and label_fragment in line:
                return line.rsplit(None, 1)[-1]
        return None

    # ------------------------------------------------------------------
    # 1. Basic structure
    # ------------------------------------------------------------------
    def test_aggregated_basic(self) -> None:
        """Output contains role labels, no engine_name, and HELP/TYPE."""
        engines = [
            self._make_snapshot(role="prefill"),
            self._make_snapshot(role="prefill"),
            self._make_snapshot(role="decode"),
            self._make_snapshot(role="decode"),
        ]
        output = generate_aggregated_prometheus_metrics(engines)

        self.assertIn('role="prefill"', output)
        self.assertIn('role="decode"', output)
        self.assertNotIn("engine_name", output)
        self.assertIn("# HELP", output)
        self.assertIn("# TYPE", output)

    # ------------------------------------------------------------------
    # 2. Summation of counters
    # ------------------------------------------------------------------
    def test_aggregated_sum(self) -> None:
        """Counter metrics are summed across engines of the same role."""
        engines = [
            self._make_snapshot(role="prefill", accepted=100, completed=50),
            self._make_snapshot(role="prefill", accepted=200, completed=150),
        ]
        output = generate_aggregated_prometheus_metrics(engines)

        accepted_val = self._extract_value(
            output, "mock_engine_accepted_total", 'role="prefill"'
        )
        self.assertEqual("300", accepted_val)

        completed_val = self._extract_value(
            output, "mock_engine_completed_total", 'role="prefill"'
        )
        self.assertEqual("200", completed_val)

    # ------------------------------------------------------------------
    # 3. Weighted average for latency
    # ------------------------------------------------------------------
    def test_aggregated_weighted_avg(self) -> None:
        """Latency avg uses weighted average sum(avg*count)/sum(count)."""
        # (100*10 + 200*30) / (10+30) = 7000 / 40 = 175.0
        engines = [
            self._make_snapshot(
                role="prefill", prefill_ms_avg=100.0, prefill_ms_count=10
            ),
            self._make_snapshot(
                role="prefill", prefill_ms_avg=200.0, prefill_ms_count=30
            ),
        ]
        output = generate_aggregated_prometheus_metrics(engines)

        val = self._extract_value(
            output, "mock_engine_prefill_ms_avg", 'role="prefill"'
        )
        self.assertEqual("175.0", val)

    # ------------------------------------------------------------------
    # 4. Max for p99
    # ------------------------------------------------------------------
    def test_aggregated_p99_max(self) -> None:
        """Latency p99 takes the max across engines."""
        engines = [
            self._make_snapshot(role="prefill", prefill_ms_p99=150.0),
            self._make_snapshot(role="prefill", prefill_ms_p99=250.0),
        ]
        output = generate_aggregated_prometheus_metrics(engines)

        val = self._extract_value(
            output, "mock_engine_prefill_ms_p99", 'role="prefill"'
        )
        self.assertEqual("250.0", val)

    # ------------------------------------------------------------------
    # 5. Empty engine list
    # ------------------------------------------------------------------
    def test_aggregated_empty(self) -> None:
        """Empty engine list produces only HELP/TYPE declarations."""
        output = generate_aggregated_prometheus_metrics([])

        self.assertIn("# HELP", output)
        self.assertIn("# TYPE", output)

        # No data lines should be present (only comments)
        for line in output.splitlines():
            if line and not line.startswith("#"):
                self.fail(f"Expected no data lines for empty engine list, got: {line}")

    # ------------------------------------------------------------------
    # 6. No cluster counters
    # ------------------------------------------------------------------
    def test_aggregated_no_cluster_counters(self) -> None:
        """When cluster_counters is None, no cluster-level data lines."""
        engines = [self._make_snapshot(role="prefill", accepted=1)]
        output = generate_aggregated_prometheus_metrics(engines, cluster_counters=None)

        # HELP/TYPE lines always exist for cluster metrics, but no *data* lines
        self.assertIsNone(
            self._extract_value(output, "flexlb_mock_grpc_error_count", "")
        )
        self.assertIsNone(
            self._extract_value(output, "flexlb_mock_grpc_retry_count", "")
        )
        self.assertIsNone(
            self._extract_value(output, "flexlb_mock_grpc_cancel_forward_count", "")
        )

    # ------------------------------------------------------------------
    # 7. With cluster counters
    # ------------------------------------------------------------------
    def test_aggregated_cluster_counters(self) -> None:
        """cluster_counters produce cluster-level metric data lines."""
        engines = [self._make_snapshot(role="prefill")]
        cluster_counters = {
            "grpc_error_count": 42,
            "grpc_retry_count": 10,
            "grpc_cancel_forward_count": 5,
        }
        output = generate_aggregated_prometheus_metrics(engines, cluster_counters)

        self.assertIn("flexlb_mock_grpc_error_count 42", output)
        self.assertIn("flexlb_mock_grpc_retry_count 10", output)
        self.assertIn("flexlb_mock_grpc_cancel_forward_count 5", output)

    # ------------------------------------------------------------------
    # 8. Zero-count latency (no divide-by-zero)
    # ------------------------------------------------------------------
    def test_aggregated_zero_count_latency(self) -> None:
        """Zero latency count produces 0.0 avg without exception."""
        engines = [
            self._make_snapshot(role="prefill", prefill_ms_count=0, decode_ms_count=0),
        ]
        output = generate_aggregated_prometheus_metrics(engines)

        prefill_avg = self._extract_value(
            output, "mock_engine_prefill_ms_avg", 'role="prefill"'
        )
        self.assertEqual("0.0", prefill_avg)

        decode_avg = self._extract_value(
            output, "mock_engine_decode_ms_avg", 'role="prefill"'
        )
        self.assertEqual("0.0", decode_avg)


if __name__ == "__main__":
    unittest.main()
