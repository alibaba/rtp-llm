"""Unit tests for cross-shard resolve_decode routing in mock_engine.

Test scenarios:
  1. test_resolve_decode_local_hit         – local decode engine found
  2. test_resolve_decode_remote_fallback    – decode addr not local → None
  3. test_cross_cluster_routing             – end-to-end gRPC forwarding
  4. test_grpc_channel_caching              – channel reuse verification
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

from online_eval.mock_engine import SENTINEL, MockEngineCluster, encode_unique_key
from online_eval.proto_utils import ensure_proto_modules
from online_eval.rt_model import PerformanceModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perf() -> PerformanceModel:
    return PerformanceModel(
        {
            "sleep_scale": 0.0,
            "prefill": {"fixed_ms": 1.0},
            "decode": {"step_ms_by_batch": [[1, 1.0]], "scale": 1.0},
        }
    )


def _make_input(
    pb2,
    request_id: int,
    decode_ip: Optional[str] = None,
    decode_grpc_port: Optional[int] = None,
    decode_http_port: int = 0,
):
    config = pb2.GenerateConfigPB(
        max_new_tokens=2,
        num_return_sequences=1,
        return_incremental=True,
        is_streaming=True,
        timeout_ms=5000,
        unique_key=encode_unique_key(
            {
                "rid": str(request_id),
                "input_len": 1024,
                "output_len": 2,
                "block_cache_keys": [11, 12],
            }
        ),
    )
    if decode_ip is not None and decode_grpc_port is not None:
        config.role_addrs.add(
            role="DECODE",
            role_type=pb2.ROLE_TYPE_DECODE,
            ip=decode_ip,
            http_port=decode_http_port,
            grpc_port=decode_grpc_port,
        )
    return pb2.GenerateInputPB(
        request_id=request_id,
        token_ids=[0] * 1024,
        generate_config=config,
        client_id="test_resolve_decode",
        start_time=0,
    )


# ---------------------------------------------------------------------------
# Test 1: resolve_decode returns local engine when address matches
# ---------------------------------------------------------------------------


class TestResolveDecodeLocalHit(unittest.IsolatedAsyncioTestCase):
    """role_addrs points to a local decode engine -> resolve_decode returns it."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
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

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_resolve_decode_local_hit(self) -> None:
        input_pb = _make_input(
            self.pb2,
            request_id=100,
            decode_ip=self.decode.host,
            decode_grpc_port=self.decode.grpc_port,
            decode_http_port=self.decode.http_port,
        )
        result = self.cluster.resolve_decode(input_pb)
        self.assertIsNotNone(result)
        self.assertEqual(result.name, "decode-0")


# ---------------------------------------------------------------------------
# Test 2: resolve_decode returns None when decode addr is not local
# ---------------------------------------------------------------------------


class TestResolveDecodeRemoteFallback(unittest.IsolatedAsyncioTestCase):
    """role_addrs points to a non-local decode engine -> resolve_decode returns None."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        # Cluster A: prefill only (no decode engines in this process)
        self.cluster_a = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill_a = await self.cluster_a.add_engine(
            name="prefill-a",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        # Cluster B: has the decode engine (different "shard"/process)
        self.cluster_b = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.decode_b = await self.cluster_b.add_engine(
            name="decode-b",
            role="decode",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )

    async def asyncTearDown(self) -> None:
        await self.cluster_a.stop()
        await self.cluster_b.stop()

    async def test_resolve_decode_returns_none_for_remote(self) -> None:
        """resolve_decode must return None when the decode addr is not in this process."""
        input_pb = _make_input(
            self.pb2,
            request_id=200,
            decode_ip=self.decode_b.host,
            decode_grpc_port=self.decode_b.grpc_port,
            decode_http_port=self.decode_b.http_port,
        )
        result = self.cluster_a.resolve_decode(input_pb)
        self.assertIsNone(result)

    async def test_get_remote_decode_addr_returns_correct_addr(self) -> None:
        """_get_remote_decode_addr must extract the gRPC address from role_addrs."""
        input_pb = _make_input(
            self.pb2,
            request_id=201,
            decode_ip=self.decode_b.host,
            decode_grpc_port=self.decode_b.grpc_port,
            decode_http_port=self.decode_b.http_port,
        )
        addr = self.prefill_a._get_remote_decode_addr(input_pb)
        self.assertIsNotNone(addr)
        expected = f"{self.decode_b.host}:{self.decode_b.grpc_port}"
        self.assertEqual(addr, expected)

    async def test_get_remote_decode_addr_returns_none_without_role_addrs(self) -> None:
        """_get_remote_decode_addr must return None when no DECODE role_addr exists."""
        input_pb = _make_input(self.pb2, request_id=202)
        addr = self.prefill_a._get_remote_decode_addr(input_pb)
        self.assertIsNone(addr)


# ---------------------------------------------------------------------------
# Test 3: end-to-end cross-cluster routing via gRPC
# ---------------------------------------------------------------------------


class TestRunPrefillBatchRoutesToCorrectDecode(unittest.IsolatedAsyncioTestCase):
    """Prefill in cluster A, decode in cluster B — verify gRPC forwarding."""

    async def asyncSetUp(self) -> None:
        import grpc

        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        # Cluster A: prefill + a LOCAL decode (to prove it is NOT used)
        self.cluster_a = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill_a = await self.cluster_a.add_engine(
            name="prefill-a",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        self.decode_local_a = await self.cluster_a.add_engine(
            name="decode-local-a",
            role="decode",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        # Cluster B: the REMOTE decode engine that should receive the request
        self.cluster_b = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.decode_remote_b = await self.cluster_b.add_engine(
            name="decode-remote-b",
            role="decode",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        # gRPC channel to prefill-a (use grpc_port, not http_port)
        self.channel = grpc.aio.insecure_channel(
            f"{self.prefill_a.host}:{self.prefill_a.grpc_port}"
        )

    async def asyncTearDown(self) -> None:
        await self.channel.close()
        await self.cluster_a.stop()
        await self.cluster_b.stop()

    async def test_cross_cluster_routing(self) -> None:
        """Decode request must reach the remote engine, not the local one."""
        request_id = 300
        input_pb = _make_input(
            self.pb2,
            request_id=request_id,
            decode_ip=self.decode_remote_b.host,
            decode_grpc_port=self.decode_remote_b.grpc_port,
            decode_http_port=self.decode_remote_b.http_port,
        )

        # Enqueue batch on cluster A's prefill engine
        batch = self.pb2.EnqueueBatchRequestPB(batch_id=1)
        slot = batch.dp_slots.add(dp_rank=0)
        slot.requests.add().input.CopyFrom(input_pb)

        stub = self.pb2_grpc.RpcServiceStub(self.channel)
        response = await stub.EnqueueBatch(batch, timeout=5.0)
        self.assertEqual(1, response.batch_id)
        self.assertIn(request_id, [int(s.request_id) for s in response.successes])

        # Consume the response stream (blocks until decode completes)
        finished_flags: list[bool] = []
        async for output in stub.FetchResponse(
            self.pb2.FetchRequestPB(request_id=request_id), timeout=10.0
        ):
            finished_flags.extend(bool(x) for x in output.flatten_output.finished)
        self.assertEqual([False, True], finished_flags)

        # Allow fire-and-forget tasks to settle
        await asyncio.sleep(0.05)

        # The REMOTE decode engine must have processed the request
        self.assertEqual(1, self.decode_remote_b._accepted)
        self.assertEqual(1, self.decode_remote_b._completed)

        # The LOCAL decode engine must NOT have been used
        self.assertEqual(0, self.decode_local_a._accepted)
        self.assertEqual(0, self.decode_local_a._completed)

    async def test_cross_cluster_channel_cached(self) -> None:
        """After a cross-cluster request, the gRPC channel must be cached."""
        request_id = 310
        input_pb = _make_input(
            self.pb2,
            request_id=request_id,
            decode_ip=self.decode_remote_b.host,
            decode_grpc_port=self.decode_remote_b.grpc_port,
            decode_http_port=self.decode_remote_b.http_port,
        )
        batch = self.pb2.EnqueueBatchRequestPB(batch_id=2)
        slot = batch.dp_slots.add(dp_rank=0)
        slot.requests.add().input.CopyFrom(input_pb)

        stub = self.pb2_grpc.RpcServiceStub(self.channel)
        await stub.EnqueueBatch(batch, timeout=5.0)

        # Consume responses
        async for _ in stub.FetchResponse(
            self.pb2.FetchRequestPB(request_id=request_id), timeout=10.0
        ):
            pass

        await asyncio.sleep(0.05)

        # A channel to the remote decode engine must be cached
        expected_target = (
            f"{self.decode_remote_b.host}:{self.decode_remote_b.grpc_port}"
        )
        self.assertIn(expected_target, self.cluster_a._grpc_channels)


# ---------------------------------------------------------------------------
# Test 4: gRPC channel caching
# ---------------------------------------------------------------------------


class TestGrpcChannelCaching(unittest.IsolatedAsyncioTestCase):
    """Verify that _get_grpc_channel caches and reuses channels."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_grpc_channel_caching(self) -> None:
        target = "127.0.0.1:19999"
        ch1 = await self.cluster._get_grpc_channel(target)
        ch2 = await self.cluster._get_grpc_channel(target)
        self.assertIs(ch1, ch2)
        self.assertEqual(1, len(self.cluster._grpc_channels))

        other = "127.0.0.1:18888"
        ch3 = await self.cluster._get_grpc_channel(other)
        self.assertIsNot(ch1, ch3)
        self.assertEqual(2, len(self.cluster._grpc_channels))
        self.assertIn(target, self.cluster._grpc_channels)
        self.assertIn(other, self.cluster._grpc_channels)


# ---------------------------------------------------------------------------
# Helper: async iterator from a list
# ---------------------------------------------------------------------------


async def _async_iter(items):
    """Create an async iterator from a list of items."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Test 5: _run_remote_decode cancel forwarding
# ---------------------------------------------------------------------------


class TestRunRemoteDecodeCancelForward(unittest.IsolatedAsyncioTestCase):
    """Verify cancel signal is forwarded to the remote engine during _run_remote_decode."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill = await self.cluster.add_engine(
            name="prefill-0",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_cancel_forwarded_to_remote(self) -> None:
        """When request is cancelled, stub.Cancel must be called and counter incremented."""
        request_id = 500
        input_pb = _make_input(
            self.pb2,
            request_id=request_id,
            decode_ip="10.0.0.99",
            decode_grpc_port=9999,
            decode_http_port=9998,
        )

        # A mock output that the remote stream would yield
        mock_output = self.pb2.GenerateOutputsPB(request_id=request_id)

        # Build mock stub
        mock_stub = MagicMock()
        mock_stub.GenerateStreamCall = MagicMock(
            return_value=_async_iter([mock_output])
        )
        mock_stub.Cancel = AsyncMock()

        # Inject mocks into cluster
        self.cluster._get_grpc_channel = AsyncMock(return_value=MagicMock())
        self.cluster.pb2_grpc = MagicMock()
        self.cluster.pb2_grpc.RpcServiceStub = MagicMock(return_value=mock_stub)

        # Mark the request as cancelled BEFORE _run_remote_decode runs
        self.prefill._cancelled.add(request_id)

        queue = asyncio.Queue()
        await self.prefill._run_remote_decode(input_pb, queue)

        # stub.Cancel must have been called exactly once
        mock_stub.Cancel.assert_called_once()
        # Cluster counter must reflect the forward
        self.assertEqual(1, self.cluster._grpc_cancel_forward_count)

        # SENTINEL must be in queue; the output must NOT be (break before queue.put)
        items = []
        while not queue.empty():
            items.append(await queue.get())
        self.assertEqual([SENTINEL], items)


# ---------------------------------------------------------------------------
# Test 6: gRPC retry — first attempt fails, second succeeds
# ---------------------------------------------------------------------------


class TestRunRemoteDecodeRetrySuccess(unittest.IsolatedAsyncioTestCase):
    """Verify gRPC retry: first attempt fails, second succeeds."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill = await self.cluster.add_engine(
            name="prefill-0",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_grpc_retry_then_success(self) -> None:
        """First GenerateStreamCall fails, second succeeds — verify retry counters and output."""
        request_id = 600
        input_pb = _make_input(
            self.pb2,
            request_id=request_id,
            decode_ip="10.0.0.99",
            decode_grpc_port=9999,
            decode_http_port=9998,
        )

        mock_output = self.pb2.GenerateOutputsPB(request_id=request_id)

        call_count = [0]

        def gen_stream_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated gRPC failure")
            return _async_iter([mock_output])

        mock_stub = MagicMock()
        mock_stub.GenerateStreamCall = MagicMock(side_effect=gen_stream_side_effect)
        mock_stub.Cancel = AsyncMock()

        self.cluster._get_grpc_channel = AsyncMock(return_value=MagicMock())
        self.cluster.pb2_grpc = MagicMock()
        self.cluster.pb2_grpc.RpcServiceStub = MagicMock(return_value=mock_stub)

        queue = asyncio.Queue()
        await self.prefill._run_remote_decode(input_pb, queue)

        # Counters
        self.assertEqual(1, self.cluster._grpc_error_count)
        self.assertEqual(1, self.cluster._grpc_retry_count)

        # Output + SENTINEL in queue
        items = []
        while not queue.empty():
            items.append(await queue.get())
        self.assertEqual(2, len(items))
        self.assertEqual(mock_output, items[0])
        self.assertIs(SENTINEL, items[1])


# ---------------------------------------------------------------------------
# Test 7: gRPC exceeds MAX_RETRIES — all attempts fail
# ---------------------------------------------------------------------------


class TestRunRemoteDecodeExceedsRetries(unittest.IsolatedAsyncioTestCase):
    """Verify that after MAX_RETRIES, no more retries are attempted."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill = await self.cluster.add_engine(
            name="prefill-0",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_grpc_exceeds_max_retries(self) -> None:
        """All attempts fail — verify exactly 2 calls (1 original + 1 retry) and SENTINEL."""
        request_id = 700
        input_pb = _make_input(
            self.pb2,
            request_id=request_id,
            decode_ip="10.0.0.99",
            decode_grpc_port=9999,
            decode_http_port=9998,
        )

        mock_stub = MagicMock()
        mock_stub.GenerateStreamCall = MagicMock(
            side_effect=RuntimeError("simulated gRPC failure")
        )
        mock_stub.Cancel = AsyncMock()

        self.cluster._get_grpc_channel = AsyncMock(return_value=MagicMock())
        self.cluster.pb2_grpc = MagicMock()
        self.cluster.pb2_grpc.RpcServiceStub = MagicMock(return_value=mock_stub)

        queue = asyncio.Queue()
        await self.prefill._run_remote_decode(input_pb, queue)

        # 2 attempts total (1 original + 1 retry)
        self.assertEqual(2, mock_stub.GenerateStreamCall.call_count)
        self.assertEqual(2, self.cluster._grpc_error_count)
        self.assertEqual(1, self.cluster._grpc_retry_count)

        # Only SENTINEL in queue
        items = []
        while not queue.empty():
            items.append(await queue.get())
        self.assertEqual([SENTINEL], items)


# ---------------------------------------------------------------------------
# Test 8: cache consistency in cross-shard routing
# ---------------------------------------------------------------------------


class TestCrossShardCacheConsistency(unittest.IsolatedAsyncioTestCase):
    """Verify cache state independence when prefill and decode are in different clusters."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        # Cluster A: prefill only
        self.cluster_a = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.prefill_a = await self.cluster_a.add_engine(
            name="prefill-a",
            role="prefill",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )
        # Cluster B: decode only (different "shard"/process)
        self.cluster_b = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)
        self.decode_b = await self.cluster_b.add_engine(
            name="decode-b",
            role="decode",
            host="127.0.0.1",
            port=0,
            cache_capacity_blocks=4,
            total_kv_tokens=4096,
            block_size=1024,
        )

    async def asyncTearDown(self) -> None:
        await self.cluster_a.stop()
        await self.cluster_b.stop()

    async def test_get_remote_decode_addr_returns_correct_addr(self) -> None:
        """_get_remote_decode_addr must extract the gRPC address from role_addrs."""
        input_pb = _make_input(
            self.pb2,
            request_id=800,
            decode_ip=self.decode_b.host,
            decode_grpc_port=self.decode_b.grpc_port,
            decode_http_port=self.decode_b.http_port,
        )
        addr = self.prefill_a._get_remote_decode_addr(input_pb)
        self.assertIsNotNone(addr)
        expected = f"{self.decode_b.host}:{self.decode_b.grpc_port}"
        self.assertEqual(addr, expected)

    async def test_resolve_decode_returns_none_for_remote(self) -> None:
        """resolve_decode must return None (not round-robin) for non-local decode addr."""
        input_pb = _make_input(
            self.pb2,
            request_id=801,
            decode_ip=self.decode_b.host,
            decode_grpc_port=self.decode_b.grpc_port,
            decode_http_port=self.decode_b.http_port,
        )
        result = self.cluster_a.resolve_decode(input_pb)
        self.assertIsNone(result)

    async def test_cache_independence_across_shards(self) -> None:
        """Prefill cache and decode cache must be independent across shards."""
        # Admit blocks to prefill engine
        self.prefill_a.cache.admit([100, 200])
        self.assertEqual(2, len(self.prefill_a.cache.keys))

        # Decode engine cache must be unaffected
        self.assertEqual(0, len(self.decode_b.cache.keys))

        # Admit different blocks to decode engine
        self.decode_b.cache.admit([300, 400])
        self.assertEqual(2, len(self.decode_b.cache.keys))

        # Each cache must contain only its own blocks
        self.assertEqual({100, 200}, set(self.prefill_a.cache.keys))
        self.assertEqual({300, 400}, set(self.decode_b.cache.keys))


# ---------------------------------------------------------------------------
# Test 9: Prometheus metrics output for cluster-level counters
# ---------------------------------------------------------------------------


class TestPrometheusClusterMetrics(unittest.IsolatedAsyncioTestCase):
    """Verify generate_prometheus_metrics outputs cluster-level gRPC counters."""

    async def asyncSetUp(self) -> None:
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.perf = _make_perf()
        self.cluster = MockEngineCluster(self.pb2, self.pb2_grpc, self.perf)

    async def asyncTearDown(self) -> None:
        await self.cluster.stop()

    async def test_cluster_metrics_output(self) -> None:
        """generate_prometheus_metrics must include cluster-level gRPC counters with values."""
        self.cluster._grpc_error_count = 5
        self.cluster._grpc_retry_count = 3
        self.cluster._grpc_cancel_forward_count = 2

        metrics = await self.cluster.generate_prometheus_metrics()

        # Cluster-level metric lines with correct values
        self.assertIn("flexlb_mock_grpc_error_count 5", metrics)
        self.assertIn("flexlb_mock_grpc_retry_count 3", metrics)
        self.assertIn("flexlb_mock_grpc_cancel_forward_count 2", metrics)

        # HELP and TYPE declarations
        self.assertIn("# HELP flexlb_mock_grpc_error_count", metrics)
        self.assertIn("# TYPE flexlb_mock_grpc_error_count counter", metrics)
        self.assertIn("# HELP flexlb_mock_grpc_retry_count", metrics)
        self.assertIn("# TYPE flexlb_mock_grpc_retry_count counter", metrics)
        self.assertIn("# HELP flexlb_mock_grpc_cancel_forward_count", metrics)
        self.assertIn("# TYPE flexlb_mock_grpc_cancel_forward_count counter", metrics)


if __name__ == "__main__":
    unittest.main()
