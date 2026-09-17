import concurrent.futures
import threading
import unittest
from unittest import mock

import torch
import torch.nn as nn

from rtp_llm.multimodal.mm_embedding_cache import (
    MMEmbeddingCache,
    MMEmbeddingCacheEntry,
    _copy_result_to_devices,
    _TensorMemoryPool,
)
from rtp_llm.multimodal.mm_process_engine import _feature_hashes_from_result
from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import minimax_m3_vl_mixin as m3
from rtp_llm.multimodal.multimodal_util import build_multimodal_output_pb
from rtp_llm.utils.cuda_graph_gate import CudaGraphGate, cuda_graph_gate


class CudaGraphGateTest(unittest.TestCase):
    def test_pool_waits_for_capture_before_taking_its_mutex(self):
        pool = _TensorMemoryPool(64, torch.device("cpu"))
        block = pool.reserve(8)
        attempted, finished = threading.Event(), threading.Event()
        event = mock.Mock()
        event.query.return_value = True

        def release():
            attempted.set()
            try:
                pool.release(block, [event])
            finally:
                finished.set()

        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            with cuda_graph_gate.capture():
                future = executor.submit(release)
                self.assertTrue(attempted.wait(5))
                self.assertFalse(finished.wait(0.05))
                # torch.cuda.graph.__enter__ runs GC. A pool owner finalized
                # there must not encounter a mutex held by a blocked reader.
                acquired = pool._lock.acquire(timeout=0.1)
                if acquired:
                    pool._lock.release()
                self.assertTrue(acquired, "pool mutex held while waiting for capture")
            future.result(timeout=5)

    def test_operations_remain_concurrent(self):
        gate = CudaGraphGate()
        barrier = threading.Barrier(3, timeout=5)

        def operation():
            with gate.operation(), gate.operation():
                barrier.wait()

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(operation) for _ in range(2)]
            barrier.wait()
            for future in futures:
                future.result(timeout=5)

    def test_capture_waits_for_operation_and_blocks_new_operations(self):
        gate = CudaGraphGate()
        started, captured, release = (threading.Event() for _ in range(3))
        operated = threading.Event()

        def capture():
            started.set()
            with gate.capture():
                captured.set()
                self.assertTrue(release.wait(5))

        def operation():
            with gate.operation():
                operated.set()

        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            try:
                with gate.operation():
                    writer = pool.submit(capture)
                    self.assertTrue(started.wait(5))
                    self.assertFalse(captured.wait(0.05))
                    # Existing readers can re-enter even with a writer waiting.
                    with gate.operation():
                        pass
                self.assertTrue(captured.wait(5))
                reader = pool.submit(operation)
                self.assertFalse(operated.wait(0.05))
            finally:
                release.set()
            writer.result(timeout=5)
            reader.result(timeout=5)
            self.assertTrue(operated.is_set())

    def test_exception_releases_gate_and_upgrade_fails_without_deadlock(self):
        gate = CudaGraphGate()
        with self.assertRaisesRegex(ValueError, "test"):
            with gate.capture():
                raise ValueError("test")
        with gate.operation():
            with self.assertRaisesRegex(RuntimeError, "upgrade"):
                with gate.capture():
                    self.fail("upgrade must fail")
        with gate.capture():
            pass

    def test_waiting_for_result_does_not_block_capture(self):
        entry = MMEmbeddingCacheEntry()
        started = threading.Event()

        def wait():
            started.set()
            return entry.wait(timeout=5)

        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            future = pool.submit(wait)
            self.assertTrue(started.wait(5))
            with cuda_graph_gate.capture():
                entry.complete("ready")
            self.assertEqual(future.result(timeout=5), "ready")


class _TinyVision(nn.Module):
    def prepare_cuda_graph_attention_context(self, *args):
        return None

    def forward(self, pixels, grid, attention_context=None):
        return pixels * 2 + 1


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaGraphCacheConcurrencyTest(unittest.TestCase):
    def test_capture_excludes_cache_sync_copy_and_pool_insertion(self):
        producer = torch.cuda.Stream()
        consumer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            source = torch.full((1024, 1024), 7.0, device="cuda")
            ready = torch.cuda.Event()
            ready.record()
        ready.synchronize()
        sample = torch.ones((4, 8), device="cuda")
        grid = torch.tensor([[1, 2, 2]])
        cache = MMEmbeddingCache(0, 8 * 1024 * 1024)
        _, entry = cache.try_acquire("image")
        uncached = MMEmbeddingCacheEntry()
        uncached.producer_streams = [producer]
        uncached.complete(source)
        actions = {
            "ready_event": lambda: MMEmbeddingCache._ready_result(source, [ready]),
            "blocking_copy": lambda: _copy_result_to_devices(
                source, torch.device("cpu")
            ),
            "pooled_cache": lambda: entry.complete((source, None)),
            "uncached_wait": lambda: uncached.wait(),
            "uncached_stream_wait": lambda: uncached.wait(wait_on_current_stream=True),
            "feature_hash": lambda: _feature_hashes_from_result((source, None)),
            "protobuf_export": lambda: build_multimodal_output_pb([source], None, None),
        }
        original_enter = torch.cuda.graph.__enter__
        for name, action in actions.items():
            with self.subTest(action=name):
                graph_cache = m3._MiniMaxM3VLVisionGraphCache(
                    _TinyVision(), capture_after=1
                )
                go, attempted, done = (threading.Event() for _ in range(3))

                def worker():
                    self.assertTrue(go.wait(5))
                    attempted.set()
                    try:
                        return action()
                    finally:
                        done.set()

                def enter(context):
                    value = original_enter(context)
                    go.set()
                    self.assertTrue(attempted.wait(5))
                    self.assertFalse(done.wait(0.05))
                    return value

                with concurrent.futures.ThreadPoolExecutor(1) as pool:
                    future = pool.submit(worker)
                    try:
                        with mock.patch.object(
                            torch.cuda.graph, "__enter__", enter
                        ), mock.patch.object(
                            m3, "_select_attention_backend", return_value="fa4"
                        ):
                            output = graph_cache.run(sample, grid)
                    finally:
                        go.set()
                    result = future.result(timeout=5)
                self.assertEqual(graph_cache.stats()["capture"], 1)
                self.assertEqual(graph_cache.stats()["fallback"], 0)
                torch.testing.assert_close(output, sample * 2 + 1)
                if name == "blocking_copy":
                    torch.testing.assert_close(result, source.cpu())
                if name == "pooled_cache":
                    torch.testing.assert_close(entry.wait()[0], source)

        # A failed capture used to strand these record_stream frees indefinitely.
        for _ in range(20):
            with torch.cuda.stream(producer):
                value = torch.ones(2 * 1024 * 1024, device="cuda")
            with torch.cuda.stream(consumer):
                consumer.wait_stream(producer)
                value.add_(1)
                value.record_stream(consumer)
            torch.cuda.synchronize()
            del value
            probe = torch.empty(1, device="cuda")
            del probe
        pending = sum(
            block["size"]
            for segment in torch.cuda.memory_snapshot()
            for block in segment["blocks"]
            if block["state"] == "active_pending_free"
        )
        self.assertEqual(pending, 0)

    def test_capture_end_failure_exits_instead_of_eager_fallback(self):
        graph = m3._MiniMaxM3VLCUDAGraph()
        with mock.patch.object(
            torch.cuda.CUDAGraph,
            "capture_end",
            side_effect=RuntimeError("capture invalidated"),
        ), mock.patch.object(m3.os, "_exit", side_effect=SystemExit(1)) as exit_worker:
            with self.assertRaises(SystemExit):
                graph.capture_end()
        exit_worker.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
