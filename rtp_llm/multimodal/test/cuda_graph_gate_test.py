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


if __name__ == "__main__":
    unittest.main()
