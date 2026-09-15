"""Run under the production TMS preload shim, once per allocator configuration.

Use separate processes for sleep levels 1/2 and large_segment_size_mb values.
No model or distributed group is needed; the test uses one idle CUDA device.
"""

import os
import unittest
from contextlib import contextmanager
from unittest import mock

import torch

from rtp_llm.model_loader import weight_memory_saver as wms


@contextmanager
def capture_without_allocator_cleanup(graph, stream):
    # torch.cuda.graph.__enter__ flushes the default allocator itself. Use the
    # same capture API directly so the no-flush assertion tests our scratch
    # allocation, rather than the convenience context's startup housekeeping.
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        try:
            yield
        finally:
            graph.capture_end()


@unittest.skipUnless(torch.cuda.is_available(), "requires an idle CUDA device")
class PausableScratchGpuTest(unittest.TestCase):
    def test_growth_graph_replay_and_sleep_restore(self):
        import librtp_compute_ops

        self.assertTrue(
            librtp_compute_ops.rtp_llm_ops.sleep_memory_allocator_available(),
            "launch this test with the production torch_memory_saver preload shim",
        )
        level = int(os.environ.get("SLEEP_MODE_LEVEL", "2"))
        wms.configure_from_runtime(True, level)
        # Follow service startup ordering, then leave runtime activations on the
        # user's original allocator policy for every scratch allocation below.
        with wms.weights_region():
            weight = torch.ones(1024, device="cuda")
        saver = wms._get_tms()
        self.assertIsNotNone(saver)
        wms.release_init_segment_splitting()
        wms.enable_runtime_expandable()
        runtime_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        runtime_expandable = wms._expandable_live
        sentinel = torch.full((1024,), 3.0, device="cuda")

        with (
            mock.patch.object(
                torch.cuda,
                "empty_cache",
                side_effect=AssertionError("online cache flush"),
            ),
            mock.patch.object(
                wms,
                "_apply_live_alloc_conf",
                side_effect=AssertionError("online policy change"),
            ),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Warm up the operation, but create the first scratch pool and
                # its first allocation inside capture, as lazy model buffers do.
                torch.add(sentinel, weight)
            torch.cuda.current_stream().wait_stream(stream)
            allocation_graph = torch.cuda.CUDAGraph()
            with capture_without_allocator_cleanup(allocation_graph, stream):
                captured = wms.pausable_empty((1024,), device="cuda")
                torch.add(sentinel, weight, out=captured)
            captured_pointer = captured.data_ptr()
            allocation_graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.all(captured == 4).item())

            buffers = [
                wms.pausable_empty((size,), device="cuda")
                for size in (1024, 1048576, 4194304)
            ]
            for buffer in buffers:
                buffer.fill_(7)
            pointers = [buffer.data_ptr() for buffer in buffers]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Warm up before capture; graph output is held alive at its VA.
                graph_output = wms.pausable_empty((1024,), device="cuda")
                torch.add(sentinel, weight, out=graph_output)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with capture_without_allocator_cleanup(graph, stream):
                torch.add(sentinel, weight, out=graph_output)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.all(graph_output == 4).item())

            for _ in range(2):
                saver.pause(wms.WEIGHTS_TAG)
                saver.resume(wms.WEIGHTS_TAG)
                self.assertEqual([buffer.data_ptr() for buffer in buffers], pointers)
                if level == 1:
                    for buffer in buffers:
                        self.assertTrue(torch.all(buffer == 7).item())
                # Level 2 is discard/reload: refill inputs, not scratch outputs.
                weight.fill_(1)
                graph.replay()
                allocation_graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.all(graph_output == 4).item())
                self.assertEqual(captured.data_ptr(), captured_pointer)
                self.assertTrue(torch.all(captured == 4).item())
                self.assertTrue(torch.all(sentinel == 3).item())

            # Growth after wake must still avoid global allocator mutation.
            grown = wms.pausable_empty((8388608,), device="cuda")
            grown.fill_(9)
            self.assertTrue(torch.all(grown == 9).item())

            # The Python tensor may die before a consumer on another stream.
            # record_stream must delay reuse of its private-pool block.
            source = wms.pausable_empty((262144,), device="cuda")
            source.fill_(11)
            observed = torch.empty_like(source)
            finished = torch.cuda.Event()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10000000)
                observed.copy_(source)
                finished.record()
            source.record_stream(stream)
            source_pointer = source.data_ptr()
            del source
            replacement = wms.pausable_empty((262144,), device="cuda")
            if not finished.query():
                self.assertNotEqual(replacement.data_ptr(), source_pointer)
            replacement.fill_(17)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            self.assertTrue(torch.all(observed == 11).item())
        self.assertEqual(wms._expandable_live, runtime_expandable)
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), runtime_conf)


if __name__ == "__main__":
    unittest.main()
