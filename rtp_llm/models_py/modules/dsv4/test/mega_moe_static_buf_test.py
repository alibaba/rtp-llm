"""Stream B — Mega MoE static output buffer (CUDA graph safety).

Tests that:
  1. A pre-allocated tensor slice shares storage with the original tensor
     (no reallocation) — this is the key property for CUDA graph capture.
  2. The slice `self._mega_y[:T]` is correctly sized for the live batch.
  3. The slice shares data_ptr with the full buffer (graph-safe).

These tests validate the buffer-slice design without requiring deep_gemm or
torch.distributed (needed for the full MoE init path). Full integration is
covered by the SM100_ARM smoke suite.
"""

import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


class TestStaticBufferSliceProperty(unittest.TestCase):
    """Pre-allocated buffer slice is a view (no reallocation) — key for CUDA graphs."""

    def test_slice_shares_data_ptr(self):
        """y = buf[:T] must have the same data_ptr as buf (no new allocation)."""
        D = 128
        max_T = 64
        T = 16
        buf = torch.empty((max_T, D), dtype=torch.bfloat16)
        y = buf[:T]
        # Same underlying storage.
        self.assertEqual(y.data_ptr(), buf.data_ptr())
        # Not a copy.
        buf[0, 0] = 42.0
        self.assertEqual(y[0, 0].item(), 42.0)

    def test_slice_shape(self):
        D = 256
        max_T = 100
        buf = torch.empty((max_T, D), dtype=torch.bfloat16)
        for T in (1, 32, 99, 100):
            y = buf[:T]
            self.assertEqual(y.shape, (T, D))

    def test_write_through_slice_visible_in_buf(self):
        """Kernel writes to slice are visible in the pre-allocated buffer
        (same memory region); simulates deep_gemm writing to y = buf[:T]."""
        D = 64
        max_T = 32
        T = 8
        buf = torch.zeros((max_T, D), dtype=torch.bfloat16)
        y = buf[:T]
        y.fill_(1.0)  # simulate kernel writing to y
        # Rows 0..T-1 in buf should now be 1.0.
        self.assertTrue(torch.all(buf[:T] == 1.0))
        # Rows T..max_T should remain 0.
        self.assertTrue(torch.all(buf[T:] == 0.0))


class TestMegaMoeBufCodeChange(unittest.TestCase):
    """Verify the Stream B code change semantics in isolation."""

    def test_pre_alloc_avoids_realloc(self):
        """Simulate the forward: buf already allocated, y = buf[:T] is stable."""
        D = 512
        max_T = 384  # typical max_tokens_per_rank on SM100
        T = 37
        # This represents self._mega_y (pre-allocated in _setup_mega_moe).
        mega_y = torch.empty((max_T, D), dtype=torch.bfloat16)
        ptr_before = mega_y.data_ptr()

        # This is what the new code does on every forward step.
        y = mega_y[:T]

        # No new allocation: y.data_ptr() == mega_y.data_ptr()
        self.assertEqual(y.data_ptr(), ptr_before)
        # Shape is correct for the batch.
        self.assertEqual(y.shape, (T, D))

    def test_float_cast_on_slice_does_not_change_buf_dtype(self):
        """y.float() returns a new tensor; does not alter mega_y's bf16 storage."""
        D = 64
        mega_y = torch.zeros((32, D), dtype=torch.bfloat16)
        T = 10
        y = mega_y[:T]
        y.fill_(1.0)
        out = y.float()  # this is what return y.float() does
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(mega_y.dtype, torch.bfloat16)
        self.assertEqual(out.shape, (T, D))


class TestMegaBufferGraphBakedGate(unittest.TestCase):
    """Sleep-time buffer releases must be NO-OPS when CUDA graphs are captured.

    Both Mega MoE buffers (symm ``_mega_buf`` + output staging ``_mega_y``) have
    their device pointers baked into the captured decode graph, so freeing them at
    sleep dangles the VA a post-wake replay writes into (illegal access, both
    ranks -> SIGABRT). When graphs are NOT captured (prefill) they must still be
    freed as before. GPU-free: seeds fake cache/registry entries and toggles the
    flag directly.
    """

    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.moe import mega_buf

        self.mega_buf = mega_buf
        self._saved = mega_buf.mega_buffers_graph_baked()
        self._key = ("cpu", 8, torch.bfloat16)
        self._buf = torch.empty((4, 8), dtype=torch.bfloat16)
        mega_buf._MEGA_OUTPUT_CACHE.clear()
        mega_buf._MEGA_OUTPUT_CACHE[self._key] = self._buf

        class _FakeStrat:
            pass

        self.strat = _FakeStrat()
        self.strat._mega_y = self._buf
        self.strat._mega_buf = object()  # opaque; must not be touched when baked
        mega_buf._MEGA_STRATEGY_REGISTRY.add(self.strat)

    def tearDown(self):
        self.mega_buf.set_mega_buffers_graph_baked(self._saved)
        self.mega_buf._MEGA_OUTPUT_CACHE.clear()
        try:
            self.mega_buf._MEGA_STRATEGY_REGISTRY.discard(self.strat)
        except Exception:
            pass

    def test_output_release_is_noop_when_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(True)
        self.assertEqual(self.mega_buf.release_mega_output_buffers(), (0, 0.0))
        # Buffer + per-strategy ref stay resident across sleep.
        self.assertEqual(len(self.mega_buf._MEGA_OUTPUT_CACHE), 1)
        self.assertIsNotNone(self.strat._mega_y)
        # Footprint is still reportable for the sleep-reclaim note.
        self.assertGreater(self.mega_buf.mega_output_buffer_gib(), 0.0)

    def test_output_release_frees_when_not_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(False)
        entries, gib = self.mega_buf.release_mega_output_buffers()
        self.assertEqual(entries, 1)
        self.assertGreater(gib, 0.0)
        self.assertEqual(len(self.mega_buf._MEGA_OUTPUT_CACHE), 0)
        self.assertIsNone(self.strat._mega_y)

    def test_symm_release_is_noop_when_graph_baked(self):
        self.mega_buf.set_mega_buffers_graph_baked(True)
        # Returns early before dereferencing the opaque _mega_buf / cache.
        self.assertEqual(self.mega_buf.release_mega_symm_buffers(), 0.0)
        self.assertIsNotNone(self.strat._mega_buf)


if __name__ == "__main__":
    unittest.main()
