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
import types
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


class TestMegaGatePackViews(unittest.TestCase):
    def test_live_rows_reuse_and_buffer_replacement(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy

        strategy = MegaMoEStrategy.__new__(MegaMoEStrategy)
        torch.nn.Module.__init__(strategy)
        strategy._mega_buf = types.SimpleNamespace(
            x=torch.zeros(16, 8),
            x_sf=torch.zeros(16, 2),
            topk_idx=torch.zeros(16, 2, dtype=torch.int64),
            topk_weights=torch.zeros(16, 2),
        )
        strategy._mega_y = torch.zeros(16, 8)
        views = strategy._gate_pack_views(7)
        self.assertIs(strategy._gate_pack_views(7), views)
        views[0].fill_(3)
        torch.testing.assert_close(strategy._mega_buf.x[:7], torch.full((7, 8), 3.0))
        self.assertEqual(strategy._mega_buf.x[7:].count_nonzero().item(), 0)
        for rows in (0, 1, 16, 7):
            current = strategy._gate_pack_views(rows)
            self.assertTrue(all(t.shape[0] == rows for t in current))
            self.assertIs(strategy._gate_pack_views(rows), current)
        strategy._mega_y = torch.ones(16, 8)
        self.assertEqual(strategy._gate_pack_views(7)[-1].sum().item(), 56)
        strategy._mega_buf = types.SimpleNamespace(
            **{
                name: value.clone() + 1
                for name, value in vars(strategy._mega_buf).items()
            }
        )
        self.assertEqual(strategy._gate_pack_views(7)[0][0, 0].item(), 4)


if __name__ == "__main__":
    unittest.main()
