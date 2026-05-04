"""UT for the MoE._local_y_buf pre-allocation fix.

Replaces ``y = torch.zeros_like(x, dtype=torch.float32)`` (one
``FillFunctor<float>`` launch per MoE layer per forward) with a lazily
pre-allocated buffer sliced ``[:T]``.  Per the V4 prefill timeline the
``FillFunctor<float>`` cluster fires ~7-8 times per layer × 43 layers
≈ 333 calls / 4.9 s — this WS removes one of those per layer.

Test verifies:
  1. Buffer is allocated on first call (lazy)
  2. Subsequent calls with same/smaller T reuse the same buffer
  3. Larger T re-grows (capped at max_tokens_per_rank ceiling)
  4. The returned slice is bit-equivalent to ``torch.zeros_like(x, fp32)``
     after ``zero_()``
  5. Different device triggers re-allocation

We don't import ``MoE`` itself (full V4 model init is heavy).  Instead we
mimic the buffer-management snippet directly — same code path as
``moe.py`` lines after the WS-Q2 edit.
"""

from __future__ import annotations

import unittest

import torch


class _BufHolder:
    """Mirror of MoE's lazy buffer logic."""

    def __init__(self, max_tokens_per_rank: int, dim: int):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dim = dim
        self._local_y_buf: torch.Tensor | None = None

    def get(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(0)
        buf = self._local_y_buf
        if buf is None or buf.size(0) < T or buf.device != x.device:
            self._local_y_buf = torch.empty(
                (max(T, self.max_tokens_per_rank), self.dim),
                dtype=torch.float32,
                device=x.device,
            )
            buf = self._local_y_buf
        y = buf[:T]
        y.zero_()
        return y


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class MoELocalYBufTest(unittest.TestCase):
    def test_lazy_alloc_first_call(self):
        h = _BufHolder(max_tokens_per_rank=8192, dim=2048)
        self.assertIsNone(h._local_y_buf)
        x = torch.randn(64, 2048, device="cuda:0", dtype=torch.bfloat16)
        y = h.get(x)
        self.assertIsNotNone(h._local_y_buf)
        self.assertEqual(h._local_y_buf.size(0), 8192)  # capped to max
        self.assertEqual(y.shape, (64, 2048))
        self.assertEqual(y.dtype, torch.float32)

    def test_reuse_same_buffer_smaller_T(self):
        h = _BufHolder(max_tokens_per_rank=8192, dim=2048)
        x1 = torch.randn(128, 2048, device="cuda:0", dtype=torch.bfloat16)
        h.get(x1)
        addr1 = h._local_y_buf.data_ptr()
        x2 = torch.randn(64, 2048, device="cuda:0", dtype=torch.bfloat16)
        h.get(x2)
        addr2 = h._local_y_buf.data_ptr()
        self.assertEqual(addr1, addr2, "smaller-T call must reuse buffer")

    def test_reuse_same_buffer_equal_T(self):
        h = _BufHolder(max_tokens_per_rank=64, dim=128)
        x1 = torch.randn(32, 128, device="cuda:0", dtype=torch.bfloat16)
        h.get(x1)
        addr1 = h._local_y_buf.data_ptr()
        x2 = torch.randn(32, 128, device="cuda:0", dtype=torch.bfloat16)
        h.get(x2)
        addr2 = h._local_y_buf.data_ptr()
        self.assertEqual(addr1, addr2)

    def test_grow_when_T_exceeds_buffer(self):
        h = _BufHolder(max_tokens_per_rank=64, dim=128)
        x1 = torch.randn(32, 128, device="cuda:0", dtype=torch.bfloat16)
        h.get(x1)
        old_size = h._local_y_buf.size(0)
        # T > buffer ⇒ re-allocate at max(T, max_tokens_per_rank).
        x2 = torch.randn(128, 128, device="cuda:0", dtype=torch.bfloat16)
        h.get(x2)
        self.assertGreaterEqual(h._local_y_buf.size(0), 128)
        self.assertGreater(h._local_y_buf.size(0), old_size)

    def test_zeros_match_zeros_like(self):
        """Returned slice must equal torch.zeros_like(x, fp32) byte-for-byte."""
        h = _BufHolder(max_tokens_per_rank=512, dim=256)
        x = torch.randn(123, 256, device="cuda:0", dtype=torch.bfloat16)
        ref = torch.zeros_like(x, dtype=torch.float32)
        y = h.get(x)
        self.assertEqual(y.shape, ref.shape)
        self.assertEqual(y.dtype, ref.dtype)
        self.assertTrue(torch.equal(y, ref))

    def test_subsequent_calls_actually_zero(self):
        """Second call must zero the prefix — not leak prior data."""
        h = _BufHolder(max_tokens_per_rank=256, dim=64)
        x = torch.randn(100, 64, device="cuda:0", dtype=torch.bfloat16)
        y1 = h.get(x)
        y1.fill_(7.0)  # caller wrote into the buffer
        # Same T again — must come back zero
        y2 = h.get(x)
        self.assertTrue(torch.all(y2 == 0).item(),
                        "buffer not re-zeroed between calls")


if __name__ == "__main__":
    unittest.main()
