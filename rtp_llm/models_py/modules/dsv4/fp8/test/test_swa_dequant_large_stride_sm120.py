"""SM120 cases split out of test_swa_dequant_large_stride.py.
These cases live in this sibling file, leaving the original test file
untouched."""

from __future__ import annotations

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    HEAD_DIM,
    dequantize_and_gather_k_cache,
    gather_k_cache_slots_packed,
)

"""Regression test for large-stride SWA FP8 dequant workspace writes.

The production crash this covers used a workspace with shape
``[28, 167319, 512]``.  The dequant kernel's output-row offset crossed
signed int32 at batch 27, so the store address wrapped before the start of
the allocation and raised ``Warp Illegal Address``.
"""
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    HEAD_DIM,
    dequantize_and_gather_k_cache,
)

CORE_DUMP_BATCH_SIZE = 28
CORE_DUMP_WORKSPACE_ROWS = 167319
CORE_DUMP_TARGET_BATCH = 27
CORE_DUMP_GATHER_LEN = 128
CORE_DUMP_TARGET_WORKER = 95


class SwaDequantLargeStrideTestSm120(unittest.TestCase):
    """Sibling of ``SwaDequantLargeStrideTest`` carrying the SM120 cases."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        self.device = torch.device("cuda", torch.cuda.current_device())

    def test_slot_gather_zeros_negative_and_out_of_range_indices(self) -> None:
        k_cache = torch.full(
            (1, 2, ENTRY_BYTES), 7, dtype=torch.uint8, device=self.device
        )
        slots = torch.tensor([-1, 0, 2], dtype=torch.int64, device=self.device)

        gathered = gather_k_cache_slots_packed(k_cache, slots)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(gathered[0], torch.zeros_like(gathered[0])))
        self.assertTrue(torch.equal(gathered[1], torch.full_like(gathered[1], 7)))
        self.assertTrue(torch.equal(gathered[2], torch.zeros_like(gathered[2])))


if __name__ == "__main__":
    import unittest

    unittest.main()
