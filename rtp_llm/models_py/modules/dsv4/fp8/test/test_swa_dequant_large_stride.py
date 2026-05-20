"""Regression test for large-stride SWA FP8 dequant workspace writes.

The production crash this covers used a workspace with shape
``[28, 167319, 512]``.  The dequant kernel's output-row offset crossed
signed int32 at batch 27, so the store address wrapped before the start of
the allocation and raised ``Warp Illegal Address``.
"""

from __future__ import annotations

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


class SwaDequantLargeStrideMathTest(unittest.TestCase):
    def test_core_dump_shape_crosses_int32_row_offset(self) -> None:
        out_stride0 = CORE_DUMP_WORKSPACE_ROWS * HEAD_DIM
        batch_offset = CORE_DUMP_TARGET_BATCH * out_stride0
        self.assertGreater(batch_offset, 2**31 - 1)


class SwaDequantLargeStrideTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda")

    def test_output_row_offset_uses_int64_for_core_dump_shape(self) -> None:
        batch_size = CORE_DUMP_BATCH_SIZE
        workspace_rows = CORE_DUMP_WORKSPACE_ROWS
        block_size = 128
        gather_len = CORE_DUMP_GATHER_LEN
        target_batch = CORE_DUMP_TARGET_BATCH
        target_worker = CORE_DUMP_TARGET_WORKER

        required_bytes = batch_size * workspace_rows * HEAD_DIM * 2
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < required_bytes + (1 << 30):
            self.skipTest(
                "large-stride regression requires about "
                f"{(required_bytes + (1 << 30)) / (1 << 30):.1f} GiB free GPU memory"
            )

        k_cache = torch.zeros(
            1, block_size, ENTRY_BYTES, dtype=torch.uint8, device=self.device
        )
        out = torch.empty(
            batch_size,
            workspace_rows,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=self.device,
        )

        seq_lens = torch.full(
            (batch_size,), gather_len, dtype=torch.int32, device=self.device
        )
        block_table = torch.zeros(batch_size, 1, dtype=torch.int32, device=self.device)

        # With the pre-fix int32 arithmetic:
        #   27 * (167319 * 512) = 2_313_017_856 > INT32_MAX
        # so the first store from the target worker wrapped before the
        # allocation and raised Warp Illegal Address.
        for offset in (0, 64):
            with self.subTest(offset=offset):
                target_row = offset + target_worker
                out[target_batch, target_row].fill_(-7)

                dequantize_and_gather_k_cache(
                    out=out,
                    k_cache=k_cache,
                    seq_lens=seq_lens,
                    gather_lens=None,
                    block_table=block_table,
                    block_size=block_size,
                    offset=offset,
                )
                torch.cuda.synchronize()

                recovered = out[target_batch, target_row]
                self.assertTrue(
                    torch.all(recovered == 0),
                    msg=(
                        "Expected the zero k_cache to dequantize into an all-zero "
                        f"workspace row at batch={target_batch}, row={target_row}. "
                        f"Got min={recovered.min().item()} max={recovered.max().item()}."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
