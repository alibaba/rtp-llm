"""Tests for Mega MoE SE shared-L1 activation-scale staging."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack import (
    stage_mega_moe_se_shared_l1_scales,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MegaMoESEScaleStageTest(unittest.TestCase):
    def _run_case(self, tokens: int, block_m: int) -> None:
        packed_k = 5
        capacity_rows = 4096
        source = torch.arange(
            max(tokens, 1) * packed_k,
            dtype=torch.int32,
            device="cuda",
        ).reshape(max(tokens, 1), packed_k)[:tokens]
        destination = torch.empty_strided(
            (capacity_rows, packed_k),
            (1, capacity_rows),
            dtype=torch.int32,
            device="cuda",
        )
        destination.fill_(1234567)

        stage_mega_moe_se_shared_l1_scales(source, destination, tokens, block_m)
        aligned_block_m = ((block_m + 127) // 128) * 128
        active_rows = (
            ((tokens + block_m - 1) // block_m) * aligned_block_m if tokens else 0
        )
        expected = torch.zeros(
            (active_rows, packed_k), dtype=torch.int32, device="cuda"
        )
        if tokens:
            token_idx = torch.arange(tokens, dtype=torch.long, device="cuda")
            within = token_idx % block_m
            transformed = (
                token_idx // block_m * aligned_block_m
                + (within // 128) * 128
                + (within % 32) * 4
                + (within % 128) // 32
            )
            expected[transformed] = source
        torch.testing.assert_close(destination[:active_rows], expected, rtol=0, atol=0)
        if active_rows < capacity_rows:
            self.assertTrue(torch.all(destination[active_rows:] == 1234567).item())

    def test_reference_mapping_all_kernel_block_sizes(self):
        for block_m in (16, 32, 64, 96, 128, 192):
            with self.subTest(block_m=block_m):
                self._run_case(257, block_m)

    def test_zero_tokens_is_noop(self):
        self._run_case(0, 16)

    def test_reuse_clears_active_holes(self):
        packed_k = 3
        capacity_rows = 1024
        destination = torch.empty_strided(
            (capacity_rows, packed_k),
            (1, capacity_rows),
            dtype=torch.int32,
            device="cuda",
        )
        destination.fill_(99)
        first = torch.full((257, packed_k), 7, dtype=torch.int32, device="cuda")
        second = torch.full((33, packed_k), 11, dtype=torch.int32, device="cuda")
        stage_mega_moe_se_shared_l1_scales(first, destination, 257, 192)
        stage_mega_moe_se_shared_l1_scales(second, destination, 33, 16)

        active_rows = 3 * 128
        expected = torch.zeros(
            (active_rows, packed_k), dtype=torch.int32, device="cuda"
        )
        token_idx = torch.arange(33, dtype=torch.long, device="cuda")
        within = token_idx % 16
        transformed = (
            token_idx // 16 * 128
            + (within // 128) * 128
            + (within % 32) * 4
            + (within % 128) // 32
        )
        expected[transformed] = second
        torch.testing.assert_close(destination[:active_rows], expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
