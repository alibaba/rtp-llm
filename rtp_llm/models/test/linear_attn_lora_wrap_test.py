import unittest

import torch

from rtp_llm.model_loader.linear_attn_weight import (
    ba_lora_a_func_wrap,
    ba_lora_b_func_wrap,
    qkvz_lora_a_func_wrap,
    qkvz_lora_b_func_wrap,
)


H, QKV_OUT, Z_OUT = 32, 24, 16


class LinearAttnPeftLoraWrapTest(unittest.TestCase):
    """JEV-style peft adapters over linear_attn in_proj_qkv/z (codex B4)."""

    def test_both_projections_merge_per_projection(self):
        rq, rz = 4, 3
        qkvA, zA = torch.randn(rq, H), torch.randn(rz, H)
        qkvB, zB = torch.randn(QKV_OUT, rq), torch.randn(Z_OUT, rz)

        a = qkvz_lora_a_func_wrap([qkvA, zA])
        expect_a = torch.cat([qkvA.t(), zA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkvz_lora_b_func_wrap([qkvB, zB])
        expect_b = torch.block_diag(qkvB.t(), zB.t())
        torch.testing.assert_close(b, expect_b)

        scale = 2.0
        merged = a @ b * scale
        expect_delta = torch.cat(
            [(qkvB @ qkvA * scale).t(), (zB @ zA * scale).t()], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)

    def test_z_only_adapter_zero_fills_qkv(self):
        # adapter targeting only in_proj_z: the qkv delta is exactly zero.
        # The untargeted side's zero blocks borrow the targeted side's rank
        # (the rank of an untargeted projection is meaningless; the delta is
        # zero either way).
        rz = 3
        zA, zB = torch.randn(rz, H), torch.randn(Z_OUT, rz)

        a = qkvz_lora_a_func_wrap([None, zA])
        expect_a = torch.cat([torch.zeros(H, rz), zA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkvz_lora_b_func_wrap([None, zB])
        # qkv zero block: B rows [rz, QKV_OUT]; z block: [rz, Z_OUT]
        expect_b = torch.block_diag(
            torch.zeros(Z_OUT, rz).t(), zB.t()
        )
        torch.testing.assert_close(b, expect_b)

    def test_fully_untargeted_returns_none(self):
        self.assertIsNone(qkvz_lora_a_func_wrap([None, None]))
        self.assertIsNone(qkvz_lora_b_func_wrap([None, None]))
        self.assertIsNone(ba_lora_a_func_wrap([None, None]))
        self.assertIsNone(ba_lora_b_func_wrap([None, None]))

    def test_ba_wrap_matches_qkvz_semantics(self):
        rb, ra = 2, 5
        bA, aA = torch.randn(rb, H), torch.randn(ra, H)
        bB, aB = torch.randn(8, rb), torch.randn(6, ra)
        a = ba_lora_a_func_wrap([bA, aA])
        expect_a = torch.cat([bA.t(), aA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)
        b = ba_lora_b_func_wrap([bB, aB])
        expect_b = torch.block_diag(bB.t(), aB.t())
        torch.testing.assert_close(b, expect_b)


if __name__ == "__main__":
    unittest.main()
