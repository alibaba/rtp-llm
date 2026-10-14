import unittest

import torch

from rtp_llm.model_loader.linear_attn_weight import (
    ba_lora_a_func_wrap,
    ba_lora_b_func_wrap,
    qkvz_lora_a_func_wrap,
    qkvz_lora_b_func_wrap,
)


# JEV-like geometry: qkv [2*k_heads*k_dim + v_heads*v_dim], z [v_heads*v_dim]
K_HEADS, K_DIM, V_HEADS, V_DIM = 8, 128, 16, 128
H = 5120
QKV_OUT = 2 * K_HEADS * K_DIM + V_HEADS * V_DIM  # 10,240
Z_OUT = V_HEADS * V_DIM  # 6,144
B_OUT = V_HEADS  # 16
A_OUT = V_HEADS * V_DIM  # 6,144


class LinearAttnPeftLoraWrapTest(unittest.TestCase):
    """JEV-style peft adapters over linear_attn in_proj_qkv/z (codex B4/C2)."""

    def test_both_projections_merge_per_projection(self):
        rq, rz = 4, 3
        qkvA, zA = torch.randn(rq, H), torch.randn(rz, H)
        qkvB, zB = torch.randn(QKV_OUT, rq), torch.randn(Z_OUT, rz)

        a = qkvz_lora_a_func_wrap([qkvA, zA])
        expect_a = torch.cat([qkvA.t(), zA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkvz_lora_b_func_wrap([qkvB, zB], QKV_OUT, Z_OUT)
        expect_b = torch.block_diag(qkvB.t(), zB.t())
        torch.testing.assert_close(b, expect_b)

        scale = 2.0
        merged = a @ b * scale
        expect_delta = torch.cat(
            [(qkvB @ qkvA * scale).t(), (zB @ zA * scale).t()], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)
        self.assertEqual(merged.shape, (H, QKV_OUT + Z_OUT))

    def test_z_only_zero_fills_qkv_with_qkv_width(self):
        # z-only adapter: the qkv zero-B block must use QKV's own output
        # width (10,240), never z's (6,144) — borrowing the sibling width
        # breaks the fused layout (codex round-2 C2).
        rz = 3
        zA, zB = torch.randn(rz, H), torch.randn(Z_OUT, rz)

        a = qkvz_lora_a_func_wrap([None, zA])
        expect_a = torch.cat([torch.zeros(H, rz), zA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkvz_lora_b_func_wrap([None, zB], QKV_OUT, Z_OUT)
        expect_b = torch.block_diag(
            torch.zeros(rz, QKV_OUT), zB.t()
        )
        torch.testing.assert_close(b, expect_b)

        scale = 2.0
        merged = a @ b * scale
        expect_delta = torch.cat(
            [torch.zeros(H, QKV_OUT), (zB @ zA * scale).t()], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)
        self.assertEqual(merged.shape, (H, QKV_OUT + Z_OUT))

    def test_fully_untargeted_returns_none(self):
        self.assertIsNone(qkvz_lora_a_func_wrap([None, None]))
        self.assertIsNone(qkvz_lora_b_func_wrap([None, None], QKV_OUT, Z_OUT))
        self.assertIsNone(ba_lora_a_func_wrap([None, None]))
        self.assertIsNone(ba_lora_b_func_wrap([None, None], B_OUT, A_OUT))

    def test_ba_wrap_widths(self):
        rb, ra = 2, 5
        bA, aA = torch.randn(rb, H), torch.randn(ra, H)
        bB, aB = torch.randn(B_OUT, rb), torch.randn(A_OUT, ra)
        a = ba_lora_a_func_wrap([bA, aA])
        expect_a = torch.cat([bA.t(), aA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)
        b = ba_lora_b_func_wrap([bB, aB], B_OUT, A_OUT)
        expect_b = torch.block_diag(bB.t(), aB.t())
        torch.testing.assert_close(b, expect_b)

    def test_ba_b_only_zero_uses_b_width(self):
        # b-only adapter: a-side zero block carries A's own width
        rb = 2
        bA, bB = torch.randn(rb, H), torch.randn(B_OUT, rb)
        b = ba_lora_b_func_wrap([bB, None], B_OUT, A_OUT)
        expect_b = torch.block_diag(bB.t(), torch.zeros(rb, A_OUT))
        torch.testing.assert_close(b, expect_b)


if __name__ == "__main__":
    unittest.main()
