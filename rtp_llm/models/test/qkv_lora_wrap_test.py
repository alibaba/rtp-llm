import unittest

import torch

from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    gated_q_lora_a_func_wrap,
    gated_q_lora_b_func_wrap,
    qkv_lora_a_func_wrap,
    qkv_lora_b_func_wrap,
)


HEAD_NUM, HEAD_DIM = 4, 64


class QkvPeftPartialTargetTest(unittest.TestCase):
    """Partial-projection peft adapters must zero-fill the untargeted ones."""

    def test_q_and_v_without_k_zero_fills_k(self):
        # target_modules=["q_proj","v_proj"]: k is untargeted and must
        # contribute an exact zero delta, not drop the q/v deltas (codex B5).
        h, r = HEAD_NUM * HEAD_DIM, 4
        qA, vA = torch.randn(r, h), torch.randn(r, h)
        qB, vB = torch.randn(h, r), torch.randn(h, r)

        a = qkv_lora_a_func_wrap([qA, None, vA])
        expect_a = torch.cat([qA.t(), torch.zeros(h, r), vA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkv_lora_b_func_wrap([qB, None, vB], HEAD_NUM, HEAD_DIM)
        # q half of the gated q_proj rows, untransposed [heads*head_dim/2, r]
        q_rows = (
            qB.reshape(HEAD_NUM, 2, HEAD_DIM // 2, r)[:, 0, :, :]
            .reshape(-1, r)
        )
        expect_b = torch.block_diag(
            q_rows.t(),
            torch.zeros(r, h),
            vB.t(),
        )
        torch.testing.assert_close(b, expect_b)

        scale = 2.0
        merged = a @ b * scale
        q_delta = (q_rows @ qA * scale).t()
        v_delta = (vB @ vA * scale).t()
        expect_delta = torch.cat(
            [q_delta, torch.zeros(h, h), v_delta], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)

    def test_fully_untargeted_qkv_returns_none(self):
        self.assertIsNone(qkv_lora_a_func_wrap([None, None, None]))
        self.assertIsNone(qkv_lora_b_func_wrap([None, None, None], HEAD_NUM, HEAD_DIM))

    def test_gated_untargeted_returns_none(self):
        self.assertIsNone(gated_q_lora_a_func_wrap([None]))
        self.assertIsNone(gated_q_lora_b_func_wrap([None], HEAD_NUM, HEAD_DIM))


if __name__ == "__main__":
    unittest.main()
