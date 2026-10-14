import unittest

import torch

from rtp_llm.models.qwen3_next.qwen3_next_weight import (
    gated_q_lora_a_func_wrap,
    gated_q_lora_b_func_wrap,
    qkv_lora_a_func_wrap,
    qkv_lora_b_func_wrap,
)


# GQA geometry: q full width 256 (gated: q half + gate half interleaved),
# k/v each 64 — mirrors the real qwen3.5 attention shapes.
HEAD_NUM, HEAD_DIM, KV_HEAD_NUM = 4, 64, 1
H = HEAD_NUM * HEAD_DIM
Q_OUT = HEAD_NUM * HEAD_DIM
KV_OUT = KV_HEAD_NUM * HEAD_DIM


class QkvPeftPartialTargetTest(unittest.TestCase):
    """Partial-projection peft adapters must zero-fill the untargeted ones."""

    def test_q_and_v_without_k_zero_fills_k_with_k_width(self):
        # target_modules=["q_proj","v_proj"]: k is untargeted and must
        # contribute an exact zero delta with K'S OWN output width (GQA:
        # k/v outputs are narrower than q; codex round-2 C2/B5).
        r = 4
        qA, vA = torch.randn(r, H), torch.randn(r, H)
        # gated q B spans q+gate rows; the wrap splits out the q half
        qB_full = torch.randn(HEAD_NUM * 2 * HEAD_DIM // 2 * 2, r)
        qB = qB_full[: HEAD_NUM * HEAD_DIM]
        vB = torch.randn(KV_OUT, r)

        a = qkv_lora_a_func_wrap([qA, None, vA])
        expect_a = torch.cat([qA.t(), torch.zeros(H, r), vA.t()], dim=1)
        torch.testing.assert_close(a, expect_a)

        b = qkv_lora_b_func_wrap(
            [qB, None, vB], HEAD_NUM, HEAD_DIM, KV_HEAD_NUM
        )
        q_rows = qB.reshape(HEAD_NUM, 2, HEAD_DIM // 2, r)[:, 0, :, :].reshape(-1, r)
        expect_b = torch.block_diag(
            q_rows.t(),
            torch.zeros(r, KV_OUT),
            vB.t(),
        )
        torch.testing.assert_close(b, expect_b)

        scale = 2.0
        merged = a @ b * scale
        q_delta = (q_rows @ qA * scale).t()
        v_delta = (vB @ vA * scale).t()
        expect_delta = torch.cat(
            [q_delta, torch.zeros(H, KV_OUT), v_delta], dim=1
        )
        torch.testing.assert_close(merged, expect_delta)
        # fused delta spans each projection's own width: the gated q rows
        # split to their q half (128), k zero-filled at 64, v at 64
        q_half = HEAD_NUM * HEAD_DIM // 2  # half of the interleaved q/gate rows
        self.assertEqual(merged.shape, (H, q_half + 2 * KV_OUT))

    def test_fully_untargeted_qkv_returns_none(self):
        self.assertIsNone(qkv_lora_a_func_wrap([None, None, None]))
        self.assertIsNone(
            qkv_lora_b_func_wrap([None, None, None], HEAD_NUM, HEAD_DIM, KV_HEAD_NUM)
        )

    def test_gated_untargeted_returns_none(self):
        self.assertIsNone(gated_q_lora_a_func_wrap([None]))
        self.assertIsNone(gated_q_lora_b_func_wrap([None], HEAD_NUM, HEAD_DIM))


if __name__ == "__main__":
    unittest.main()
