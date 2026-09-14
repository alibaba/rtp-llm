import unittest

import torch

from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.gpu_video import nv12_to_rgb
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_vit import (
    apply_rotary_pos_emb_vision,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class VisionKernelsTest(unittest.TestCase):
    def test_nv12_matches_integer_reference_for_both_color_matrices(self):
        torch.manual_seed(27)
        for n, h, w in [(1, 2, 2), (3, 32, 48), (2, 720, 1280)]:
            video = torch.randint(0, 256, (n, h * 3 // 2, w), dtype=torch.uint8)
            for color_space in (1, 2, 5, 6):
                expected = nv12_to_rgb(video, h, color_space)
                actual = nv12_to_rgb(video.cuda(), h, color_space)
                torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)

    def test_rope_matches_fp32_reference_with_strided_qkv_and_positions(self):
        torch.manual_seed(28)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for n, heads, dim in [(257, 16, 72), (3, 2, 8), (0, 2, 8), (1, 1, 2)]:
                qkv = torch.randn(n, 3, heads, dim, device="cuda", dtype=dtype)
                q, k, _ = qkv.unbind(1)
                angles = torch.randn(n, dim * 2, device="cuda")
                cosine, sine = angles.cos()[:, ::2], angles.sin()[:, ::2]
                expected = []
                for value in (q, k):
                    value = value.float()
                    rotated = torch.cat(
                        (-value[..., dim // 2 :], value[..., : dim // 2]), -1
                    )
                    expected.append(
                        (value * cosine[:, None] + rotated * sine[:, None]).to(dtype)
                    )
                actual = apply_rotary_pos_emb_vision(q, k, cosine, sine)
                for got, ref in zip(actual, expected):
                    torch.testing.assert_close(got, ref, atol=0, rtol=0)

    def test_rope_materializes_fp32_rounding_before_bf16_conversion(self):
        # A real activation whose FP32 sum is exactly halfway between two
        # BF16 values. Keep the 72-wide strided QKV layout that exposed it.
        qkv = torch.empty(513, 3, 16, 72, device="cuda", dtype=torch.bfloat16)
        q, k, _ = qkv.unbind(1)
        q[..., :36] = 3.03125
        q[..., 36:] = -2.875
        k.copy_(q)
        cos = torch.full((513, 72), 0.9988769292831421, device="cuda")
        sin = torch.full((513, 72), 0.047379810363054276, device="cuda")
        expected = torch.empty_like(q)
        expected[..., :36] = 3.15625
        expected[..., 36:] = -2.734375
        actual = apply_rotary_pos_emb_vision(q, k, cos, sin)
        for result in actual:
            torch.testing.assert_close(result, expected, atol=0, rtol=0)

    def test_rope_retains_autograd_fallback(self):
        q = torch.randn(5, 2, 8, device="cuda", requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        cos, sin = torch.ones(5, 8, device="cuda"), torch.zeros(5, 8, device="cuda")
        oq, ok = apply_rotary_pos_emb_vision(q, k, cos, sin)
        (oq.sum() + ok.sum()).backward()
        torch.testing.assert_close(q.grad, torch.ones_like(q))
        torch.testing.assert_close(k.grad, torch.ones_like(k))


if __name__ == "__main__":
    unittest.main()
