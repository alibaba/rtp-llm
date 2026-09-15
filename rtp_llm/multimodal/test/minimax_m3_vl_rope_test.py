import unittest

import torch

from rtp_llm.multimodal.multimodal_mixins.minimax_m3_vl import (
    minimax_m3_vl_rope as rope_module,
)


class MiniMaxM3VLRopeTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_qkv_rope_uses_int64_output_projection_offset(self):
        num_heads = 16
        head_dim = 80
        projection_width = num_heads * head_dim
        sequence_length = torch.iinfo(torch.int32).max // (2 * projection_width) + 1

        qkv_base = torch.zeros(
            (1, 3, num_heads, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        qkv_base[:, 0].fill_(1)
        qkv_base[:, 1].fill_(2)
        qkv_base[:, 2].fill_(3)
        qkv = qkv_base.expand(sequence_length, -1, -1, -1)
        cos = torch.ones((1, 1, head_dim), device="cuda", dtype=torch.bfloat16).expand(
            sequence_length, -1, -1
        )
        sin = torch.zeros_like(cos)

        result = rope_module.fused_qkv_rope(qkv, cos, sin)
        self.assertIsNotNone(result)
        q, k, v = result
        torch.cuda.synchronize()

        torch.testing.assert_close(q[0], qkv_base[0, 0])
        torch.testing.assert_close(q[-1], qkv_base[0, 0])
        torch.testing.assert_close(k[0], qkv_base[0, 1])
        torch.testing.assert_close(k[-1], qkv_base[0, 1])
        torch.testing.assert_close(v[0], qkv_base[0, 2])
        torch.testing.assert_close(v[-1], qkv_base[0, 2])


if __name__ == "__main__":
    unittest.main()
