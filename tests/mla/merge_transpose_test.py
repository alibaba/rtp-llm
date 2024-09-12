import os
import unittest
import torch

class TestRope(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        torch.classes.load_library(os.environ['TEST_SRCDIR'] + "/maga_transformer/tests/libtest_ops.so")
        self.merge_transpose_op = torch.classes.unittest.MergeTransposeOP()

    def test_merge_transpose(self):
        def rope_transpose(x: torch.Tensor):
            s, h, d = x.shape
            return x.view(s, h, d // 2, 2).transpose(3, 2).reshape(s, h, d)

        nope = 128
        rope = nope // 2
        vhead = nope
        for token_num in range(512, 1025, 512):
            for head_num in [16, 32]:
                q = torch.rand(token_num, head_num, nope + rope).cuda()
                k_nope = torch.rand(token_num, head_num, nope).cuda()
                k_rope = torch.rand(token_num, 1, rope).cuda()
                v = torch.rand(token_num, head_num, vhead).cuda()
                qkv = torch.rand(token_num, head_num, 3 * (nope + rope)).cuda()
                self.merge_transpose_op.forward(q, k_nope, k_rope, v, qkv)

                q_nope, q_rope = q.split([nope, rope], dim=-1)
                q_rope = rope_transpose(q_rope)
                k_rope = rope_transpose(k_rope)
                qkv_expected = torch.zeros(token_num, head_num, 3 * (nope + rope)).cuda()
                qkv_expected[:, :, :nope] = q_nope
                qkv_expected[:, :, nope:nope+rope] = q_rope
                qkv_expected[:, :, nope+rope:nope*2 + rope] = k_nope
                qkv_expected[:, :, nope*2+rope:nope*2+rope*2] = k_rope
                qkv_expected[:, :, nope*2+rope*2:nope*2+rope*2+vhead] = v

                torch.testing.assert_close(qkv, qkv_expected, atol=1e-3, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
