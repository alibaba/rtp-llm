"""Wire packing and base-2 merge, including Triton's token=1 specialization.

GPU kernels are real. The wire transfer is constructed locally, so this test
makes no claim about distributed all-to-all or PD request routing.
"""
import unittest

import torch

from rtp_llm.models_py.triton_kernels.mla.dcp import (
    _combine_a2a,
    _pack_a2a,
)


class MlaDcpKernelContractTest(unittest.TestCase):
    def test_tp4_tp8_one_and_many_tokens_eager_and_graph(self):
        for cp in (4, 8):
            for tokens in (1, 4, 7):
                with self.subTest(cp=cp, tokens=tokens):
                    self.check_wire(cp, tokens)

    def check_wire(self, cp, tokens):
        local_heads, dim = 8, 512
        heads = local_heads * cp
        values = torch.arange(1, cp + 1, device="cuda", dtype=torch.float32)
        lengths = torch.arange(1, cp + 1, device="cuda", dtype=torch.int32)[:, None].expand(cp, tokens).clone()
        lengths[-1] = 0
        if tokens > 1:
            lengths[:, -1] = 0
        partial = [torch.full((tokens, heads, dim), float(i + 1), device="cuda", dtype=torch.bfloat16) for i in range(cp)]
        lse = [row.float().log2()[:, None].expand(tokens, heads).contiguous() for row in lengths]
        # Poison empty rows. Packing must publish exactly O=0/LSE=-inf.
        for i in range(cp):
            partial[i][lengths[i] == 0] = float("nan")
            lse[i][lengths[i] == 0] = float("nan")
        packed = [torch.empty((cp, tokens, local_heads, dim + 2), device="cuda", dtype=torch.bfloat16) for _ in range(cp)]
        received = torch.empty_like(packed[0])
        output = torch.empty((local_heads, tokens, dim), device="cuda", dtype=torch.bfloat16)
        destination = cp - 1

        def forward():
            for i in range(cp):
                _pack_a2a[(tokens, heads)](
                    partial[i], lse[i], lengths[i], packed[i], packed[i].view(torch.float32), tokens,
                    heads=heads, local_heads=local_heads, dim=dim, block=512,
                )
                received[i].copy_(packed[i][destination])
            _combine_a2a[(tokens, local_heads)](
                received, received.view(torch.float32), output, tokens,
                local_heads=local_heads, dim=dim, cp_size=cp, block=512,
            )

        def check():
            for i in range(cp):
                expected_lse = torch.where(lengths[i] > 0, lengths[i].float().log2(), -float("inf"))
                torch.testing.assert_close(
                    received[i].view(torch.float32)[..., -1],
                    expected_lse[:, None].expand(tokens, local_heads), rtol=0, atol=0,
                )
                self.assertTrue((received[i, lengths[i] == 0, :, :dim] == 0).all().item())
            denominator = lengths.float().sum(0)
            expected = torch.where(denominator > 0, (lengths.float() * values[:, None]).sum(0) / denominator, 0)
            expected = expected[None, :, None].expand(local_heads, tokens, dim).to(torch.bfloat16)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        for _ in range(3):
            forward()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        # Replaying must use the new values, with the same graph metadata.
        for i in range(cp):
            partial[i][lengths[i] > 0] = float(2 * (i + 1))
        values.mul_(2)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        check()


if __name__ == "__main__":
    unittest.main()
