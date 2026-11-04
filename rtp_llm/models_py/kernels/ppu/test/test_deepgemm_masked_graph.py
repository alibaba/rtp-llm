import unittest

import torch

from rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
    m_grouped_bf16_gemm_nt_masked,
)


class DeepGemmMaskedGraphTest(unittest.TestCase):
    def test_capture_and_replay_use_current_stream(self):
        self.assertTrue(torch.cuda.is_available(), "This test requires a PPU worker")
        torch.cuda.set_device(0)
        groups, capacity = 32, 512
        counts = [1 + i % 4 for i in range(groups)]
        mask = torch.tensor(counts, device="cuda", dtype=torch.int32)
        for k, n in ((2048, 1024), (512, 2048)):
            with self.subTest(k=k, n=n):
                lhs = torch.ones(
                    (groups, capacity, k), device="cuda", dtype=torch.bfloat16
                )
                rhs = torch.ones((groups, n, k), device="cuda", dtype=torch.bfloat16)
                output = torch.zeros(
                    (groups, capacity, n), device="cuda", dtype=torch.bfloat16
                )

                def valid_output():
                    cpu = output.cpu()
                    return torch.cat(
                        [cpu[i, :count].reshape(-1) for i, count in enumerate(counts)]
                    )

                m_grouped_bf16_gemm_nt_masked(lhs, rhs, output, mask, 1)
                torch.cuda.synchronize()
                self.assertTrue(torch.all(valid_output() == k).item())
                output.zero_()
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                with torch.cuda.graph(graph, stream=stream):
                    m_grouped_bf16_gemm_nt_masked(lhs, rhs, output, mask, 1)
                torch.cuda.synchronize()
                # A launch outside the capture stream would already have written D.
                self.assertTrue(torch.all(valid_output() == 0).item())

                for value in (2, 3):
                    lhs.fill_(value)
                    output.zero_()
                    torch.cuda.synchronize()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.all(valid_output() == value * k).item())


if __name__ == "__main__":
    unittest.main()
