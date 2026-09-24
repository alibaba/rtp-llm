import unittest
import torch
from rtp_llm.platforms.ppu.kernels.ppu_bf16_radix_topk import bf16_radix_topk


@unittest.skipUnless(torch.cuda.is_available(), 'requires PPU/CUDA')
class RadixTopKTest(unittest.TestCase):
    def test_native_candidate_overflow_and_graph_replay(self):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import topk_bf16, topk_decode

        for dtype, widths in ((torch.bfloat16, (2048, 16384)),
                              (torch.float32, (2048, 16384, 32768, 65537))):
            for width in widths:
                for k in (512, 1024):
                    with self.subTest(dtype=dtype, width=width, k=k):
                        x = torch.ones(4, width + 128, device='cuda', dtype=dtype)[:, :width]
                        x[:, width // 2:] = 1.0078125
                        x[1].fill_(100000)
                        x[1, width // 2:] = 200000
                        x[2].fill_(0)
                        x[2, ::2] = -0.0
                        lengths = torch.tensor([width, width, width, 17], device='cuda', dtype=torch.int32)
                        starts = torch.zeros_like(lengths)
                        out = torch.empty(4, k, device='cuda', dtype=torch.int32)

                        def run():
                            if dtype == torch.bfloat16:
                                topk_bf16(x, starts, lengths, out)
                            else:
                                topk_decode(x, lengths, out)

                        run()  # Compile and initialize before capture.
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        for replay in range(3):
                            if replay == 1:
                                x[0].fill_(1)
                                x[0, :width // 2] = 1.0078125
                            graph.replay()
                            for row, length in enumerate(lengths.tolist()):
                                expected = torch.full((k,), -1, device='cuda', dtype=torch.int32)
                                count = min(k, length)
                                expected[:count] = torch.argsort(
                                    x[row, :length], descending=True, stable=True
                                )[:count].int()
                                self.assertTrue(torch.equal(out[row].sort().values, expected.sort().values),
                                                (row, replay, out[row, :8].tolist(), expected[:8].tolist()))

    def test_native_wide_selection_and_exact_retries(self):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import topk_bf16

        torch.manual_seed(890618)
        for width in (32768, 65537, 262144):
            for k in (512, 1024):
                with self.subTest(width=width, k=k):
                    x = torch.randn(8, ((width + 127) // 128) * 128,
                                    device='cuda', dtype=torch.bfloat16)[:, :width]
                    x[1].fill_(0)
                    x[2].fill_(100000)  # Values sharing the FP16 overflow bin.
                    x[2, ::2] = 200000
                    x[3].fill_(1)
                    x[3, ::2] = 1.0078125
                    x[4].fill_(-10)
                    x[4, 0] = float('inf')
                    x[5].fill_(0)
                    x[5, ::2] = -0.0
                    x[6, ::2] = float('-inf')
                    starts = torch.zeros(8, device='cuda', dtype=torch.int32)
                    ends = torch.full_like(starts, width)
                    ends[7] = 17
                    out = torch.empty(8, k, device='cuda', dtype=torch.int32)
                    expected = torch.full_like(out, -1)
                    for row, end in enumerate(ends.tolist()):
                        count = min(k, end)
                        expected[row, :count] = torch.argsort(
                            x[row, :end], descending=True, stable=True
                        )[:count].int()
                    expected = expected.sort().values
                    for _ in range(3):
                        topk_bf16(x, starts, ends, out)
                        self.assertTrue(torch.equal(out.sort().values, expected))

    def test_wide_scores_bounds_ties_and_stride(self):
        torch.manual_seed(890617)
        for width in (16385, 32768, 65537, 131072, 250000, 262144):
            for k in (512, 1024):
                with self.subTest(width=width, k=k):
                    x = torch.randn(8, width + 7, device='cuda').bfloat16()[:, :width]
                    x[1].fill_(0)
                    x[1, ::2] = -0.0
                    x[2].fill_(-2)
                    x[2, ::2] = 3
                    x[3, :2048] = float('inf')
                    x[3, 2048:4096] = float('-inf')
                    x[7].fill_(-10)
                    x[7, 0] = float('inf')  # K requires values beyond the maximum-key window.
                    starts = torch.tensor([0, 37, 8111, 0, width-17, 53, width, 0], device='cuda', dtype=torch.int32)
                    ends = torch.tensor([width, width-3, width, width, width, 53, width, width], device='cuda', dtype=torch.int32)
                    result = torch.empty(8, k, dtype=torch.int32, device='cuda')
                    bf16_radix_topk(x, starts, ends, result)
                    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
                        expected = torch.full((k,), -1, dtype=torch.int32, device='cuda')
                        count = min(k, end-start)
                        expected[:count] = torch.argsort(x[row, start:end], descending=True, stable=True)[:count].int() + start
                        self.assertTrue(torch.equal(result[row].sort().values, expected.sort().values), (width,k,row))
                    again = torch.empty_like(result)
                    bf16_radix_topk(x, starts, ends, again)
                    self.assertTrue(torch.equal(result, again))


if __name__ == '__main__':
    unittest.main()
