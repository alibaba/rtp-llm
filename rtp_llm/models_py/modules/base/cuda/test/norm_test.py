import itertools
from unittest import SkipTest, TestCase, main

import torch
from rtp_llm.models_py.modules import RMSNorm, RMSNormTorch
from torch import dtype as _dtype


class NormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        rms_norm = RMSNorm(w)
        rms_norm_torch = RMSNormTorch(w)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        self.assertTrue(
            torch.allclose(rms_norm_torch(x), rms_norm(x), atol=1e-2, rtol=1e-2)
        )

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_rms_norm_test(*params)

    def test_rms_norm_large_row_offsets(self):
        # Exercise the actual >2**32 element address boundary, without
        # constructing a full FP32 reference of the 8 GiB activation.
        if torch.cuda.mem_get_info()[0] < 20 * 1024**3:
            raise SkipTest("large RMSNorm address test requires 20 GiB free HBM")
        rows, width = (1 << 20) + 3, 4096
        sample_rows = [0, (1 << 19), (1 << 20) - 1, 1 << 20, rows - 1]
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                x = torch.ones((rows, width), dtype=dtype)
                x[-1, : width // 2] = 2
                output = torch.full_like(x, float("nan"))
                norm = RMSNorm(torch.ones(width, dtype=dtype))
                self.assertIs(norm(x, output), output)
                reference = x[sample_rows].float()
                reference *= torch.rsqrt(
                    reference.square().mean(-1, keepdim=True) + norm.variance_epsilon
                )
                torch.testing.assert_close(
                    output[sample_rows].float(), reference, atol=1e-2, rtol=1e-2
                )
                # Capture both launches and change the tail after capture.
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    norm(x, output)
                x[-1].fill_(3)
                graph.replay()
                torch.testing.assert_close(
                    output[-1], torch.ones_like(output[-1]), atol=1e-2, rtol=1e-2
                )
                del graph, x, output, norm
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
