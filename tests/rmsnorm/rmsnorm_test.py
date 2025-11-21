import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels import atex_rmsnorm, atex_skiprmsnorm


class RmsNormCorrectnessTest(unittest.TestCase):
    """
    A unittest class to check the correctness of the atex_rmsnorm kernel
    against torch.nn.functional.rms_norm for various sizes and dtypes.
    """

    M = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    N = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
    EPS = 1e-9

    def _run_correctness_check(self, dtype: torch.dtype):
        device = "cuda"

        # 确保CUDA可用
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {dtype} tests: CUDA not available.")
            return

        for m in self.M:
            for n in self.N:
                x = torch.randn(size=[m, n], device=device, dtype=dtype)
                w = torch.randn(size=[n], device=device, dtype=dtype)
                o = atex_rmsnorm(x, w, eps=self.EPS, normailize_shape=n)
                real = F.rms_norm(x, [n], w, eps=self.EPS)

                max_diff = torch.max(torch.abs(real - o)).item()

                self.assertTrue(
                    torch.allclose(real, o, rtol=0.01, atol=0.01),
                    f"Result incorrect for M={m}, N={n}, dtype={dtype}. Max absolute difference: {max_diff:.6f}",
                )

                print(f"Passed: M={m}, N={n}, dtype={dtype}")

    def test_float16_correctness(self):
        self._run_correctness_check(torch.float16)

    def test_bfloat16_correctness(self):
        self._run_correctness_check(torch.bfloat16)


class RmsNormCorrectnessTest(unittest.TestCase):
    """
    A unittest class to check the correctness of the atex_rmsnorm kernel
    against torch.nn.functional.rms_norm for various sizes and dtypes.
    """

    M = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
    N = [256, 512, 768, 1024, 1536, 2048, 4096, 8192, 16384]
    EPS = 1e-9

    def _run_correctness_check(self, dtype: torch.dtype):
        device = "cuda"

        # 确保CUDA可用
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {dtype} tests: CUDA not available.")
            return

        for m in self.M:
            for n in self.N:
                x = torch.randn(size=[m, n], device=device, dtype=dtype)
                r = torch.randn(size=[m, n], device=device, dtype=dtype)
                w = torch.randn(size=[n], device=device, dtype=dtype)
                o1, o2 = atex_skiprmsnorm(x, r, w, eps=self.EPS, normailize_shape=n)
                real1 = F.rms_norm(x + r, [n], w, eps=self.EPS)
                real2 = x + r

                self.assertTrue(
                    torch.allclose(real1, o1, rtol=0.01, atol=0.01),
                    f"Result incorrect for M={m}, N={n}, dtype={dtype}.",
                )

                self.assertTrue(
                    torch.allclose(real2, o2, rtol=0.01, atol=0.01),
                    f"Result incorrect for M={m}, N={n}, dtype={dtype}.",
                )

                print(f"Passed: M={m}, N={n}, dtype={dtype}")

    def test_float16_correctness(self):
        self._run_correctness_check(torch.float16)

    def test_bfloat16_correctness(self):
        self._run_correctness_check(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
