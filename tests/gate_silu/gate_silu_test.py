import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels import atex_gate_silu


def torch_gate_silu_ref(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    v = x[..., :half]
    g = x[..., half:]
    return v * torch.sigmoid(g)


class GateSiluCorrectnessTest(unittest.TestCase):

    M = [1, 2, 4, 8, 16, 32, 64, 128]
    N = [32, 64, 128, 256, 512, 1024, 2048]  # N means final D, input is 2*D

    def _run_correctness_check(self, dtype: torch.dtype):
        device = "cuda"

        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {dtype} tests: CUDA not available.")
            return

        for m in self.M:
            for n in self.N:
                x = torch.randn(size=[m, n * 2], device=device, dtype=dtype)
                y = atex_gate_silu(x)
                real = torch_gate_silu_ref(x)

                max_diff = torch.max(torch.abs(real - y)).item()

                self.assertTrue(
                    torch.allclose(real, y, rtol=0.001, atol=0.001),
                    f"Result incorrect for M={m}, N={n}, dtype={dtype}. Max diff: {max_diff:.6f}",
                )

                print(f"Passed: M={m}, N={n}, dtype={dtype}")

    def test_fp16_correctness(self):
        self._run_correctness_check(torch.float16)

    def test_bf16_correctness(self):
        self._run_correctness_check(torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
