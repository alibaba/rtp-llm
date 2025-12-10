import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedAddRMSNorm, AddRMSNormTorch


class FusedAddRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_add_rmsnorm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)

        weight = torch.randn(hidden_size, dtype=dtype)

        add_rmsnorm_torch = AddRMSNormTorch(weight)
        fused_add_rmsnorm = FusedAddRMSNorm(weight)

        input = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype)

        self.assertTrue(
            torch.allclose(
                add_rmsnorm_torch(input, residual),
                fused_add_rmsnorm(input, residual),
                atol=3e-2,
                rtol=3e-2,
            )
        )

    def test_fused_add_rmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(num_tokens=params[0], hidden_size=params[1], dtype=params[2]):
                self._run_fused_add_rmsnorm_test(*params)


if __name__ == "__main__":
    main()
