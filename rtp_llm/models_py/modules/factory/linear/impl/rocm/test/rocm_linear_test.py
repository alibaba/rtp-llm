import os
import itertools
from typing import Optional
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.utils.swizzle_utils import swizzle_tensor
from rtp_llm.ops import HWKernelConfig


class LinearTorch(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.weight = weight.T
        self.bias = bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

class LinearTest(TestCase):

    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    # (k, n) pairs: k is input hidden size, n is output size
    K_N_PAIRS = [
        (512, 512), (512, 256), (512, 1024),
        (768, 768), (768, 384), (768, 1536),
        (1024, 1024), (1024, 512), (1024, 2048),
        (2048, 2048), (2048, 1024), (2048, 4096),
        (4096, 4096), (4096, 2048), (4096, 8192),
        (8192, 8192), (8192, 4096),
        (1280, 3840),  # qkv
        (1280, 1280),  # proj
        (5120, 5120),
        (2048, 5120),
    ]
    HAS_BIAS = [True, False]
    HAS_SWIZZLE = [True, False]
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_linear_test(self, num_tokens: int, k: int, n: int, dtype: _dtype, has_bias: bool, has_swizzle: bool):
        torch.manual_seed(0)
        w = torch.randn(k, n, dtype=dtype)
        torch.nn.init.xavier_uniform_(w)
        if has_bias:
            bias = torch.empty(n, dtype=dtype)
            torch.nn.init.normal_(bias, mean=0.0, std=0.01)
        else:
            bias = None
            
        x = torch.randn(num_tokens, k, dtype=dtype)
        
        linear_torch = LinearTorch(w, bias)
        torch_output = linear_torch(x)
        hw_kernel_config=HWKernelConfig()
        if has_swizzle:
            # Follow aiter's approach: transpose to (n, k), shuffle, then transpose back to (k, n)
            # This matches the format expected by hipb_mm with bpreshuffle=True
            w_swizzled = swizzle_tensor(w.t(), False, MiM=16).t()  # (n, k) swizzled
            w_dict = {"weight": w_swizzled, "bias": bias}
            hw_kernel_config.use_swizzleA = True
        else:
            w_dict = {"weight": w, "bias": bias}
        
        linear = LinearFactory.create_linear_from_weights(
            w_dict, "weight", None, "bias", None, hw_kernel_config
        )
        my_output = linear(x)
        self.assertTrue(torch.allclose(torch_output, my_output, atol=1e-2, rtol=1e-2))

    def test_linear(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.K_N_PAIRS,
            self.DTYPES,
            self.HAS_BIAS,
            self.HAS_SWIZZLE,
        ):
            num_tokens, (k, n), dtype, has_bias, has_swizzle = params
            with self.subTest(
                num_tokens=num_tokens, k=k, n=n, dtype=dtype, has_bias=has_bias, has_swizzle=has_swizzle,
            ):
                self._run_linear_test(num_tokens, k, n, dtype, has_bias, has_swizzle)


if __name__ == "__main__":
    main()