import itertools
from typing import Optional
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory import LinearFactory


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
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_linear_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, hidden_size // 2, dtype=dtype)
        torch.nn.init.xavier_uniform_(w)
        w_dict = {"weight": w}
        linear = LinearFactory.create_linear_from_weights(
            w_dict, "weight", None, None, None
        )
        linear_torch = LinearTorch(w)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        torch_output = linear_torch(x)
        my_output = linear(x)
        self.assertTrue(torch.allclose(torch_output, my_output, atol=1e-2, rtol=1e-2))

    def test_linear(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_linear_test(*params)


if __name__ == "__main__":
    main()
