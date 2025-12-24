import itertools
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F
from torch import dtype as _dtype

from rtp_llm.models_py.modules import AddBiasResLayerNorm


class AddBiasResLayerNormROCmTorch(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, beta: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight
        self.beta = beta
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, bias: torch.Tensor
    ):
        hidden_states = hidden_states + bias + residual
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(dim=-1, keepdim=True)
        squared_sum = (hidden_states**2).mean(dim=-1, keepdim=True)

        x_normalized = (hidden_states - mean) / torch.sqrt(
            (squared_sum - (mean**2)) + self.variance_epsilon
        )
        return (self.weight * x_normalized + self.beta).to(input_dtype)


class LayerNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [48, 78, 512, 4096]
    HIDDEN_SIZES = [64, 88, 768]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_res_layernorm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        beta = torch.randn(hidden_size, dtype=dtype)
        res_layernorm = AddBiasResLayerNorm(w, beta)
        res_layernorm_torch = AddBiasResLayerNormROCmTorch(w, beta)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
        bias = torch.randn(hidden_size, dtype=dtype)

        # Check which code path will be taken
        uses_layernorm2d = num_tokens > 32 and hidden_size <= 768

        result_torch = res_layernorm_torch(x.clone(), residual.clone(), bias.clone())
        result_rocm = res_layernorm(x.clone(), residual.clone(), bias.clone())

        # Use higher tolerance for bfloat16 due to lower precision
        if dtype == torch.bfloat16:
            atol, rtol = 5e-2, 5e-2
        else:
            atol, rtol = 1e-2, 1e-2

        is_close = torch.allclose(result_torch, result_rocm, atol=atol, rtol=rtol)

        if not is_close:
            diff = (result_torch - result_rocm).abs()
            print(f"\n{'='*70}")
            print(
                f"FAILED: num_tokens={num_tokens}, hidden_size={hidden_size}, dtype={dtype}"
            )
            print(
                f"Code path: {'layernorm2d_fwd' if uses_layernorm2d else 'fused_add_layernorm'}"
            )
            print(f"Tolerance: atol={atol}, rtol={rtol}")
            print(f"Max absolute diff: {diff.max().item():.6f}")
            print(f"Mean absolute diff: {diff.mean().item():.6f}")
            print(f"Median absolute diff: {diff.median().item():.6f}")
            print(f"Result torch [0, :5]: {result_torch[0, :5]}")
            print(f"Result rocm  [0, :5]: {result_rocm[0, :5]}")
            print(f"Diff         [0, :5]: {diff[0, :5]}")

            # Check relative error
            rel_diff = diff / (result_torch.abs() + 1e-8)
            print(f"Max relative diff: {rel_diff.max().item():.6f}")
            print(f"Mean relative diff: {rel_diff.mean().item():.6f}")
            print(f"{'='*70}\n")

        self.assertTrue(is_close)

    def test_res_layernorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_res_layernorm_test(*params)


if __name__ == "__main__":
    main()
