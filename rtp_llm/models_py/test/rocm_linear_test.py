import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import Linear, LinearTorch


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_.view(x_type)


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
    ]
    HAS_BIAS = [True, False]
    SWIZZLE = [True, False]
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_linear_test(self, num_tokens: int, k: int, n: int, dtype: _dtype, has_bias: bool, swizzle: bool):
        torch.manual_seed(0)
        w = torch.randn(k, n, dtype=dtype)
        torch.nn.init.xavier_uniform_(w)
        if has_bias:
            bias = torch.empty(n, dtype=dtype)
            torch.nn.init.normal_(bias, mean=0.0, std=0.01)
        else:
            bias = None
        linear_torch = LinearTorch(w, bias)
        x = torch.randn(num_tokens, k, dtype=dtype)
        torch_output = linear_torch(x)
        if swizzle:
            # Follow aiter's approach: transpose to (n, k), shuffle, then transpose back to (k, n)
            # This matches the format expected by hipb_mm with bpreshuffle=True
            w_t = w.t()  # (n, k)
            w_t_shuffled = shuffle_weight(w_t, layout=(16, 16), use_int4=False)  # (n, k) shuffled
            w_shuffled = w_t_shuffled.t()  # (k, n) - transpose back for hipb_mm
            linear = Linear(w_shuffled, bias, bpreshuffle=True)
        else:
            linear = Linear(w, bias, bpreshuffle=False)
        my_output = linear(x)
        self.assertTrue(torch.allclose(torch_output, my_output, atol=1e-2, rtol=1e-2))

    def test_linear(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.K_N_PAIRS,
            self.DTYPES,
            self.HAS_BIAS,
            self.SWIZZLE,
        ):
            num_tokens, (k, n), dtype, has_bias, swizzle = params
            with self.subTest(
                num_tokens=num_tokens, k=k, n=n, dtype=dtype, has_bias=has_bias, swizzle=swizzle
            ):
                self._run_linear_test(num_tokens, k, n, dtype, has_bias, swizzle)


if __name__ == "__main__":
    main()
