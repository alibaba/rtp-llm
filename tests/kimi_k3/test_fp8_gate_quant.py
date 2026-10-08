"""FP8 gate producer must preserve the existing grouped quantization contract."""

import pytest
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fused_activation import (
    sigmoid_mul_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.kimi_k3.native_mla_ops import gate_sigmoid_mul
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)


@pytest.mark.parametrize("rows", [1, 8, 4096, 65536])
def test_sigmoid_mul_quant_matches_existing_path(rows):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(715)
    x = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    expected = sgl_per_token_group_quant_fp8(
        gate_sigmoid_mul(x, gate),
        128,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    actual = sigmoid_mul_per_token_group_quant_fp8(x, gate)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


def test_sigmoid_mul_quant_rejects_incompatible_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.empty((1, 129), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="divisible by 128"):
        sigmoid_mul_per_token_group_quant_fp8(x, x)
    x = torch.empty((1, 128), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="BF16"):
        sigmoid_mul_per_token_group_quant_fp8(x, x)


def test_fp8_wrapper_accepts_explicit_quantized_activation():
    class Backend(torch.nn.Module):
        def forward_quantized(self, values, scales, out=None):
            assert values.dtype == torch.float8_e4m3fn
            assert scales.dtype == torch.int32
            assert out is None
            return torch.full((values.shape[0], 4), 3, dtype=torch.bfloat16)

    wrapper = CudaFp8GEMMLinear.__new__(CudaFp8GEMMLinear)
    torch.nn.Module.__init__(wrapper)
    wrapper._deepgemm_linear = Backend()
    values = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
    scales = torch.empty((2, 1), dtype=torch.int32)
    output = wrapper.forward_quantized(values, scales)
    torch.testing.assert_close(output, torch.full((2, 4), 3, dtype=torch.bfloat16))
