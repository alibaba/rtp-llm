import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import MX_BLOCK
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
)

# DeepGEMM's default relative JIT path is not stable under Bazel's launcher.
# Set an absolute cache before the first lazy DeepGEMM import/JIT invocation.
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(os.environ.get("TEST_TMPDIR", "/tmp"), "deep_gemm_cache"),
)


def _ue8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale.float() - 127.0)


def _dequantize(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    blocked = value.float().reshape(*value.shape[:-1], -1, MX_BLOCK)
    row_major_scale = scale.float().reshape(*value.shape[:-1], -1)
    return (blocked * row_major_scale.unsqueeze(-1)).reshape(value.shape)


class CudaMxfp8LinearTest(unittest.TestCase):
    K = 6144
    N = 256

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("native MXFP8 Linear requires SM100")

        import flashinfer

        torch.manual_seed(2026)
        cls.device = "cuda"
        weight = torch.randn(cls.N, cls.K, dtype=torch.bfloat16, device=cls.device)
        cls.weight_q, weight_scale_u8 = flashinfer.mxfp8_quantize(
            weight,
            is_sf_swizzled_layout=False,
            alignment=MX_BLOCK,
            backend="cute-dsl",
        )
        cls.weight_scale = _ue8m0_to_fp32(weight_scale_u8).reshape(
            cls.N, cls.K // MX_BLOCK
        )
        cls.weight_dequant = _dequantize(cls.weight_q, cls.weight_scale)
        cls.linear = CudaMxfp8Linear(cls.weight_q, cls.weight_scale)

    def _run(self, m: int) -> None:
        import flashinfer

        x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
        x_q, x_scale_u8 = flashinfer.mxfp8_quantize(
            x,
            is_sf_swizzled_layout=False,
            alignment=MX_BLOCK,
            backend="cute-dsl",
        )
        x_scale = _ue8m0_to_fp32(x_scale_u8).reshape(m, self.K // MX_BLOCK)

        actual_internal = self.linear(x)
        actual_external = self.linear(x_q, input_scales=x_scale)
        self.assertTrue(torch.equal(actual_internal, actual_external))

        # This is the same mathematical contract used by vLLM's ModelOpt
        # MXFP8 Linear: dynamic 1x32 activation quantization, checkpoint 1x32
        # weight scales, then x @ weight.T.  The native MXFP8 GEMM may differ
        # slightly from a dequantized FP32 accumulation, so compare both max
        # absolute error and relative L2 rather than requiring bit identity.
        reference = torch.matmul(
            _dequantize(x_q, x_scale), self.weight_dequant.t()
        ).bfloat16()
        error = (actual_internal.float() - reference.float()).abs()
        relative_l2 = error.norm() / reference.float().norm().clamp_min(1e-12)
        self.assertLess(float(error.max()), 2.0)
        self.assertLess(float(relative_l2), 0.01)

    def test_hy4_hidden_size_grid(self):
        for m in (1, 8, 32):
            with self.subTest(m=m):
                self._run(m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
