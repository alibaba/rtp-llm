import os
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
    MX_BLOCK,
    _pack_flashinfer_mxfp8_scale,
    mxfp8_quant_act_packed_fused,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
    CudaMxfp8Linear,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

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


class Mxfp8QuantActPackedTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        if torch.cuda.get_device_capability()[0] < 10:
            raise unittest.SkipTest("native MXFP8 quantization requires SM100")

    def _check_exact(self, x: torch.Tensor) -> None:
        import flashinfer

        ref_q, ref_scale_u8 = flashinfer.mxfp8_quantize(
            x,
            is_sf_swizzled_layout=False,
            alignment=MX_BLOCK,
            backend="cute-dsl",
        )
        ref_scale = _pack_flashinfer_mxfp8_scale(
            ref_scale_u8, x.shape[0], x.shape[1]
        )
        actual_q, actual_scale = mxfp8_quant_act_packed_fused(x)

        # Compare storage bytes, not dequantized values.  This catches both
        # an FP8 rounding difference and any UE8M0/layout mismatch.
        self.assertTrue(
            torch.equal(ref_q.view(torch.uint8), actual_q.view(torch.uint8))
        )
        self.assertEqual(ref_scale.stride(), actual_scale.stride())
        self.assertTrue(torch.equal(ref_scale, actual_scale))

    def test_random_is_bitwise_equal_to_flashinfer(self):
        for dtype in (torch.bfloat16, torch.float16):
            for m, k in ((1, 128), (4, 6144), (32, 6144)):
                with self.subTest(dtype=dtype, m=m, k=k):
                    torch.manual_seed(20260903 + m + k)
                    x = torch.randn(m, k, dtype=dtype, device="cuda")
                    self._check_exact(x)

    def test_zero_and_power_of_two_boundaries_are_bitwise_equal(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.zeros(4, 256, dtype=dtype, device="cuda")
                # Make every 32-value group exercise a different scale while
                # keeping explicit all-zero groups for UE8M0 byte 0.
                values = torch.tensor(
                    [0.0, 2.0**-20, 2.0**-8, 1.0, 7.5, 448.0, 896.0, 4096.0],
                    dtype=dtype,
                    device="cuda",
                )
                x[:, ::MX_BLOCK] = values
                x[:, 1::MX_BLOCK] = -values
                self._check_exact(x)


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

    def test_dense_mlp_reuses_external_mxfp8_input(self):
        mlp = DenseMLP.__new__(DenseMLP)
        torch.nn.Module.__init__(mlp)
        mlp.up_proj = self.linear
        mlp.down_proj = torch.nn.Identity()
        mlp.act_fn = torch.nn.Identity()
        mlp._fuse_silu_quant = False
        mlp.parallelism_config = type(
            "Parallelism", (), {"get_ffn_tp_size": lambda self: 1}
        )()

        self.assertFalse(mlp.accepts_fp8_input)
        self.assertTrue(mlp.accepts_mxfp8_input)

        x = torch.randn(4, self.K, dtype=torch.bfloat16, device=self.device)
        x_q, x_scale = mxfp8_quant_act_packed_fused(x)
        expected = mlp(x)
        actual = mlp(x, x_fp8=x_q, x_scale=x_scale)
        self.assertTrue(torch.equal(actual, expected))

    def test_quant_and_linear_cuda_graph_replay(self):
        static_x = torch.randn(
            4, self.K, dtype=torch.bfloat16, device=self.device
        )
        replay_x = torch.randn_like(static_x)

        # Compile Triton/DeepGEMM and materialize the lazily packed weight
        # scale before capture.
        for _ in range(3):
            q, scale = mxfp8_quant_act_packed_fused(static_x)
            self.linear(q, input_scales=scale)
        ref_q, ref_scale = mxfp8_quant_act_packed_fused(replay_x)
        expected = self.linear(ref_q, input_scales=ref_scale)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_q, graph_scale = mxfp8_quant_act_packed_fused(static_x)
            graph_output = self.linear(graph_q, input_scales=graph_scale)

        static_x.copy_(replay_x)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(graph_output, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
