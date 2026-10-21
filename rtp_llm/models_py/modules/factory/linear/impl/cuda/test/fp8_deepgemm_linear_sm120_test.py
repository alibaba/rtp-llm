import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import LoadQuantPerBlockFp8Weight
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    per_block_cast_to_fp8,
    quant_weight_ue8m0_packed,
    requant_weight_ue8m0,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import block_quant_dequant
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_linear_test import (
    CudaFp8GEMMLinearTestBase,
    init_quant_config,
)
from rtp_llm.models_py.utils.arch import is_sm12x


class CudaFp8DeepGEMMLinearSM120Test(CudaFp8GEMMLinearTestBase, unittest.TestCase):
    def test_sm120(self):
        self.assertTrue(is_sm12x())
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_factory_dispatch_matches_other_cuda_arches(self):
        self.assertTrue(
            CudaFp8GEMMLinear.can_handle(
                init_quant_config("FP8_PER_BLOCK"),
                self.weight,
                self.weight_scales,
            )
        )

    def test_direct_weight_quantization_avoids_second_fp8_rounding(self):
        torch.manual_seed(20260811)
        weight = torch.randn((768, 768), device="cuda", dtype=torch.float32)

        direct_weight, direct_scale = quant_weight_ue8m0_packed(weight)
        float_weight, float_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        requant_weight, requant_scale = requant_weight_ue8m0(float_weight, float_scale)

        self.assertEqual(direct_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(direct_scale.dtype, torch.int32)
        self.assertEqual(tuple(direct_scale.shape), (768, 2))
        self.assertEqual(direct_scale.stride(-2), 1)
        self.assertEqual(requant_scale.dtype, torch.int32)

        # Reconstruct through unpacked block scales to compare the two
        # quantization histories without involving a GEMM kernel.
        direct_unpacked, direct_sf = per_block_cast_to_fp8(weight, use_ue8m0=True)
        direct_sf_dequant = block_quant_dequant(
            direct_unpacked, direct_sf, [128, 128], torch.float32
        )
        float_dequant = block_quant_dequant(
            float_weight, float_scale, [128, 128], torch.bfloat16
        )
        requant_unpacked, requant_sf = per_block_cast_to_fp8(
            float_dequant, use_ue8m0=True
        )
        requant_dequant = block_quant_dequant(
            requant_unpacked, requant_sf, [128, 128], torch.float32
        )
        direct_error = (direct_sf_dequant - weight).norm()
        requant_error = (requant_dequant - weight).norm()
        self.assertLess(direct_error, requant_error)
        self.assertTrue(torch.equal(direct_weight, direct_unpacked))

    def test_chunked_direct_quantization_matches_full_matrix(self):
        torch.manual_seed(20260811)
        weight = torch.randn((1152, 384), device="cuda", dtype=torch.float32)

        chunked_weight, _ = quant_weight_ue8m0_packed(weight)
        expected_weight, _ = per_block_cast_to_fp8(weight, use_ue8m0=True)

        # 1152 rows cross the 1024-row chunk boundary while keeping every
        # boundary aligned to the 128-row quantization block.
        self.assertTrue(torch.equal(chunked_weight, expected_weight))

    def test_online_loader_quantizes_non_square_weight_in_kernel_orientation(self):
        torch.manual_seed(20260811)
        source_weight = torch.randn((256, 384), dtype=torch.float32)

        loader = object.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 128
        loader.kernel = Mock()
        loader.kernel.name = "test_dense_weight"
        loader.kernel._load_raw_tensor.return_value = {
            loader.kernel.name: source_weight
        }
        loader.scale = SimpleNamespace(name="test_dense_scale")

        loaded = loader._load_raw_tensor(
            tensor_source=Mock(),
            layer_id=0,
            device="cuda",
            load_config=Mock(),
        )
        expected_weight, expected_scale = quant_weight_ue8m0_packed(
            source_weight.T.contiguous().cuda()
        )

        self.assertEqual(tuple(loaded[loader.kernel.name].shape), (384, 256))
        self.assertTrue(torch.equal(loaded[loader.kernel.name], expected_weight))
        self.assertEqual(loaded[loader.scale.name].dtype, torch.int32)
        self.assertEqual(loaded[loader.scale.name].stride(-2), 1)
        self.assertTrue(torch.equal(loaded[loader.scale.name], expected_scale))

    def test_online_loader_rejects_non_128_group_size(self):
        loader = object.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 64
        loader.kernel = Mock()
        loader.kernel.name = "test_dense_weight"
        loader.kernel._load_raw_tensor.return_value = {
            loader.kernel.name: torch.randn((256, 384), dtype=torch.float32)
        }
        loader.scale = SimpleNamespace(name="test_dense_scale")

        with self.assertRaisesRegex(ValueError, "requires group_size=128"):
            loader._load_raw_tensor(
                tensor_source=Mock(),
                layer_id=0,
                device="cuda",
                load_config=Mock(),
            )

    def test_direct_weight_runs_sm120_deepgemm(self):
        torch.manual_seed(20260811)
        weight = torch.randn((768, 768), device="cuda", dtype=torch.bfloat16)
        inputs = torch.randn((93, 768), device="cuda", dtype=torch.bfloat16)
        weight_fp8, weight_scale = quant_weight_ue8m0_packed(weight)
        linear = CudaFp8DeepGEMMLinear(weight_fp8, weight_scale)
        actual = linear(inputs)
        expected = (inputs.float() @ weight.float().T).bfloat16()
        relative_l2 = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        self.assertLess(relative_l2, 0.06)


class DenseSiluQuantProductionRouteSM120Test(unittest.TestCase):
    """Exercise the actual factory wrapper and DenseMLP consumer on SM120."""

    def setUp(self):
        if not torch.cuda.is_available() or not is_sm12x():
            self.skipTest("SM120 production routing test requires an SM12x GPU")
        from rtp_llm.models_py.modules.factory.linear import LinearFactory
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        self.factory = LinearFactory
        self.ops = rtp_llm_ops
        self.quant_config = init_quant_config("FP8_PER_BLOCK")
        torch.manual_seed(20261015)

    def _linear(self, with_bias=True):
        weight = torch.randn((1024, 3072), device="cuda", dtype=torch.bfloat16) * 0.01
        weight_fp8, scales = quant_weight_ue8m0_packed(weight)
        bias = (
            torch.randn(1024, device="cuda", dtype=torch.bfloat16)
            if with_bias
            else None
        )
        linear = self.factory.create_linear(weight_fp8, bias, scales, self.quant_config)
        self.assertIsInstance(linear, CudaFp8GEMMLinear)
        self.assertIsInstance(linear._deepgemm_linear, CudaFp8DeepGEMMLinear)
        return linear

    def _activation(self, gate_up):
        activated = torch.empty(
            (gate_up.shape[0], gate_up.shape[1] // 2),
            device=gate_up.device,
            dtype=gate_up.dtype,
        )
        self.ops.silu_and_mul(
            activated, gate_up, torch.cuda.current_stream().cuda_stream
        )
        return activated

    def test_factory_wrapper_real_gpu_forward(self):
        linear = self._linear()
        for tokens in (128, 256, 500, 1000):
            gate_up = torch.randn((tokens, 6144), device="cuda", dtype=torch.bfloat16)
            activated = self._activation(gate_up)
            for apply_bias in (False, True):
                expected = (
                    linear(activated)
                    if apply_bias
                    else linear.forward_without_bias(activated)
                )
                with patch.object(
                    linear._deepgemm_linear,
                    "forward_silu_and_mul_quantized",
                    wraps=linear._deepgemm_linear.forward_silu_and_mul_quantized,
                ) as delegate:
                    actual = linear.forward_silu_and_mul_quantized(
                        gate_up, apply_bias=apply_bias
                    )
                delegate.assert_called_once_with(gate_up, apply_bias=apply_bias)
                self.assertIsNotNone(actual)
                self.assertTrue(
                    torch.equal(actual, expected),
                    "Wrapper changed the actual GPU output",
                )

    def test_small_m_preserves_flashinfer_route(self):
        linear = self._linear(with_bias=False)
        # SM120 does not instantiate the SM90-only backend. A presence sentinel
        # exercises the actual dispatch predicate, without claiming a FlashInfer
        # GPU execution on this architecture. The accepted route above is real.
        linear._flashinfer_linear = torch.nn.Identity()
        gate_up = torch.randn((31, 6144), device="cuda", dtype=torch.bfloat16)
        self.assertTrue(linear._should_use_flashinfer(gate_up[:, : linear.K]))
        with patch.object(
            linear._deepgemm_linear,
            "forward_silu_and_mul_quantized",
            wraps=linear._deepgemm_linear.forward_silu_and_mul_quantized,
        ) as delegate:
            self.assertIsNone(linear.forward_silu_and_mul_quantized(gate_up))
        delegate.assert_not_called()
        # At the original threshold, retain the real DeepGEMM route.
        gate_up = torch.randn((32, 6144), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(linear._should_use_flashinfer(gate_up[:, : linear.K]))
        expected = linear(self._activation(gate_up))
        actual = linear.forward_silu_and_mul_quantized(gate_up)
        self.assertIsNotNone(actual)
        self.assertTrue(torch.equal(actual, expected))

    def test_dense_mlp_switch_and_deferred_bias(self):
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
        from rtp_llm.ops import ActivationType, ParallelismConfig
        from rtp_llm.utils.model_weight import W

        up_w, up_s = quant_weight_ue8m0_packed(
            torch.randn((6144, 1024), device="cuda", dtype=torch.bfloat16) * 0.01
        )
        down_w, down_s = quant_weight_ue8m0_packed(
            torch.randn((1024, 3072), device="cuda", dtype=torch.bfloat16) * 0.01
        )
        weights = {
            W.ffn_w13: up_w,
            W.ffn_s13: up_s,
            W.ffn_w2: down_w,
            W.ffn_s2: down_s,
            W.ffn_b2: torch.randn(1024, device="cuda", dtype=torch.bfloat16),
        }
        parallel = ParallelismConfig()
        parallel.tp_size = 1
        parallel.tp_rank = 0
        with patch.dict(os.environ, {"DISABLE_FUSED_ACTIVATION_QUANT": "1"}):
            baseline = DenseMLP(
                ActivationType.Swiglu, parallel, weights, self.quant_config
            )
        with patch.dict(os.environ, {"DISABLE_FUSED_ACTIVATION_QUANT": "0"}):
            candidate = DenseMLP(
                ActivationType.Swiglu, parallel, weights, self.quant_config
            )
            # Changing the environment after construction does not recapture
            # an existing model's graph or change its stored flag.
            self.assertFalse(baseline.enable_fused_activation_quant)
        self.assertTrue(candidate.enable_fused_activation_quant)
        self.assertIsInstance(candidate.down_proj, CudaFp8GEMMLinear)
        for tokens in (128, 256, 500, 1000):
            x = torch.randn((tokens, 1024), device="cuda", dtype=torch.bfloat16)
            expected = baseline(x)
            expected_no_bias, expected_bias = baseline.forward_without_output_bias(x)
            with patch.object(
                candidate.down_proj._deepgemm_linear,
                "forward_silu_and_mul_quantized",
                wraps=candidate.down_proj._deepgemm_linear.forward_silu_and_mul_quantized,
            ) as delegate:
                actual = candidate(x)
                actual_no_bias, actual_bias = candidate.forward_without_output_bias(x)
            self.assertEqual(delegate.call_count, 2)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_no_bias, expected_no_bias))
            self.assertTrue(torch.equal(actual_bias, expected_bias))
            self.assertTrue(torch.equal(actual_no_bias + actual_bias, actual))

    def test_wrapper_graph_replay_new_input(self):
        linear = self._linear()
        for tokens in (128, 256, 512, 1024):
            with self.subTest(tokens=tokens):
                gate_up = torch.randn(
                    (tokens, 6144), device="cuda", dtype=torch.bfloat16
                )
                linear.forward_silu_and_mul_quantized(gate_up)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = linear.forward_silu_and_mul_quantized(gate_up)
                for seed in (17, 73, 101):
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    replacement = torch.randn(
                        gate_up.shape,
                        device="cuda",
                        dtype=torch.bfloat16,
                        generator=generator,
                    )
                    expected = linear(self._activation(replacement))
                    gate_up.copy_(replacement)
                    graph.replay()
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
