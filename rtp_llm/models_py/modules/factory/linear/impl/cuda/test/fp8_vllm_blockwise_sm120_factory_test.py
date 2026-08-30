import os
import unittest
from unittest import mock

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear import (
    CudaFp8VllmBlockwiseLinear,
)
from rtp_llm.utils.sm120_fp8_backend import SM120_FP8_BACKEND_ENV


class SM120FactoryDiagnosticTest(unittest.TestCase):

    def setUp(self):
        self.backend_env = mock.patch.dict(
            os.environ, {SM120_FP8_BACKEND_ENV: "cutlass"}
        )
        self.backend_env.start()
        self.addCleanup(self.backend_env.stop)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.sm120_blockwise_backend_available",
        return_value=True,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._is_sm120_runtime",
        return_value=True,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear.is_sm12x",
        return_value=True,
    )
    def test_environment_selects_exactly_one_backend(
        self, _is_sm12x, _is_sm120_runtime, _has_cutlass
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        float_scales = torch.ones((1, 128), dtype=torch.float32)
        int_scales = torch.zeros((128, 1), dtype=torch.int32)

        for requested, expect_cutlass in (
            ("auto", False),
            ("deepgemm", False),
            ("cutlass", True),
        ):
            scales = float_scales if expect_cutlass else int_scales
            with self.subTest(requested=requested), mock.patch.dict(
                os.environ, {SM120_FP8_BACKEND_ENV: requested}
            ):
                self.assertEqual(
                    CudaFp8VllmBlockwiseLinear.can_handle(quant_config, weight, scales),
                    expect_cutlass,
                )
                self.assertEqual(
                    CudaFp8GEMMLinear.can_handle(quant_config, weight, scales),
                    not expect_cutlass,
                )

    def test_merge_preserves_physical_blockwise_layout(self):
        K = 4
        physical_a = torch.arange(8).reshape(2, K)
        physical_b = torch.arange(12).reshape(3, K) + 100
        logical_a = physical_a.reshape(K, 2)
        logical_b = physical_b.reshape(K, 3)

        merged = LinearFactory._merge_sm120_blockwise_tensors(
            [logical_a, logical_b], dim=-1
        )

        self.assertEqual((K, 5), merged.shape)
        torch.testing.assert_close(
            merged.reshape(5, K), torch.cat([physical_a, physical_b], dim=0)
        )

    @mock.patch("rtp_llm.models_py.utils.arch.is_sm120", return_value=True)
    def test_create_merged_linear_preserves_sm120_loader_layout(self, _is_sm120):
        K = 128
        physical_a = (torch.arange(128 * K).reshape(128, K) % 32).to(
            torch.float8_e4m3fn
        )
        physical_b = (torch.arange(256 * K).reshape(256, K) % 32).to(
            torch.float8_e4m3fn
        )
        weights = {
            "a": physical_a.reshape(K, 128),
            "b": physical_b.reshape(K, 256),
            "sa": torch.arange(128, dtype=torch.float32).reshape(1, 128),
            "sb": (torch.arange(256, dtype=torch.float32) + 100).reshape(1, 256),
        }

        with mock.patch.object(LinearFactory, "create_linear") as create_linear:
            LinearFactory.create_merged_linear(
                weights=weights,
                weight_keys=["a", "b"],
                scale_keys=["sa", "sb"],
                bias_keys=None,
                scale2_keys=None,
                input_scale_keys=None,
                quant_config=init_quant_config("FP8_PER_BLOCK"),
            )

        merged_weight = create_linear.call_args.kwargs["weight"]
        merged_scales = create_linear.call_args.kwargs["weight_scales"]
        torch.testing.assert_close(
            merged_weight.reshape(384, K), torch.cat([physical_a, physical_b])
        )
        torch.testing.assert_close(
            merged_scales.reshape(384, 1),
            torch.cat([weights["sa"].reshape(128, 1), weights["sb"].reshape(256, 1)]),
        )

    def test_factory_ignores_broken_rejection_diagnostic(self):
        class BrokenDiagnosticStrategy:
            @classmethod
            def can_handle(cls, *args, **kwargs):
                return False

            @classmethod
            def rejection_reason(cls, *args, **kwargs):
                raise RuntimeError("diagnostic failed")

        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)

        with (
            mock.patch.object(LinearFactory, "_strategies", [BrokenDiagnosticStrategy]),
            self.assertRaisesRegex(ValueError, "No suitable Linear strategy"),
        ):
            LinearFactory.create_linear(weight, None, weight_scales, quant_config)

    def test_constructor_rejects_missing_weight_scales(self):
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "requires weight_scales"):
            CudaFp8VllmBlockwiseLinear(weight=weight, weight_scales=None)

    def test_restore_non_square_blockwise_layout(self):
        weight = torch.arange(384 * 256, device="cuda").reshape(384, 256)
        weight_scales = torch.arange(3 * 256, device="cuda").reshape(3, 256)

        restored = CudaFp8VllmBlockwiseLinear._restore_blockwise_weight_layout(
            weight, weight_scales
        )

        restored_weight, restored_scales, K, N, scale_K, scale_N = restored
        self.assertEqual((N, K), restored_weight.shape)
        self.assertEqual((scale_N, scale_K), restored_scales.shape)
        self.assertEqual((K, N, scale_K, scale_N), (384, 256, 3, 256))
        torch.testing.assert_close(restored_weight.flatten(), weight.flatten())
        torch.testing.assert_close(restored_scales.flatten(), weight_scales.flatten())

    def test_restore_rejects_scale_shape_mismatch(self):
        weight = torch.empty((384, 256), device="cuda")
        weight_scales = torch.empty((2, 256), device="cuda")
        with self.assertRaisesRegex(ValueError, "scale dimension mismatch"):
            CudaFp8VllmBlockwiseLinear._restore_blockwise_weight_layout(
                weight, weight_scales
            )

    def test_restore_rejects_non_contiguous_inputs(self):
        weight = torch.empty((256, 384), device="cuda").transpose(0, 1)
        weight_scales = torch.empty((256, 3), device="cuda").transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "weight must be contiguous"):
            CudaFp8VllmBlockwiseLinear._restore_blockwise_weight_layout(
                weight, torch.empty((3, 2), device="cuda")
            )
        with self.assertRaisesRegex(ValueError, "scales must be contiguous"):
            CudaFp8VllmBlockwiseLinear._restore_blockwise_weight_layout(
                torch.empty((384, 256), device="cuda"), weight_scales
            )

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear.has_deep_gemm",
        return_value=False,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear.is_sm12x",
        return_value=False,
    )
    def test_deepgemm_unavailable_keeps_actionable_constructor_error(
        self, _wrapper_is_sm12x, _has_deep_gemm
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)

        self.assertTrue(
            CudaFp8GEMMLinear.can_handle(quant_config, weight, weight_scales)
        )
        with self.assertRaisesRegex(RuntimeError, "install the `deep_gemm` package"):
            CudaFp8DeepGEMMLinear(
                weight=weight,
                weight_scales=weight_scales,
                quant_config=quant_config,
            )

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._is_sm120_runtime",
        return_value=False,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.is_sm12x",
        return_value=True,
    )
    def test_other_sm12x_minor_has_actionable_rejection(
        self, _is_sm12x, _is_sm120_runtime
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)

        supported, reason = CudaFp8VllmBlockwiseLinear.classify_support(
            quant_config, weight, weight_scales
        )

        self.assertFalse(supported)
        self.assertIn("supports exact sm_120 devices only", reason)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._is_sm120_runtime",
        return_value=True,
    )
    def test_factory_reports_int32_scale_rejection(self, _is_sm120_runtime):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.zeros((128, 1), dtype=torch.int32)

        with (
            mock.patch.object(
                LinearFactory,
                "_strategies",
                [CudaFp8VllmBlockwiseLinear],
            ),
            self.assertRaisesRegex(ValueError, "requires float32 weight scales"),
        ):
            LinearFactory.create_linear(weight, None, weight_scales, quant_config)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.sm120_blockwise_backend_available",
        return_value=True,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._is_sm120_runtime",
        return_value=True,
    )
    def test_factory_reports_unaligned_shape_rejection(
        self, _is_sm120_runtime, _has_backend
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 192), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 2), dtype=torch.float32)

        with (
            mock.patch.object(
                LinearFactory,
                "_strategies",
                [CudaFp8VllmBlockwiseLinear],
            ),
            self.assertRaisesRegex(ValueError, "K and N to be multiples of 128"),
        ):
            LinearFactory.create_linear(weight, None, weight_scales, quant_config)

    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear.sm120_blockwise_backend_available",
        return_value=False,
    )
    @mock.patch(
        "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_vllm_blockwise_sm120_linear._is_sm120_runtime",
        return_value=True,
    )
    def test_factory_reports_unavailable_sm120_backend(
        self, _is_sm120_runtime, _has_backend
    ):
        quant_config = init_quant_config("FP8_PER_BLOCK")
        weight = torch.empty((128, 128), dtype=torch.float8_e4m3fn)
        weight_scales = torch.ones((1, 1), dtype=torch.float32)

        self.assertFalse(
            CudaFp8VllmBlockwiseLinear.can_handle(quant_config, weight, weight_scales)
        )
        self.assertIn(
            "backend is unavailable",
            CudaFp8VllmBlockwiseLinear.rejection_reason(
                quant_config, weight, weight_scales
            ),
        )


if __name__ == "__main__":
    unittest.main()
