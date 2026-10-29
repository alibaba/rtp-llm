import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0


class NativeUe8m0ScaleTest(unittest.TestCase):
    def test_native_scales_load_into_non_packed_linear(self):
        from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
        from rtp_llm.model_loader.weight_module import CompositeWeight
        from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
            fp8_deepgemm_linear,
        )

        loader = object.__new__(PerBlockFp8Weight)
        loader.kernel = SimpleNamespace(name="weight")
        loader.scale = SimpleNamespace(name="scale")
        config = SimpleNamespace(
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda key, value, **kwargs: value
            ),
            use_swizzleA=False,
            moe_strategy="auto",
        )
        weight = torch.ones((256, 384)).to(torch.float8_e4m3fn)
        scales = torch.exp2(torch.arange(6, dtype=torch.float32).reshape(2, 3) - 3)
        for dtype in (torch.float8_e8m0fnu, torch.float32):
            with self.subTest(dtype=dtype), patch.object(
                CompositeWeight,
                "_postprocess",
                return_value={"weight": weight, "scale": scales.to(dtype)},
            ), patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=False,
            ), patch.object(
                fp8_deepgemm_linear, "is_deep_gemm_e8m0_used", return_value=False
            ), patch.object(
                fp8_deepgemm_linear, "has_deep_gemm", return_value=True
            ):
                result = loader._postprocess({}, "cpu", config)
                linear = fp8_deepgemm_linear.CudaFp8DeepGEMMLinear(
                    result["weight"], result["scale"]
                )
                self.assertEqual(linear.weight_scales.dtype, torch.float32)
                torch.testing.assert_close(linear.weight_scales, scales, rtol=0, atol=0)
                torch.testing.assert_close(
                    linear.weight.view(torch.uint8), weight.view(torch.uint8)
                )

    def test_checkpoint_scales_pack_identically_to_float_scales(self):
        self.assertTrue(torch.cuda.is_available())
        exponents = torch.tensor(
            [[-120, -1, 0, 120], [-100, -10, 10, 100]], device="cuda"
        )
        values = torch.exp2(exponents.float())
        native = values.to(torch.float8_e8m0fnu)
        torch.testing.assert_close(native.float(), values, rtol=0, atol=0)
        for rows in (129, 256):
            expected = _transform_scale_ue8m0(values, rows)
            actual = _transform_scale_ue8m0(native, rows)
            self.assertEqual(actual.dtype, torch.int32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
