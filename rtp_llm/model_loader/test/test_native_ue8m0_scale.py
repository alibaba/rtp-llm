import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import _transform_scale_ue8m0
from rtp_llm.utils.model_weight import W


class NativeUe8m0ScaleTest(unittest.TestCase):
    def _loader(self, weight_name="weight"):
        loader = object.__new__(PerBlockFp8Weight)
        loader.kernel = SimpleNamespace(name=weight_name)
        loader.scale = SimpleNamespace(name="scale")
        loader.quant_config = Fp8BlockWiseQuantConfig(scale_fmt="ue8m0")
        return loader

    def _config(self, device_type, moe_strategy="auto"):
        return SimpleNamespace(
            exported_device=SimpleNamespace(
                get_device_type=lambda: device_type,
                get_device_id=lambda: 0,
                maybe_rewrite_weight_by_key=lambda key, value, **kwargs: value,
            ),
            use_swizzleA=False,
            moe_strategy=moe_strategy,
        )

    def test_native_scales_load_into_non_packed_linear(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda import (
            fp8_deepgemm_linear,
        )

        loader = self._loader()
        config = self._config(DeviceType.Cpu)
        weight = torch.ones((256, 384)).to(torch.float8_e4m3fn)
        scales = torch.exp2(torch.arange(6, dtype=torch.float32).reshape(2, 3) - 3)
        for dtype, missing_native_dtype in (
            (torch.float8_e8m0fnu, False),
            (torch.float32, False),
            (torch.float32, True),
        ):
            with self.subTest(dtype=dtype, missing=missing_native_dtype), patch.object(
                torch, "float8_e8m0fnu", torch.float8_e8m0fnu
            ), patch.object(
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
                if missing_native_dtype:
                    del torch.float8_e8m0fnu
                result = loader._postprocess({}, "cpu", config)
                linear = fp8_deepgemm_linear.CudaFp8DeepGEMMLinear(
                    result["weight"], result["scale"]
                )
                self.assertEqual(linear.weight_scales.dtype, torch.float32)
                torch.testing.assert_close(linear.weight_scales, scales, rtol=0, atol=0)
                torch.testing.assert_close(
                    linear.weight.view(torch.uint8), weight.view(torch.uint8)
                )

    def test_postprocess_preserves_or_packs_native_ue8m0_for_the_consumer(self):
        for weight_name, strategy, expect_pack in (
            (W.attn_o_w, "auto", True),
            (W.moe_w1, "auto", True),
            (W.moe_w1, "mega_moe_fp8", False),
            (W.moe_w2, "mega_moe_fp8_se", False),
        ):
            with self.subTest(weight=weight_name, strategy=strategy):
                weight = torch.ones((256, 384), device="cuda").to(torch.float8_e4m3fn)
                scales = torch.full((2, 3), 0.125, device="cuda").to(
                    torch.float8_e8m0fnu
                )
                if weight_name in (W.moe_w1, W.moe_w2):
                    weight = weight.unsqueeze(0)
                    scales = scales.unsqueeze(0)
                loader = self._loader(weight_name)
                config = self._config(DeviceType.Cuda, strategy)
                with (
                    patch.object(
                        CompositeWeight,
                        "_postprocess",
                        return_value={weight_name: weight, "scale": scales},
                    ),
                    patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        return_value=True,
                    ),
                ):
                    result = loader._postprocess({}, "cuda", config)
                self.assertIs(result[weight_name], weight)
                if expect_pack:
                    packed = _transform_scale_ue8m0(scales.float(), mn=256)
                    self.assertEqual(result["scale"].dtype, torch.int32)
                    torch.testing.assert_close(result["scale"], packed, rtol=0, atol=0)
                else:
                    self.assertIs(result["scale"], scales)

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
