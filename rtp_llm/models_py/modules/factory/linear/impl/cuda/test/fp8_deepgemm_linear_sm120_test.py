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
    pack_weight_scale_ue8m0,
    per_block_cast_to_fp8,
    quant_weight_ue8m0,
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
from rtp_llm.test.utils.numeric_util import calc_diff


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

        direct_weight, direct_unpacked_scale = quant_weight_ue8m0(weight, [128, 128])
        direct_scale = pack_weight_scale_ue8m0(direct_unpacked_scale, weight.shape[0])
        float_weight, float_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        _, requant_scale = requant_weight_ue8m0(float_weight, float_scale)

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

    def test_online_loader_quantizes_non_square_weight_in_kernel_orientation(self):
        torch.manual_seed(20260811)
        source_weight = torch.randn((256, 512), dtype=torch.float32)

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
        expected_weight, expected_scale = quant_weight_ue8m0(
            source_weight.T.contiguous().cuda(), [128, 128]
        )

        self.assertEqual(tuple(loaded[loader.kernel.name].shape), (512, 256))
        self.assertTrue(torch.equal(loaded[loader.kernel.name], expected_weight))
        self.assertEqual(loaded[loader.scale.name].dtype, torch.float32)
        self.assertTrue(loaded[loader.scale.name].is_contiguous())
        self.assertTrue(torch.equal(loaded[loader.scale.name], expected_scale))

    def test_online_loader_packs_ue8m0_scale_after_tp_split(self):
        torch.manual_seed(20260811)
        source_weight = torch.randn((256, 512), dtype=torch.float32)

        loader = object.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 128
        loader.kernel = Mock()
        loader.kernel.name = "test_dense_weight"
        loader.kernel._load_raw_tensor.return_value = {
            loader.kernel.name: source_weight
        }
        loader.scale = Mock()
        loader.scale.name = "test_dense_scale"
        raw = loader._load_raw_tensor(Mock(), 0, "cuda", Mock())
        # Model TP splits N and its matching scale rows before _postprocess.
        local_weight = raw[loader.kernel.name][:256]
        local_scale = raw[loader.scale.name][:2]
        exported_device = SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda _, tensor: tensor
        )

        with patch(
            "rtp_llm.model_loader.weight_module.CompositeWeight._postprocess",
            return_value={
                loader.kernel.name: local_weight,
                loader.scale.name: local_scale,
            },
        ):
            processed = loader._postprocess(
                {
                    loader.kernel.name: local_weight,
                    loader.scale.name: local_scale,
                },
                "cuda",
                SimpleNamespace(exported_device=exported_device),
            )
        expected_scale = pack_weight_scale_ue8m0(local_scale, 256)

        self.assertTrue(torch.equal(processed[loader.kernel.name], local_weight))
        self.assertEqual(processed[loader.scale.name].dtype, torch.int32)
        self.assertEqual(processed[loader.scale.name].stride(-2), 1)
        self.assertTrue(torch.equal(processed[loader.scale.name], expected_scale))

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
        weight_fp8, unpacked_scale = quant_weight_ue8m0(weight, [128, 128])
        weight_scale = pack_weight_scale_ue8m0(unpacked_scale, weight.shape[0])
        linear = CudaFp8DeepGEMMLinear(weight_fp8, weight_scale)
        actual = linear(inputs)
        expected = (inputs.float() @ weight.float().T).bfloat16()
        self.assertLess(calc_diff(actual, expected), 0.0011)

    def test_non_square_direct_and_legacy_weight_gemm_are_close(self):
        torch.manual_seed(20260901)
        weight = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
        inputs = torch.randn((93, 512), device="cuda", dtype=torch.bfloat16)

        direct_weight, direct_unpacked_scale = quant_weight_ue8m0(weight, [128, 128])
        direct_scale = pack_weight_scale_ue8m0(direct_unpacked_scale, weight.shape[0])
        float_weight, float_scale = per_block_cast_to_fp8(weight, use_ue8m0=False)
        legacy_weight, legacy_scale = requant_weight_ue8m0(float_weight, float_scale)

        direct_output = CudaFp8DeepGEMMLinear(direct_weight, direct_scale)(inputs)
        legacy_output = CudaFp8DeepGEMMLinear(legacy_weight, legacy_scale)(inputs)
        reference = (inputs.float() @ weight.float().T).bfloat16()

        self.assertLess(calc_diff(direct_output, legacy_output), 0.0011)
        self.assertLess(calc_diff(direct_output, reference), 0.0011)


if __name__ == "__main__":
    unittest.main()
