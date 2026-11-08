"""SM120 per-block FP8 scale-format and output-major layout coverage.

Sibling of test_inline_fp8_quant.py, which is left unchanged. Covers:
  - LoadQuantPerBlockFp8Weight chooses packed int32 scales on the SM120
    inference device vs floating-point block scales elsewhere
  - output-major FP8 layout preservation for rectangular / merged /
    transpose / transpose-pad descriptors
"""

import functools
import unittest
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_type import DeviceType
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    per_block_cast_to_fp8,
)
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models.llama_weight import merge_qkv_hf as llama_merge_qkv_hf
from rtp_llm.utils.model_weight import CkptWeightInfo, W, transpose, transpose_pad


class TestPerBlockOnlineSm120Layout(unittest.TestCase):
    def setUp(self):
        # DeepGEMM MoE uses packed scales on the same SM120 device where
        # dense CUTLASS consumes floating-point block scales.
        patcher = patch(
            "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
            return_value=True,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _mock_loader(self, raw):
        loader = LoadQuantPerBlockFp8Weight.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 128
        loader.kernel = MagicMock()
        loader.kernel.name = W.attn_o_w
        loader.kernel.need_transpose = False
        loader.scale = MagicMock()
        loader.scale.name = W.attn_o_s
        loader.scale.need_transpose = False
        loader.kernel._load_raw_tensor.return_value = {W.attn_o_w: raw}
        return loader

    def _load_descriptor(self, descriptor, tensors):
        loader = LoadQuantPerBlockFp8Weight(
            descriptor,
            Fp8BlockWiseQuantConfig(is_quanted=False),
            name=descriptor.name,
        )
        source = MagicMock()
        source.load_tensor.side_effect = lambda name, dtype: [tensors[name]]
        load_config = MagicMock(compute_dtype=torch.float32)
        with patch(
            "rtp_llm.model_loader.per_block_fp8_quant_weight."
            "_preserve_output_major_fp8_layout",
            return_value=True,
        ):
            loaded = loader._load_raw_tensor(
                source, layer_id=0, device="cpu", load_config=load_config
            )
        return loader, source, loaded

    def test_raw_and_postprocess_select_scale_format_for_inference_device(self):
        for current, target in ((0, 1), (1, 0)):
            with self.subTest(current=current, target=target):
                raw = torch.ones(128, 128)
                loader = self._mock_loader(raw)
                config = MagicMock()
                config.exported_device.get_device_type.return_value = DeviceType.Cuda
                config.exported_device.get_device_id.return_value = str(target)
                config.exported_device.maybe_rewrite_weight_by_key.side_effect = (
                    lambda key, tensor, **kwargs: tensor
                )
                packed = (
                    raw.to(torch.float8_e4m3fn),
                    torch.ones(128, 1, dtype=torch.int32),
                )
                with (
                    patch("torch.cuda.current_device", return_value=current),
                    patch(
                        "rtp_llm.model_loader.per_block_fp8_quant_weight._preserve_output_major_fp8_layout",
                        return_value=False,
                    ),
                    patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        side_effect=lambda device=None: (
                            torch.cuda.current_device() if device is None else device
                        )
                        == 0,
                    ) as scale_policy,
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
                        return_value=packed,
                    ) as quantize_packed,
                ):
                    loaded = loader._load_raw_tensor(MagicMock(), 0, "cpu", config)
                    with patch.object(
                        CompositeWeight, "_postprocess", return_value=loaded.copy()
                    ):
                        processed = loader._postprocess(loaded, "cpu", config)
                self.assertEqual(
                    [call.args for call in scale_policy.call_args_list],
                    [(target,), (target,)],
                )
                self.assertEqual(quantize_packed.call_count, int(target == 0))
                self.assertEqual(
                    processed[W.attn_o_s].dtype == torch.int32, target == 0
                )

    def test_moe_weights_keep_packed_deepgemm_scales(self):
        for kernel_name, scale_name in (
            (W.moe_w1, W.moe_s1),
            (W.moe_w2, W.moe_s2),
        ):
            with self.subTest(kernel=kernel_name):
                raw = torch.linspace(-1.0, 1.0, 128 * 256).reshape(128, 256)
                loader = self._mock_loader(raw)
                loader.kernel.name = kernel_name
                loader.scale.name = scale_name
                loader.kernel._load_raw_tensor.return_value = {kernel_name: raw}
                config = MagicMock()
                config.exported_device.maybe_rewrite_weight_by_key.side_effect = (
                    lambda key, tensor, **kwargs: tensor
                )
                packed_scale = torch.ones(2, 1, dtype=torch.int32)
                with (
                    patch(
                        "rtp_llm.model_loader.per_block_fp8_quant_weight._preserve_output_major_fp8_layout",
                        return_value=True,
                    ) as preserve,
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.quant_weight_ue8m0_packed",
                    ) as quantize_packed,
                    patch(
                        "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
                        side_effect=lambda weight, scale: (weight, packed_scale),
                    ) as requant,
                ):
                    loaded = loader._load_raw_tensor(MagicMock(), 0, "cpu", config)
                    with patch.object(
                        CompositeWeight, "_postprocess", return_value=loaded.copy()
                    ):
                        processed = loader._postprocess(loaded, "cpu", config)
                preserve.assert_not_called()
                quantize_packed.assert_not_called()
                requant.assert_called_once()
                self.assertNotEqual(loaded[scale_name].dtype, torch.int32)
                self.assertEqual(processed[scale_name].dtype, torch.int32)

    def test_rectangular_input_major_weight_becomes_output_major(self):
        raw_weight = torch.linspace(-1.0, 1.0, 256 * 384).reshape(384, 256)
        loader = self._mock_loader(raw_weight)
        expected_weight, expected_scale = per_block_cast_to_fp8(
            raw_weight, loader.group_size
        )

        with patch(
            "rtp_llm.model_loader.per_block_fp8_quant_weight."
            "_preserve_output_major_fp8_layout",
            return_value=True,
        ):
            loaded = loader._load_raw_tensor(
                MagicMock(), layer_id=0, device="cpu", load_config=MagicMock()
            )

        self.assertEqual(loaded[W.attn_o_w].shape, (256, 384))
        self.assertEqual(loaded[W.attn_o_s].shape, (2, 3))
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight.T))
        torch.testing.assert_close(loaded[W.attn_o_s], expected_scale.T)

    def test_real_qkv_merge_descriptor_restores_output_major_layout(self):
        tensors = {
            "q": torch.linspace(-1.0, 1.0, 256 * 384).reshape(256, 384),
            "k": torch.linspace(-0.7, 0.9, 128 * 384).reshape(128, 384),
            "v": torch.linspace(-0.4, 0.3, 128 * 384).reshape(128, 384),
        }
        descriptor = AttnAtomicWeight(
            W.attn_qkv_w,
            [CkptWeightInfo(key) for key in tensors],
            functools.partial(
                llama_merge_qkv_hf, hidden_size=384, head_num_kv=1, head_num=2
            ),
        )
        loader, tensor_source, loaded = self._load_descriptor(descriptor, tensors)
        expected_weight, expected_scale = per_block_cast_to_fp8(
            torch.cat(list(tensors.values())), loader.group_size
        )
        self.assertEqual(loaded[W.attn_qkv_w].shape, (512, 384))
        self.assertEqual(loaded[W.attn_qkv_s].shape, (4, 3))
        torch.testing.assert_close(
            loaded[W.attn_qkv_w].float(), expected_weight.float(), rtol=0, atol=0
        )
        torch.testing.assert_close(loaded[W.attn_qkv_s], expected_scale, rtol=0, atol=0)

    def test_real_transpose_descriptor_restores_output_major_layout(self):
        raw_weight = torch.linspace(-1.0, 1.0, 256 * 384).reshape(256, 384)
        descriptor = AttnAtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("layers.{i}.self_attn.o_proj.weight")],
            transpose,
        )
        loader, tensor_source, loaded = self._load_descriptor(
            descriptor, {"layers.0.self_attn.o_proj.weight": raw_weight}
        )
        expected_weight, expected_scale = per_block_cast_to_fp8(
            raw_weight, loader.group_size
        )

        tensor_source.load_tensor.assert_called_once_with(
            "layers.0.self_attn.o_proj.weight", torch.float32
        )
        self.assertEqual(loaded[W.attn_o_w].shape, (256, 384))
        self.assertEqual(loaded[W.attn_o_s].shape, (2, 3))
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight))
        torch.testing.assert_close(loaded[W.attn_o_s], expected_scale)

    def test_real_transpose_pad_descriptor_quantizes_non_aligned_tensor(self):
        raw_weight = torch.linspace(-1.0, 1.0, 130 * 257).reshape(130, 257)
        descriptor = AttnAtomicWeight(
            W.attn_o_w,
            [CkptWeightInfo("layers.{i}.self_attn.o_proj.weight")],
            functools.partial(transpose_pad, align_size=128, dim=0),
        )
        loader, tensor_source, loaded = self._load_descriptor(
            descriptor, {"layers.0.self_attn.o_proj.weight": raw_weight}
        )

        processed = transpose_pad([raw_weight], align_size=128, dim=0)
        expected_weight, expected_scale = per_block_cast_to_fp8(
            processed, loader.group_size
        )
        expected_weight = expected_weight.T.contiguous()
        expected_scale = expected_scale.reshape(
            expected_scale.shape[0], -1
        ).T.contiguous()

        self.assertTrue(loader.kernel.need_transpose)
        self.assertEqual(loaded[W.attn_o_w].shape, (256, 257))
        self.assertEqual(loaded[W.attn_o_s].shape, (2, 3))
        self.assertTrue(torch.equal(loaded[W.attn_o_w], expected_weight))
        torch.testing.assert_close(loaded[W.attn_o_s], expected_scale)


if __name__ == "__main__":
    unittest.main()
