import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    use_e8m0_scale_layout,
)
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.utils.model_weight import W


class PerBlockFp8LayoutTest(unittest.TestCase):
    @staticmethod
    def _make_weight(kernel_name, scale_name):
        weight = object.__new__(PerBlockFp8Weight)
        weight.kernel = SimpleNamespace(name=kernel_name)
        weight.scale = SimpleNamespace(name=scale_name)
        return weight

    @staticmethod
    def _load_config():
        return SimpleNamespace(
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda _key, tensor: tensor
            )
        )

    def test_sm12x_dense_keeps_float_scale_layout(self):
        self.assertFalse(
            use_e8m0_scale_layout(
                is_sm12x_device=True,
                is_sm120_device=True,
                deep_gemm_e8m0_enabled=True,
            )
        )

    def test_sm12x_moe_keeps_float_scale_layout(self):
        self.assertFalse(
            use_e8m0_scale_layout(
                is_sm12x_device=True,
                is_sm120_device=True,
                deep_gemm_e8m0_enabled=False,
            )
        )

    def test_non_sm120_sm12x_fails_before_layout_mutation(self):
        with self.assertRaisesRegex(Exception, "other than exact sm_120"):
            use_e8m0_scale_layout(
                is_sm12x_device=True,
                is_sm120_device=False,
                deep_gemm_e8m0_enabled=False,
            )

    def test_non_sm12x_preserves_deep_gemm_layout_choice(self):
        self.assertTrue(
            use_e8m0_scale_layout(
                is_sm12x_device=False,
                is_sm120_device=False,
                deep_gemm_e8m0_enabled=True,
            )
        )
        self.assertFalse(
            use_e8m0_scale_layout(
                is_sm12x_device=False,
                is_sm120_device=False,
                deep_gemm_e8m0_enabled=False,
            )
        )

    @mock.patch("rtp_llm.models_py.utils.arch.is_sm120", return_value=True)
    @mock.patch("rtp_llm.models_py.utils.arch.is_sm12x", return_value=True)
    def test_sm12x_dense_postprocess_keeps_float_scale_physical_layout(
        self, _is_sm12x, _is_sm120
    ):
        weight = self._make_weight(W.ffn_w1, W.ffn_s1)
        physical_weight = torch.arange(256 * 128).reshape(256, 128)
        physical_scale = torch.arange(2).reshape(2, 1).float()
        loaded = {W.ffn_w1: physical_weight, W.ffn_s1: physical_scale}
        fp8_kernel = SimpleNamespace(requant_weight_ue8m0=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper": SimpleNamespace(
                        is_deep_gemm_e8m0_used=mock.Mock(return_value=True)
                    ),
                    "rtp_llm.models_py.kernels.cuda.fp8_kernel": fp8_kernel,
                },
            ),
            mock.patch.object(CompositeWeight, "_postprocess", return_value=loaded),
        ):
            processed = weight._postprocess(None, "cpu", self._load_config())

        fp8_kernel.requant_weight_ue8m0.assert_not_called()
        self.assertEqual((128, 256), processed[W.ffn_w1].shape)
        self.assertEqual((1, 2), processed[W.ffn_s1].shape)
        torch.testing.assert_close(
            processed[W.ffn_w1].flatten(), physical_weight.flatten()
        )
        torch.testing.assert_close(
            processed[W.ffn_s1].flatten(), physical_scale.flatten()
        )

    @unittest.skipUnless(is_sm120(), "requires an sm_120 runtime GPU")
    def test_cpu_conversion_uses_sm120_runtime_layout(self):
        """Host-side conversion must still select the runtime GPU layout."""
        weight = self._make_weight(W.ffn_w1, W.ffn_s1)
        physical_weight = torch.arange(256 * 128).reshape(256, 128)
        physical_scale = torch.arange(2).reshape(2, 1).float()
        loaded = {W.ffn_w1: physical_weight, W.ffn_s1: physical_scale}
        fp8_kernel = SimpleNamespace(requant_weight_ue8m0=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper": SimpleNamespace(
                        is_deep_gemm_e8m0_used=mock.Mock(return_value=True)
                    ),
                    "rtp_llm.models_py.kernels.cuda.fp8_kernel": fp8_kernel,
                },
            ),
            mock.patch.object(CompositeWeight, "_postprocess", return_value=loaded),
        ):
            processed = weight._postprocess(None, "cpu", self._load_config())

        fp8_kernel.requant_weight_ue8m0.assert_not_called()
        self.assertEqual((128, 256), processed[W.ffn_w1].shape)
        self.assertEqual((1, 2), processed[W.ffn_s1].shape)

    @mock.patch("rtp_llm.models_py.utils.arch.is_sm120", return_value=True)
    @mock.patch("rtp_llm.models_py.utils.arch.is_sm12x", return_value=True)
    def test_sm12x_moe_postprocess_keeps_native_layout(self, _is_sm12x, _is_sm120):
        weight = self._make_weight(W.moe_w1, W.moe_s1)
        loaded = {
            W.moe_w1: torch.empty((1, 128, 128)),
            W.moe_s1: torch.empty((1, 1, 1)),
        }
        fp8_kernel = SimpleNamespace(requant_weight_ue8m0=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper": SimpleNamespace(
                        is_deep_gemm_e8m0_used=mock.Mock(return_value=False)
                    ),
                    "rtp_llm.models_py.kernels.cuda.fp8_kernel": fp8_kernel,
                },
            ),
            mock.patch.object(CompositeWeight, "_postprocess", return_value=loaded),
        ):
            processed = weight._postprocess(None, "cpu", self._load_config())

        fp8_kernel.requant_weight_ue8m0.assert_not_called()
        self.assertEqual((1, 128, 128), processed[W.moe_w1].shape)
        self.assertEqual((1, 1, 1), processed[W.moe_s1].shape)


if __name__ == "__main__":
    unittest.main()
