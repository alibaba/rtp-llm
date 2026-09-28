"""CPU contracts for FlashInfer's canonical MoE FP8 scale sidecars."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    _flashinfer_raw_scale_key,
)
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.utils.model_weight import W


class FlashInferMoeRawScaleTest(unittest.TestCase):
    def _weight(self, name=W.moe_w1, group_size=128):
        weight = object.__new__(PerBlockFp8Weight)
        weight.kernel = SimpleNamespace(name=name)
        weight.scale = SimpleNamespace(name=W.moe_s1 if name == W.moe_w1 else W.moe_s2)
        weight.group_size = group_size
        return weight

    @staticmethod
    def _load_config():
        return SimpleNamespace(
            exported_device=SimpleNamespace(
                maybe_rewrite_weight_by_key=lambda _key, value, **_kwargs: value
            ),
            use_swizzleA=False,
        )

    def test_sidecar_keys_are_opt_in_and_distinct(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOE_TP_PREFILL_BACKEND", None)
            self.assertIsNone(_flashinfer_raw_scale_key(W.moe_w1))

        with patch.dict(
            os.environ, {"MOE_TP_PREFILL_BACKEND": "flashinfer_sm12x"}, clear=False
        ):
            self.assertEqual(_flashinfer_raw_scale_key(W.moe_w1), W.moe_s1_raw)
            self.assertEqual(_flashinfer_raw_scale_key(W.moe_w2), W.moe_s2_raw)
            self.assertIsNone(_flashinfer_raw_scale_key(W.ffn_w1))
            self.assertNotEqual(W.moe_s1_raw, W.moe_s1)
            self.assertNotEqual(W.moe_s2_raw, W.moe_s2)

    def test_flashinfer_postprocess_retains_local_canonical_sidecar(self):
        weight = self._weight()
        kernel = torch.empty((3, 256, 128), dtype=torch.float8_e4m3fn)
        scale = torch.empty((3, 2, 1), dtype=torch.float32)
        # A non-contiguous raw scale makes the sidecar's ownership/shape
        # contract explicit: it is the already-local raw scale, made contiguous
        # for the FlashInfer adapter, not re-sharded or reconstructed.
        raw_scale = (
            torch.arange(6, dtype=torch.float32).reshape(3, 1, 2).transpose(-1, -2)
        )
        packed_scale = torch.empty((3, 2, 1), dtype=torch.int32)

        with (
            patch.dict(
                os.environ,
                {"MOE_TP_PREFILL_BACKEND": "flashinfer_sm12x"},
                clear=False,
            ),
            patch.object(
                CompositeWeight,
                "_postprocess",
                return_value={W.moe_w1: kernel, W.moe_s1: scale},
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
                return_value=(kernel, packed_scale, raw_scale),
            ) as requant,
        ):
            result = weight._postprocess({}, "cpu", self._load_config())

        requant.assert_called_once_with(kernel, scale, return_raw_scale=True)
        self.assertIs(result[W.moe_w1], kernel)
        self.assertIs(result[W.moe_s1], packed_scale)
        self.assertTrue(torch.equal(result[W.moe_s1_raw], raw_scale))
        self.assertTrue(result[W.moe_s1_raw].is_contiguous())
        self.assertEqual(result[W.moe_s1_raw].shape, raw_scale.shape)

    def test_default_postprocess_does_not_create_sidecar(self):
        weight = self._weight()
        kernel = torch.empty((3, 256, 128), dtype=torch.float8_e4m3fn)
        scale = torch.empty((3, 2, 1), dtype=torch.float32)
        packed_scale = torch.empty((3, 2, 1), dtype=torch.int32)

        with (
            patch.dict(os.environ, {"MOE_TP_PREFILL_BACKEND": "default"}, clear=False),
            patch.object(
                CompositeWeight,
                "_postprocess",
                return_value={W.moe_w1: kernel, W.moe_s1: scale},
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
                return_value=(kernel, packed_scale),
            ) as requant,
        ):
            result = weight._postprocess({}, "cpu", self._load_config())

        requant.assert_called_once_with(kernel, scale)
        self.assertNotIn(W.moe_s1_raw, result)

    def test_flashinfer_rejects_packed_scale_without_inverse(self):
        weight = self._weight()
        kernel = torch.empty((3, 256, 128), dtype=torch.float8_e4m3fn)
        packed_scale = torch.empty((3, 2, 1), dtype=torch.int32)

        with (
            patch.dict(
                os.environ,
                {"MOE_TP_PREFILL_BACKEND": "flashinfer_sm12x"},
                clear=False,
            ),
            patch.object(
                CompositeWeight,
                "_postprocess",
                return_value={W.moe_w1: kernel, W.moe_s1: packed_scale},
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=True,
            ),
            patch(
                "rtp_llm.models_py.kernels.cuda.fp8_kernel.requant_weight_ue8m0",
            ) as requant,
        ):
            with self.assertRaisesRegex(ValueError, "implemented inverse"):
                weight._postprocess({}, "cpu", self._load_config())

        requant.assert_not_called()


if __name__ == "__main__":
    unittest.main()
