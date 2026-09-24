"""GPU regressions for BERT-only fused norm dispatch and loader isolation."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    PerBlockFp8Weight,
    _unpack_ue8m0_scale_bytes,
)
from rtp_llm.model_loader.weight_module import CompositeWeight
from rtp_llm.models_py.model_desc.bert import BertDecoderLayer
from rtp_llm.models_py.modules.base.cuda.norm import AddBiasResLayerNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.utils.model_weight import W


class BertFp8FusionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("This target requires a real NVIDIA GPU")

    @torch.inference_mode()
    def test_backend_capability_matches_actual_scales(self):
        packed = torch.cuda.get_device_capability()[0] in (10, 12)
        weight = torch.ones((128, 128), device="cuda").to(torch.float8_e4m3fn)
        scales = (
            torch.full((128, 1), 0x7F7F7F7F, dtype=torch.int32, device="cuda")
            if packed
            else torch.ones((1, 1), device="cuda")
        )
        linear = CudaFp8GEMMLinear(
            weight, weight_scales=scales, quant_config=init_quant_config("FP8_PER_BLOCK")
        )
        for backend in (linear, linear._deepgemm_linear):
            self.assertEqual(backend.supports_prequantized_activation, packed)
            hidden = torch.randn(4, 128, dtype=torch.bfloat16, device="cuda")
            self.assertEqual(
                BertDecoderLayer._can_fuse_layernorm_quant(hidden, backend), packed
            )

    @torch.inference_mode()
    def test_norm_fusion_shape_and_format_fallback(self):
        # The consumer descriptor isolates dispatch; the selected norm is the
        # real GPU binding, not a mock of the operation under test.
        for width in (128, 768, 1024, 1280, 1536, 2048):
            for packed in (False, True):
                with self.subTest(width=width, packed=packed):
                    linear = SimpleNamespace(
                        supports_prequantized_activation=True,
                        fused_activation_quant_format=(
                            "fp8_ue8m0_block128_colmajor" if packed else None
                        ),
                    )
                    hidden = torch.randn(7, width, device="cuda", dtype=torch.bfloat16)
                    residual = torch.randn_like(hidden)
                    bias = torch.randn(width, device="cuda", dtype=hidden.dtype)
                    norm = AddBiasResLayerNorm(
                        torch.ones_like(bias), torch.zeros_like(bias)
                    )
                    expected = norm(hidden.clone(), residual.clone(), bias)
                    fused = BertDecoderLayer._can_fuse_layernorm_quant(hidden, linear)
                    self.assertEqual(fused, packed and width <= 1024)
                    if fused:
                        actual, fp8, scales = norm.forward_quantized(
                            hidden.clone(), residual.clone(), bias
                        )
                        self.assertEqual(fp8.dtype, torch.float8_e4m3fn)
                        self.assertEqual(scales.dtype, torch.int32)
                    else:
                        actual = norm(hidden.clone(), residual.clone(), bias)
                    if fused:
                        torch.testing.assert_close(actual, expected)
                    else:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cutlass_postprocess_preserves_moe_ue8m0(self):
        # Isolate checkpoint I/O and device discovery, but execute the real
        # _postprocess and GPU requant kernels for dense and expert weights.
        device = SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda key, value, **kw: value
        )
        config = SimpleNamespace(exported_device=device, use_swizzleA=False)
        for name, scale_name, shape in (
            (W.ffn_w1, W.ffn_s1, (128, 128)),
            (W.moe_w1, W.moe_s1, (2, 128, 128)),
            (W.moe_w2, W.moe_s2, (2, 128, 128)),
        ):
            with self.subTest(weight=name):
                loader = object.__new__(PerBlockFp8Weight)
                loader.kernel = SimpleNamespace(name=name)
                loader.scale = SimpleNamespace(name=scale_name)
                weight = torch.ones(shape, device="cuda").to(torch.float8_e4m3fn)
                scales = torch.ones((*shape[:-2], 1, 1), device="cuda")
                with (
                    mock.patch.object(
                        CompositeWeight, "_postprocess",
                        return_value={name: weight, scale_name: scales},
                    ),
                    mock.patch("rtp_llm.models_py.utils.arch.is_sm120", return_value=True),
                    mock.patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        return_value=True,
                    ),
                    mock.patch.dict(os.environ, {"RTP_LLM_SM120_FP8_BACKEND": "cutlass"}),
                ):
                    result = loader._postprocess({}, "cuda", config)
                self.assertEqual(
                    result[scale_name].dtype,
                    torch.float32 if name == W.ffn_w1 else torch.int32,
                )
                if name == W.ffn_w1:
                    restored = result[name].float()
                else:
                    exponent = _unpack_ue8m0_scale_bytes(result[scale_name], 128, 128)
                    scale = torch.exp2(exponent.float() - 127).repeat_interleave(128, -1)
                    restored = result[name].float() * scale
                torch.testing.assert_close(restored, weight.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
