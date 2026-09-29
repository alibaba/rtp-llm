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
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    per_block_cast_to_fp8,
    quant_weight_ue8m0_packed,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.model_desc.bert import BertDecoderLayer
from rtp_llm.models_py.modules.base.cuda.norm import AddBiasResLayerNorm
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType
from rtp_llm.utils.model_weight import W


class BertFp8FusionGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError("This target requires a real NVIDIA GPU")

    @torch.inference_mode()
    def test_norm_quantized_values_and_scale_layout(self):
        torch.manual_seed(20260929)
        for dtype in (torch.float16, torch.bfloat16):
            for rows, width in ((1, 128), (7, 768), (17, 1024)):
                with self.subTest(dtype=dtype, rows=rows, width=width):
                    hidden = torch.randn(rows, width, device="cuda", dtype=dtype)
                    residual = torch.randn_like(hidden)
                    bias = torch.randn(width, device="cuda", dtype=dtype)
                    norm = AddBiasResLayerNorm(
                        torch.randn_like(bias), torch.randn_like(bias)
                    )
                    expected = norm(hidden.clone(), residual.clone(), bias)
                    actual, quantized, scales = norm.forward_quantized(
                        hidden.clone(), residual.clone(), bias
                    )
                    torch.testing.assert_close(actual, expected)
                    # Quantization must match exactly for the emitted norm
                    # values; small reduction rounding is checked separately.
                    reference_q, reference_s = sgl_per_token_group_quant_fp8(
                        actual,
                        128,
                        eps=1e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    torch.testing.assert_close(
                        quantized.float(), reference_q.float(), rtol=0, atol=0
                    )
                    # Unused packed bytes and TMA row padding are unspecified.
                    torch.testing.assert_close(
                        _unpack_ue8m0_scale_bytes(scales, width, 128),
                        _unpack_ue8m0_scale_bytes(reference_s, width, 128),
                        rtol=0,
                        atol=0,
                    )
                    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[
                        0
                    ] in (10, 12):
                        weight, weight_scales = quant_weight_ue8m0_packed(
                            torch.randn(128, width, device="cuda", dtype=dtype) * 0.05
                        )
                        linear = CudaFp8DeepGEMMLinear(
                            weight,
                            weight_scales=weight_scales,
                            quant_config=init_quant_config("FP8_PER_BLOCK"),
                        )
                        torch.testing.assert_close(
                            linear.forward_quantized(quantized, scales),
                            linear(actual),
                            rtol=0,
                            atol=0,
                        )

    @torch.inference_mode()
    def test_f16_linear_preserves_shared_input_contract(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )

        for shape, dtype in (((7, 128), torch.float32), ((2, 7, 128), torch.bfloat16)):
            with self.subTest(shape=shape, dtype=dtype):
                weight = torch.randn(128, 64, device="cuda", dtype=dtype)
                bias = torch.randn(64, device="cuda", dtype=dtype)
                x = torch.randn(shape, device="cuda", dtype=dtype)
                linear = CudaF16Linear(weight, bias=bias)
                torch.testing.assert_close(
                    linear.forward_with_bias_gelu(x),
                    torch.nn.functional.gelu(
                        torch.nn.functional.linear(x, weight.T, bias)
                    ),
                    rtol=0,
                    atol=0,
                )
        # Autocast may change GEMM's output dtype independently of the input
        # and bias. Preserve the original PyTorch contract in that context.
        weight = torch.randn(128, 64, device="cuda", dtype=torch.float16)
        bias = torch.randn(64, device="cuda", dtype=torch.float16)
        x = torch.randn(7, 128, device="cuda", dtype=torch.float16)
        linear = CudaF16Linear(weight, bias=bias)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            torch.testing.assert_close(
                linear.forward_with_bias_gelu(x),
                torch.nn.functional.gelu(torch.nn.functional.linear(x, weight.T, bias)),
                rtol=0,
                atol=0,
            )

    @torch.inference_mode()
    def test_mlp_fusion_ignores_retired_environment_switch(self):
        # Exercise real GPU GEMMs and, on UE8M0 backends, fused GELU+quant.
        # Only factory selection is patched: this test is independent of the
        # server's CUTLASS selection and does not launch a model service.
        packed = torch.cuda.get_device_capability()[0] in (10, 12)
        quant_config = init_quant_config("FP8_PER_BLOCK")
        source_weight = torch.eye(128, device="cuda", dtype=torch.bfloat16)

        def make_linear():
            # Use the loader's native scale layout, including TMA padding.
            # A contiguous (128, 1) int32 tensor has stride (1, 1), whereas
            # DeepGEMM requires the scale-column stride to remain 128.
            # Do not clone/contiguous the scales and collapse that singleton
            # dimension's nonstandard stride.
            if packed:
                weight, scales = quant_weight_ue8m0_packed(source_weight)
                self.assertEqual(scales.stride(), (1, 128))
            else:
                weight, scales = per_block_cast_to_fp8(source_weight, use_ue8m0=False)
            return CudaFp8DeepGEMMLinear(
                weight,
                weight_scales=scales,
                bias=torch.full((128,), 0.125, dtype=torch.bfloat16, device="cuda"),
                quant_config=quant_config,
            )

        up, down = make_linear(), make_linear()
        x = torch.linspace(-3, 3, 128, device="cuda", dtype=torch.bfloat16)
        x = x.repeat(128, 1).contiguous()
        expected = None
        for value in (None, "0", "1"):
            with self.subTest(retired_switch=value), mock.patch.dict(os.environ):
                if value is None:
                    os.environ.pop("DISABLE_FUSED_ACTIVATION_QUANT", None)
                else:
                    os.environ["DISABLE_FUSED_ACTIVATION_QUANT"] = value
                with mock.patch(
                    "rtp_llm.models_py.modules.hybrid.dense_mlp.LinearFactory.create_linear_from_weights",
                    side_effect=[up, down],
                ):
                    mlp = DenseMLP(
                        ActivationType.Gelu,
                        SimpleNamespace(get_ffn_tp_size=lambda: 1),
                        {},
                        quant_config,
                    )
                with mock.patch.object(
                    up,
                    "forward_with_bias_gelu_quantized",
                    wraps=up.forward_with_bias_gelu_quantized,
                ) as fused:
                    output, bias = mlp.forward_without_output_bias(x)
                self.assertEqual(fused.call_count, int(packed))
                self.assertIs(bias, down.bias)
                self.assertTrue(torch.isfinite(output).all())
                if expected is None:
                    expected = output.clone()
                else:
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)

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
            weight,
            weight_scales=scales,
            quant_config=init_quant_config("FP8_PER_BLOCK"),
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
                        CompositeWeight,
                        "_postprocess",
                        return_value={name: weight, scale_name: scales},
                    ),
                    mock.patch(
                        "rtp_llm.models_py.utils.arch.is_sm120", return_value=True
                    ),
                    mock.patch(
                        "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                        return_value=True,
                    ),
                    mock.patch.dict(
                        os.environ, {"RTP_LLM_SM120_FP8_BACKEND": "cutlass"}
                    ),
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
                    scale = torch.exp2(exponent.float() - 127).repeat_interleave(
                        128, -1
                    )
                    restored = result[name].float() * scale
                torch.testing.assert_close(restored, weight.float(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
