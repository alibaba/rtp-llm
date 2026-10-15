"""Tests for layered online NVFP4 dense FFN quantization.

The layered mode is gated by ``quant_config.enable_nvfp4_dense_ffn`` on top of
a non-fp4 base quantization (e.g. FP8_PER_BLOCK): the dense FFN weights are
claimed by OnlineModelOptFp4DenseWeight instead of the base per-block FP8
loader, and the produced packed uint8 kernels route to the FP4 GEMM linear.
"""

import unittest
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import torch

import rtp_llm.model_loader  # noqa: F401  (registers QuantWeight classes)
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.device.device_impl import CudaImpl
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, MoeAtomicWeight
from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    NVFP4_BLOCK_SIZE,
    OnlineModelOptFp4DenseWeight,
    quantize_expert_to_nvfp4,
)
from rtp_llm.model_loader.per_block_fp8_quant_weight import PerBlockFp8Weight
from rtp_llm.utils.model_weight import W


def _quant_config(flag: bool = True):
    # PerBlockFp8Weight.support() isinstance-checks the real algo config, and
    # init_precision_config injects enable_nvfp4_dense_ffn onto it.
    config = Fp8BlockWiseQuantConfig(is_quanted=True)
    config.enable_nvfp4_dense_ffn = flag
    return config


def _ffn_weight(name: str) -> FfnAtomicWeight:
    return FfnAtomicWeight(
        name=name,
        weights=[],
        process_fun=lambda tensors: tensors[0],
        data_type=None,
        config=None,
    )


class OnlineNvfp4DenseWeightSupportTest(unittest.TestCase):
    def test_support_requires_flag(self) -> None:
        src = _ffn_weight(W.ffn_w1)
        self.assertFalse(OnlineModelOptFp4DenseWeight.support(_quant_config(False), src))
        self.assertTrue(OnlineModelOptFp4DenseWeight.support(_quant_config(True), src))

    def test_support_claims_all_dense_ffn_weights(self) -> None:
        for name in (W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13):
            with self.subTest(name=name):
                self.assertTrue(
                    OnlineModelOptFp4DenseWeight.support(_quant_config(True), _ffn_weight(name))
                )

    def test_support_rejects_moe_weights(self) -> None:
        src = MoeAtomicWeight(
            name=W.moe_w1, weights=[], process_fun=lambda t: t[0], data_type=None, config=None
        )
        self.assertFalse(OnlineModelOptFp4DenseWeight.support(_quant_config(True), src))

    def test_per_block_fp8_yields_ffn_when_layered_flag_on(self) -> None:
        for name in (W.ffn_w1, W.ffn_w2, W.ffn_w3, W.ffn_w13):
            with self.subTest(name=name):
                src = _ffn_weight(name)
                self.assertTrue(PerBlockFp8Weight.support(_quant_config(False), src))
                self.assertFalse(PerBlockFp8Weight.support(_quant_config(True), src))

    def test_per_block_fp8_keeps_non_ffn_weights(self) -> None:
        src = _ffn_weight(W.attn_qkv_w)
        self.assertTrue(PerBlockFp8Weight.support(_quant_config(True), src))


class OnlineNvfp4QuantMathTest(unittest.TestCase):
    def test_quantize_dense_weight_round_trip(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        torch.manual_seed(11)
        weight = (torch.randn(64, 128, device="cuda") * 0.05).to(torch.bfloat16)
        packed, block_scale, scale_2 = quantize_expert_to_nvfp4(weight, NVFP4_BLOCK_SIZE)

        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (64, 64))
        self.assertEqual(block_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(block_scale.shape), (64, 128 // NVFP4_BLOCK_SIZE))
        self.assertEqual(scale_2.dtype, torch.float32)
        self.assertEqual(scale_2.numel(), 1)

        # Dequantize: e2m1 ordinal -> value, then per-block scale * scale2.
        # The magnitude of ordinal 15 is 6 (not 5), so index the LUT with the
        # full 4-bit ordinal instead of masking off the sign bit.
        low = (packed & 0x0F)
        high = (packed >> 4)
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            device=packed.device,
        )
        sign = lambda q: torch.where(q >= 8, -1.0, 1.0)
        magnitude = lambda q: lut[q.long()]
        unpacked = torch.stack(
            [sign(low) * magnitude(low), sign(high) * magnitude(high)], dim=-1
        ).view(64, 128)
        dequant = (
            unpacked.view(64, 128 // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
            * block_scale.float().unsqueeze(-1)
            * scale_2.float()
        ).view(64, 128)
        # NVFP4 e2m1 worst case: a value just above 4 in a block whose grid
        # rounds it down to 4 (or just above 5 to 4 for negative six) carries
        # up to ~33% relative error, and near-zero values need an absolute
        # allowance; the round trip must stay inside that envelope.
        torch.testing.assert_close(dequant, weight.float(), rtol=0.34, atol=0.02)

    def test_weight_class_quantizes_and_names_outputs(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        torch.manual_seed(13)
        # K must be a multiple of the 128-wide NVFP4 block.
        weight = (torch.randn(32, 256, device="cuda") * 0.1).to(torch.bfloat16)
        src = _ffn_weight(W.ffn_w2)
        quant_weight = OnlineModelOptFp4DenseWeight(src, _quant_config(True))

        with mock.patch.object(
            FfnAtomicWeight,
            "_load_raw_tensor",
            return_value={W.ffn_w2: weight},
        ):
            tensors = quant_weight._load_raw_tensor(
                tensor_source=mock.Mock(),
                layer_id=0,
                device="cuda",
                load_config=SimpleNamespace(exported_device=CudaImpl),
            )

        self.assertEqual(set(tensors), {W.ffn_w2, W.ffn_s2, W.ffn_w2_s2, W.ffn_w2_i_s})
        self.assertEqual(tensors[W.ffn_w2].dtype, torch.uint8)
        self.assertEqual(tensors[W.ffn_s2].dtype, torch.float8_e4m3fn)
        # The GEMM scale leaves in the 128x4-swizzled layout: M padded to 128
        # (32 -> 128) and 16 scale columns per 256-wide K.
        self.assertEqual(tuple(tensors[W.ffn_s2].shape), (128, 16))
        self.assertEqual(tensors[W.ffn_w2_s2].dtype, torch.float32)
        self.assertEqual(tensors[W.ffn_w2_i_s].dtype, torch.float32)
        self.assertEqual(tensors[W.ffn_w2_i_s].flatten().tolist(), [1.0])

    def test_online_shapes_through_fp4_gemm(self) -> None:
        """B4: the loader's exact output tensors must run through mm_fp4.

        Exercises the layered online path end to end on the GEMM level:
        packed e2m1 + per-block fp8 scale + 0-dim fp32 scale_2 with
        input_scale=1.0 (dynamic per-block activation quantization), routed
        through CudaFp4GEMMLinear with the layered flag.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
            CudaFp4GEMMLinear,
        )

        torch.manual_seed(19)
        output_size, hidden_size = 256, 512
        weight_bf16 = (
            torch.randn(output_size, hidden_size, device="cuda") * 0.05
        ).to(torch.bfloat16)
        packed, block_scale, scale_2 = quantize_expert_to_nvfp4(
            weight_bf16, NVFP4_BLOCK_SIZE
        )
        self.assertEqual(scale_2.numel(), 1)
        # The loader applies the GEMM scale swizzle; mirror it here.
        gemm_weight, gemm_scale = CudaImpl.convert_fp4_gemm_weight_params(
            packed, block_scale
        )

        quant_config = _quant_config(True)
        self.assertTrue(
            CudaFp4GEMMLinear.can_handle(
                quant_config,
                gemm_weight,
                gemm_scale,
                weight_scale_2=scale_2,
                input_scale=torch.tensor(1.0),
            )
        )
        linear = CudaFp4GEMMLinear(
            weight=gemm_weight,
            weight_scales=gemm_scale,
            input_scales=torch.tensor(1.0),
            bias=None,
            quant_config=quant_config,
            weight_scale_2=scale_2,
        )
        self.assertEqual(linear.backend, "cutlass")

        x = (torch.randn(32, hidden_size, device="cuda") * 0.05).to(torch.bfloat16)
        out = linear(x)
        reference = (x.float() @ weight_bf16.float().t()).to(torch.bfloat16)
        # e2m1 weight quantization: each block snaps to a 16-level grid, so a
        # coarse mean-error bound is the right acceptance criterion.
        mean_abs_err = (out.float() - reference.float()).abs().mean()
        mean_abs_ref = reference.float().abs().mean()
        self.assertLess(float(mean_abs_err / mean_abs_ref), 0.12)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())


class OnlineNvfp4LinearRoutingTest(unittest.TestCase):
    def test_should_use_fp4_linear_layered_flag(self) -> None:
        from rtp_llm.models_py.modules.factory.linear import LinearFactory

        weights = {W.ffn_w1: torch.zeros(4, 8, dtype=torch.uint8)}
        bf16_weights = {W.ffn_w1: torch.zeros(4, 16, dtype=torch.bfloat16)}
        self.assertTrue(
            LinearFactory.should_use_fp4_linear(_quant_config(True), weights, W.ffn_w1)
        )
        # Without the flag the base quant method must reject the weights.
        self.assertFalse(
            LinearFactory.should_use_fp4_linear(_quant_config(False), weights, W.ffn_w1)
        )
        # BF16 weights never route to FP4 even with the flag on.
        self.assertFalse(
            LinearFactory.should_use_fp4_linear(_quant_config(True), bf16_weights, W.ffn_w1)
        )

    def test_fp4_linear_can_handle_layered_flag(self) -> None:
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
            CudaFp4GEMMLinear,
        )

        packed = torch.zeros(4, 8, dtype=torch.uint8)
        scales = torch.zeros(4, 1, dtype=torch.float8_e4m3fn)
        scale_2 = torch.ones(1, dtype=torch.float32)
        input_scale = torch.ones(1, dtype=torch.float32)
        kwargs = dict(
            weight_scales=scales,
            weight_scale_2=scale_2,
            input_scale=input_scale,
        )
        self.assertTrue(
            CudaFp4GEMMLinear.can_handle(_quant_config(True), packed, **kwargs)
        )
        self.assertFalse(
            CudaFp4GEMMLinear.can_handle(_quant_config(False), packed, **kwargs)
        )
        # FP8-packed weights must not route to the FP4 GEMM.
        fp8_weight = torch.zeros(4, 16, dtype=torch.float8_e4m3fn)
        self.assertFalse(
            CudaFp4GEMMLinear.can_handle(_quant_config(True), fp8_weight, **kwargs)
        )


class OnlineNvfp4DenseMLPTest(unittest.TestCase):
    def _layered_weights(self, hidden: int = 32, inter: int = 64) -> dict:
        # Minimal online-NVFP4 weight set for a gated DenseMLP.
        def packed(n: int, k: int) -> torch.Tensor:
            return torch.zeros(n, k // 2, dtype=torch.uint8)

        def block_scale(n: int, k: int) -> torch.Tensor:
            return torch.ones(n, k // NVFP4_BLOCK_SIZE, dtype=torch.float8_e4m3fn)

        return {
            W.ffn_w1: packed(inter, hidden),
            W.ffn_s1: block_scale(inter, hidden),
            W.ffn_w1_s2: torch.ones(1, dtype=torch.float32),
            W.ffn_w1_i_s: torch.ones(1, dtype=torch.float32),
            W.ffn_w3: packed(inter, hidden),
            W.ffn_s3: block_scale(inter, hidden),
            W.ffn_w3_s2: torch.ones(1, dtype=torch.float32),
            W.ffn_w3_i_s: torch.ones(1, dtype=torch.float32),
            W.ffn_w2: packed(hidden, inter),
            W.ffn_s2: block_scale(hidden, inter),
            W.ffn_w2_s2: torch.ones(1, dtype=torch.float32),
            W.ffn_w2_i_s: torch.ones(1, dtype=torch.float32),
        }

    def _make_mlp(self, weights: dict):
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
        from rtp_llm.ops import ActivationType

        return DenseMLP(
            activation_type=ActivationType.Swiglu,
            parallelism_config=SimpleNamespace(get_ffn_tp_size=lambda: 1),
            weights=weights,
            quant_config=_quant_config(True),
        )

    def test_gated_dense_mlp_uses_separate_gate_up_linears(self) -> None:
        created = []

        def fake_create(weights, weight_key, *args, **kwargs):
            created.append(weight_key)
            return SimpleNamespace(forward=lambda x: x, bias=None)

        with mock.patch(
            "rtp_llm.models_py.modules.factory.linear.LinearFactory."
            "create_linear_from_weights",
            side_effect=fake_create,
        ):
            mlp = self._make_mlp(self._layered_weights())

        self.assertTrue(mlp.separate_gated_nvfp4)
        self.assertEqual(created, [W.ffn_w1, W.ffn_w3, W.ffn_w2])
        self.assertIsNone(mlp.w4a16)

    def test_gated_dense_mlp_separate_path_forward_math(self) -> None:
        class Recorder:
            def __init__(self, factor: float) -> None:
                self.factor = factor
                self.bias = None

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.factor

        with mock.patch(
            "rtp_llm.models_py.modules.factory.linear.LinearFactory."
            "create_linear_from_weights",
            side_effect=lambda weights, weight_key, *a, **kw: Recorder(
                {W.ffn_w1: 2.0, W.ffn_w3: 3.0, W.ffn_w2: 5.0}[weight_key]
            ),
        ):
            mlp = self._make_mlp(self._layered_weights())

        x = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        # silu(gate(x)) * up(x) then down(...)
        expected = torch.nn.functional.silu(2.0 * x) * (3.0 * x) * 5.0
        torch.testing.assert_close(mlp(x, skip_allreduce=True), expected)


if __name__ == "__main__":
    unittest.main()
