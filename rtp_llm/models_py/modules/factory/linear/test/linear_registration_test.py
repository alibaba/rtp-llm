import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, ModelOptFp4Config
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)


class LinearRegistrationTest(unittest.TestCase):
    def test_hybrid_fp4_attention_accepts_fp8_backend(self):
        config = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=True,
            hybrid_attn_quant_method="FP8_PER_BLOCK",
        )
        weight = torch.zeros(128, 128).to(torch.float8_e4m3fn)
        scales = torch.ones(1, 1)
        self.assertTrue(CudaFp8GEMMLinear.can_handle(config, weight, scales))
        self.assertFalse(CudaF16Linear.can_handle(config, weight, scales))
        for weight_scale_2, input_scale in (
            (torch.ones(1), None),
            (None, torch.ones(1)),
        ):
            self.assertFalse(
                CudaFp8GEMMLinear.can_handle(
                    config,
                    weight,
                    scales,
                    weight_scale_2=weight_scale_2,
                    input_scale=input_scale,
                )
            )
        plain_fp4 = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
        self.assertFalse(CudaFp8GEMMLinear.can_handle(plain_fp4, weight, scales))

    def test_unscaled_fp32_projection_keeps_nonquantized_backend(self):
        self.assertIn(CudaF16Linear, LinearFactory._strategies)
        weight = torch.arange(12, dtype=torch.float32).view(4, 3)
        for hw in (None, MagicMock()):
            with self.subTest(hw=hw):
                layer = LinearFactory.create_linear(
                    weight, None, None, Fp8BlockWiseQuantConfig(), hw
                )
                self.assertIs(type(layer), CudaF16Linear)
                actual = layer(torch.ones(2, 4))
                torch.testing.assert_close(
                    actual, weight.sum(0).expand(2, 3), rtol=0, atol=0
                )

    def test_batch_invariant_layer_uses_fixed_order_operator(self):
        from rtp_llm.models_py.determinism import batch_invariant

        weight = torch.arange(12, dtype=torch.bfloat16).view(4, 3)
        bias = torch.ones(3, dtype=weight.dtype)
        x = torch.ones(2, 4, dtype=weight.dtype)
        layer = LinearFactory.create_linear(
            weight, bias, None, None, SimpleNamespace(batch_invariant=True)
        )
        self.assertIsInstance(layer, batch_invariant.BatchInvariantLinear)
        layer.maybe_cache_quant_scale(1024)
        with patch.object(
            batch_invariant, "batch_invariant_linear", wraps=torch.nn.functional.linear
        ) as op:
            actual = layer(x)
        op.assert_called_once_with(x, layer.weight, bias)
        torch.testing.assert_close(actual, (weight.sum(0) + bias).expand(2, 3))

    def test_batch_invariant_rejects_bias_without_a_shared_output_row(self):
        from rtp_llm.models_py.determinism.batch_invariant import batch_invariant_linear

        x = torch.empty(0, 4, dtype=torch.bfloat16)
        weight = torch.empty(3, 4, dtype=x.dtype)
        for shape in ((), (1,), (3,), (1, 3)):
            with self.subTest(shape=shape):
                bias = torch.ones(shape, dtype=x.dtype)
                result = batch_invariant_linear(x, weight, bias)
                self.assertEqual(result.shape, (0, 3))
        for shape in ((2,), (2, 3)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                batch_invariant_linear(x, weight, torch.ones(shape, dtype=x.dtype))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
        "Qwen3 batch-invariant inference requires SM103",
    )
    def test_qwen3_mlp_projection_is_exact_across_prefill_and_decode(self):
        from rtp_llm.models_py.determinism import batch_invariant as bi

        generator = torch.Generator(device="cuda").manual_seed(1119)

        def randn(*shape):
            return torch.randn(
                shape, generator=generator, dtype=torch.bfloat16, device="cuda"
            )

        with patch.object(bi, "_persistent_mlp_linear") as kernel:
            for weight in (randn(127, 64), randn(128, 65), randn(64, 128).T):
                x = randn(256, weight.shape[1])
                expected = bi.batch_invariant_linear(x, weight)
                result = bi.batch_invariant_linear(x, weight, mlp_num_sms=1)
                self.assertTrue(torch.equal(result, expected))
            kernel.__getitem__.assert_not_called()

        for n, k in ((24576, 4096), (4096, 12288)):
            with self.subTest(n=n, k=k):
                original = bi.BatchInvariantLinear(randn(k, n))
                projection = bi.BatchInvariantMLPLinear(original)
                batch = randn(513, k)
                rows = [0, 1, 127, 255, 256, 511, 512]
                batch[rows] = batch[0].expand(len(rows), -1).clone()
                expected = bi.batch_invariant_linear(
                    batch[:1], original.weight, output_fp32=True
                )
                sample = batch[:1].double() @ original.weight[:16].double().T
                torch.testing.assert_close(
                    expected[:, :16].double(), sample, rtol=2e-4, atol=5e-3
                )
                persistent = bi._persistent_mlp_linear
                with patch.object(bi, "_persistent_mlp_linear") as kernel:
                    kernel.__getitem__.side_effect = persistent.__getitem__
                    for m in (1, 255, 256, 257, 513):
                        selected = [r for r in rows if r < m]
                        result = bi.batch_invariant_linear(
                            batch[:m],
                            projection.weight,
                            output_fp32=True,
                            mlp_num_sms=projection.num_sms,
                        )
                        self.assertTrue(
                            torch.equal(
                                result[selected], expected.expand(len(selected), -1)
                            )
                        )
                    self.assertEqual(kernel.__getitem__.call_count, 3)
                x = batch[:4].clone()
                projection(x)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = projection(x)
                x.copy_(batch[4:8])
                graph.replay()
                self.assertTrue(torch.equal(output, projection(batch[4:8])))

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
        "Qwen3 batch-invariant inference requires SM103",
    )
    def test_qwen3_mlp_keeps_loader_and_compute_weights_in_sync(self):
        from rtp_llm.models_py.determinism.batch_invariant import (
            BatchInvariantLinear,
            BatchInvariantMLPLinear,
        )
        from rtp_llm.models_py.model_desc import qwen3
        from rtp_llm.ops import ActivationType
        from rtp_llm.resident_weights import WeightDict
        from rtp_llm.utils.model_weight import W

        config = MagicMock(
            model_type="qwen_3",
            num_layers=36,
            hidden_size=4096,
            inter_size=12288,
            activation_type=ActivationType.Swiglu,
        )
        parallel = MagicMock()
        parallel.get_ffn_tp_size.return_value = 1
        hw = SimpleNamespace(batch_invariant=True)
        options = dict(device="cuda", dtype=torch.bfloat16)
        x = torch.ones((256, 128), **options)

        def make_layer(weights):
            with patch.object(
                qwen3, "CausalAttention", return_value=torch.nn.Identity()
            ):
                return qwen3.Qwen3DecoderLayer(config, parallel, 0, weights, None, hw)

        for merged in (True, False):
            with self.subTest(merged=merged):
                weights = {
                    W.ffn_w2: torch.ones((256, 128), **options),
                    W.pre_ln_gamma: torch.ones(128, **options),
                    W.post_ln_gamma: torch.ones(128, **options),
                }
                if merged:
                    weights[W.ffn_w13] = torch.ones((128, 512), **options)
                else:
                    weights[W.ffn_w1] = torch.ones((128, 256), **options)
                    weights[W.ffn_w3] = torch.ones((128, 256), **options)
                manager = MagicMock()
                resident = WeightDict(mapping=weights)
                resident.bind(manager)
                resident_layer = make_layer(resident)
                self.assertIs(type(resident_layer.mlp.down_proj), BatchInvariantLinear)
                for key in weights:
                    self.assertIs(resident[key], weights[key])

                layer = make_layer(weights)
                self.assertIsInstance(layer.mlp.up_proj, BatchInvariantMLPLinear)
                self.assertIsInstance(layer.mlp.down_proj, BatchInvariantMLPLinear)
                self.assertTrue(layer.mlp.up_proj.weight.is_contiguous())
                up_keys = [W.ffn_w13] if merged else [W.ffn_w1, W.ffn_w3]
                for i, key in enumerate(up_keys, start=1):
                    self.assertEqual(
                        weights[key].untyped_storage().data_ptr(),
                        layer.mlp.up_proj.weight.untyped_storage().data_ptr(),
                    )
                    weights[key].fill_(i + 1)
                weights[W.ffn_w2].fill_(4)
                expected_up = torch.cat([weights[key] for key in up_keys], dim=1)
                self.assertTrue(torch.equal(layer.mlp.up_proj(x), x @ expected_up))
                self.assertTrue(
                    torch.equal(
                        layer.mlp.down_proj(torch.ones((1, 256), **options)),
                        torch.full((1, 128), 1024, **options),
                    )
                )
                config.model_type = "qwen_2"
                other = make_layer(weights)
                self.assertIs(type(other.mlp.up_proj), BatchInvariantLinear)
                config.model_type = "qwen_3"


if __name__ == "__main__":
    unittest.main()
