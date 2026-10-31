import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from flashinfer import fp4_quantize

import rtp_llm.models_py.kernels.cuda.modelopt_fp4_quant as quant
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class ModelOptFp4ProducerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1548)
        self.enabled = patch.object(quant, "_FUSED_PRODUCER_ENABLED", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
        self.mask = torch.full(
            (1,), torch.iinfo(torch.int32).max, device="cuda", dtype=torch.int32
        )
        self.act = FusedSiluAndMul()

    def assert_quant_equal(self, x, scale):
        expected_q, expected_s = fp4_quantize(self.act(x), scale)
        q, s = quant.silu_fp4_quant(x, scale, self.mask)
        self.assertTrue(torch.equal(q, expected_q))
        m, two_k = x.shape
        k = two_k // 2
        i = torch.arange(m, device=x.device)[:, None]
        j = torch.arange(k // 16, device=x.device)[None, :]
        # Padding rows have no defined values and are never GEMM operands.
        offsets = (
            (i // 128) * (k // 64 * 512)
            + (j // 4) * 512
            + (i % 32) * 16
            + ((i % 128) // 32) * 4
            + j % 4
        )
        self.assertTrue(
            torch.equal(s.flatten()[offsets], expected_s.flatten()[offsets])
        )

    def test_quant_bytes_and_scale_layout(self):
        for k in (8704, 17408):
            for m in (1, 3, 17, 64, 257):
                for value in (0.125, 3.7, 32.0):
                    with self.subTest(k=k, m=m, scale=value):
                        x = (
                            torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
                            * 2
                        )
                        scale = torch.tensor([value], device="cuda")
                        self.assert_quant_equal(x, scale)

    def test_zero_and_small_values(self):
        x = torch.zeros(3, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        self.assert_quant_equal(x, scale)
        x[:, :8704] = torch.linspace(-80, 80, 8704, device="cuda")
        x[:, 8704:] = 0.0001
        self.assert_quant_equal(x, scale)

    def test_graph_replay_reads_current_input_and_scale(self):
        x = torch.randn(17, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        quant.silu_fp4_quant(x, scale, self.mask)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, s = quant.silu_fp4_quant(x, scale, self.mask)
        x.normal_()
        scale.fill_(3.7)
        graph.replay()
        expected_q, expected_s = quant.silu_fp4_quant(x, scale, self.mask)
        self.assertTrue(torch.equal(q, expected_q))
        # All scales in the first 16-row swizzle group belong to row 0.
        self.assertTrue(torch.equal(s.flatten()[:4], expected_s.flatten()[:4]))
        graph.reset()

    def test_projection_matches_old_silu_quant_gemm(self):
        k, n = 8704, 5120
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wscale = torch.tensor([32.0], device="cuda")
        wq, ws = fp4_quantize(weight, wscale)
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq,
                ws,
                torch.tensor([1 / 32], device="cuda"),
                bias=torch.randn(n, device="cuda", dtype=torch.bfloat16),
                weight_scale_2=1 / wscale,
            )
        for m in (1, 3, 5, 10, 17, 20, 40):
            x = torch.randn(m, k * 2, device="cuda", dtype=torch.bfloat16)
            layer.input_scale_inv = torch.tensor([1 / 0.01713], device="cuda")
            layer.alpha = torch.tensor([0.01713 / 32], device="cuda")
            layer._decode_auto_backend = False
            reference = layer(self.act(x))
            layer._decode_auto_backend = True
            torch.testing.assert_close(
                layer.forward_with_silu(x), reference, rtol=0, atol=0
            )

    def test_unsupported_and_disabled_fallback(self):
        x = torch.randn(1, 17408, device="cuda", dtype=torch.bfloat16)
        scale = torch.tensor([32.0], device="cuda")
        for unsupported in (x.float(), x[:, :8704], x.expand(2, -1), x[:0]):
            self.assertIsNone(quant.silu_fp4_quant(unsupported, scale, self.mask))
        with patch.object(quant, "_FUSED_PRODUCER_ENABLED", False):
            self.assertIsNone(quant.silu_fp4_quant(x, scale, self.mask))
        self.assertIsNone(quant.silu_fp4_quant(x, scale, None))

    def test_changed_grouped_layout_retains_original_producer(self):
        x = torch.randn(5, 17408, device="cuda", dtype=torch.bfloat16)
        before = x.clone()
        scale = torch.ones(1, device="cuda")
        q = torch.empty(5, 4352, 1, device="cuda", dtype=torch.uint8)
        sf = torch.empty(32, 4, 1, 4, 136, 1, device="cuda")
        for changed in [(q.permute(2, 0, 1), sf), (q, sf.transpose(0, 1))]:
            with patch.object(
                quant,
                "silu_and_mul_scaled_nvfp4_experts_quantize",
                return_value=changed,
            ):
                self.assertIsNone(quant.silu_fp4_quant(x, scale, self.mask))
        self.assertTrue(torch.equal(x, before))

    def test_large_dense_input_retains_original_producer(self):
        width = 17408
        rows = (2**31 + width - 1) // width
        x = torch.empty(rows, width, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda", dtype=torch.float32)
        self.assertIsNone(quant.silu_fp4_quant(x, scale, self.mask))

    def test_ar_norm_unsupported_inputs_preserve_tensors(self):
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        gamma = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        scale = torch.ones(1, device="cuda")
        proxy = SimpleNamespace(is_supported=lambda tensor: True)
        before_x, before_r = x.clone(), r.clone()
        self.assertIsNone(quant.residual_norm_fp4_quant(x, r, gamma, scale, 1e-6))
        for bad in (x.float(), x[:0], x[:1], x[:3], x[:, ::2]):
            self.assertIsNone(
                quant.residual_norm_fp4_quant(bad, r, gamma, scale, 1e-6, proxy)
            )
        self.assertIsNone(
            quant.residual_norm_fp4_quant(x, x, gamma, scale, 1e-6, proxy)
        )
        with patch.object(quant, "_FUSED_PRODUCER_ENABLED", False):
            self.assertIsNone(
                quant.residual_norm_fp4_quant(x, r, gamma, scale, 1e-6, proxy)
            )
        self.assertTrue(torch.equal(x, before_x))
        self.assertTrue(torch.equal(r, before_r))

    def test_up_projection_matches_cutlass(self):
        k, n = 5120, 17408
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wscale = torch.tensor([31.7], device="cuda")
        wq, ws = fp4_quantize(weight, wscale)
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq,
                ws,
                torch.tensor([0.01713], device="cuda"),
                weight_scale_2=1 / wscale,
            )
        runner = layer._decode_tiled_runner
        self.assertIsNotNone(runner)
        for m in (1, 3, 5, 10, 17, 20, 32, 40, 64, 65, 71):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            layer._decode_tiled_runner = None
            reference = layer(x)
            layer._decode_tiled_runner = runner
            with patch.object(runner, "forward", wraps=runner.forward) as tiled:
                actual = layer(x)
                self.assertEqual(tiled.call_count, int(m <= 64))
            self.assertTrue(
                torch.equal(actual.view(torch.int16), reference.view(torch.int16))
            )

    def test_lm_head_tiled_projection_and_fallback(self):
        k, n = 5120, 124160
        scale = torch.tensor([31.7], device="cuda")
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wq, ws = fp4_quantize(weight, scale)
        del weight
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq,
                ws,
                torch.tensor([0.01713], device="cuda"),
                bias=torch.randn(n, device="cuda", dtype=torch.bfloat16),
                weight_scale_2=1 / scale,
            )
        runner = layer._decode_tiled_runner
        self.assertIsNotNone(runner)
        for m in (1, 5, 8, 17, 40, 64, 65, 71):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            layer._decode_tiled_runner = None
            expected = layer(x)
            layer._decode_tiled_runner = runner
            with patch.object(runner, "forward", wraps=runner.forward) as tiled:
                actual = layer(x)
                self.assertEqual(tiled.call_count, int(m <= 64))
            self.assertTrue(
                torch.equal(actual.view(torch.int16), expected.view(torch.int16))
            )
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "0"}):
            disabled = CudaFp4GEMMLinear(
                wq, ws, layer.input_scale, weight_scale_2=1 / scale
            )
        self.assertIsNone(disabled._decode_tiled_runner)

    def test_tiled_graph_reads_current_inputs_weights_and_alpha(self):
        k, n, m = 5120, 17408, 17
        scale = torch.tensor([31.7], device="cuda")
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        wq, ws = fp4_quantize(weight, scale)
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            layer = CudaFp4GEMMLinear(
                wq, ws, torch.tensor([0.01713], device="cuda"), weight_scale_2=1 / scale
            )
        self.assertIsNotNone(layer._decode_tiled_runner)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        layer(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer(x)
        x.normal_()
        wq.fill_(42)
        with torch.inference_mode():
            layer.alpha.mul_(0.7)
        graph.replay()
        layer._decode_tiled_runner = None
        expected = layer(x)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )
        graph.reset()

    def _fused_mlp_layers(self, enabled=True):
        scale = torch.tensor([32.0], device="cuda")
        layers = []
        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": str(int(enabled))}):
            for n, k in [(17408, 5120), (5120, 8704)]:
                w, sf = fp4_quantize(
                    torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1,
                    scale,
                )
                layers.append(
                    CudaFp4GEMMLinear(w, sf, 1 / scale, weight_scale_2=1 / scale)
                )
        return layers

    def test_optional_runner_failure_and_cuda12_fallback(self):
        up, down = self._fused_mlp_layers(enabled=False)
        x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
        expected = up(x)
        runner_path = "flashinfer.gemm.gemm_base._b12x_gemm_fp4_runner"

        def new_layer(original):
            return CudaFp4GEMMLinear(
                original.weight,
                original.weight_scales,
                original.input_scale,
                weight_scale_2=original.weight_scale_2,
            )

        with patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}):
            for error in (ImportError, AttributeError, TypeError):
                with self.subTest(error=error), patch(runner_path, side_effect=error):
                    layer = new_layer(up)
                    self.assertIsNone(layer._decode_tiled_runner)
                    self.assertIsNone(layer._swiglu_weights)
                    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
            with patch.object(torch.version, "cuda", "12.9"), patch(
                runner_path
            ) as runner:
                layer = new_layer(up)
                down_layer = new_layer(down)
                runner.assert_not_called()
                self.assertIsNone(layer._decode_tiled_runner)
                self.assertIsNone(layer._swiglu_weights)
                self.assertFalse(down_layer._decode_auto_backend)
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_deferred_reduction_requires_ffn_tensor_parallelism(self):
        from rtp_llm.models_py.model_desc import qwen3_next as qwen

        config = SimpleNamespace(
            hybrid_attention_config=SimpleNamespace(
                hybrid_attention_types=[qwen.HybridAttentionType.LINEAR]
            ),
            linear_attention_config=None,
            layernorm_eps=1e-6,
            quant_config=None,
            moe_style=0,
            activation_type="swiglu",
            enable_w4a16_sm120_dense_ffn=False,
        )
        mlp = torch.nn.Module()
        mlp.is_gated = True
        mlp.up_proj = SimpleNamespace(
            hidden_size=5120,
            output_size=17408,
            forward_with_residual_norm=lambda *args: None,
        )
        weights = {
            name: torch.ones(5120, device="cuda", dtype=torch.bfloat16)
            for name in (qwen.W.pre_ln_gamma, qwen.W.post_ln_gamma)
        }
        with (
            patch.dict(os.environ, {"RTP_LLM_SM120_FUSION": "1"}),
            patch.object(qwen, "DenseMLP", return_value=mlp),
            patch.object(
                qwen, "Qwen3NextGatedDeltaNet", return_value=torch.nn.Module()
            ),
        ):
            for attn_tp, ffn_tp in [(2, 1), (2, 2), (1, 2), (2, 4)]:
                parallelism = SimpleNamespace(
                    get_attn_tp_size=lambda: attn_tp,
                    get_ffn_tp_size=lambda: ffn_tp,
                )
                layer = qwen.Qwen3NextDecoderLayer(
                    config, parallelism, weights, 0, None
                )
                self.assertEqual(layer._fused_ar_norm, (attn_tp, ffn_tp) == (2, 2))

    def test_fused_gemm_swiglu_matches_original_mlp(self):
        up, down = self._fused_mlp_layers()
        for m in (1, 5, 17, 31, 32, 33, 40, 64):
            with self.subTest(m=m):
                x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
                expected = down.forward_with_silu(up(x))
                actual = up.forward_fused_mlp(x, down)
                self.assertTrue(
                    torch.equal(expected.view(torch.int16), actual.view(torch.int16))
                )

    @torch.inference_mode()
    def test_fused_gemm_graph_reads_current_operands(self):
        up, down = self._fused_mlp_layers()
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        up.forward_fused_mlp(x, down)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = up.forward_fused_mlp(x, down)
        x.normal_()
        up.alpha.mul_(0.9)
        down.input_scale_inv.fill_(17.2)
        # The same bytewise change commutes with the gate/up row permutation.
        up.weight.bitwise_xor_(1)
        up._swiglu_weights[0].bitwise_xor_(1)
        up.weight_scales.view(torch.uint8).bitwise_xor_(1)
        up._swiglu_weights[1].view(torch.uint8).bitwise_xor_(1)
        graph.replay()
        expected = down.forward_with_silu(up(x))
        self.assertTrue(
            torch.equal(expected.view(torch.int16), actual.view(torch.int16))
        )
        graph.reset()

    def test_fused_gemm_unsupported_and_disabled_preserve_input(self):
        up, down = self._fused_mlp_layers()
        for shape, dtype in [((65, 5120), torch.bfloat16), ((5, 5120), torch.float16)]:
            x = torch.randn(*shape, device="cuda", dtype=dtype)
            before = x.clone()
            self.assertIsNone(up.forward_fused_mlp(x, down))
            self.assertTrue(torch.equal(x, before))
        up, down = self._fused_mlp_layers(enabled=False)
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(up.forward_fused_mlp(x, down))

    def test_dense_mlp_fused_reduces_exactly_once(self):
        from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP

        up, down = self._fused_mlp_layers()
        mlp = DenseMLP.__new__(DenseMLP)
        torch.nn.Module.__init__(mlp)
        mlp.w4a16 = None
        mlp.is_gated = True
        mlp.up_proj, mlp.down_proj = up, down
        mlp.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 2)
        x = torch.randn(5, 5120, device="cuda", dtype=torch.bfloat16)
        expected = down.forward_with_silu(up(x))
        with patch(
            "rtp_llm.models_py.modules.hybrid.dense_mlp.all_reduce",
            side_effect=lambda y, **kw: y + 1,
        ) as reduce:
            actual = mlp.forward(x)
            self.assertEqual(reduce.call_count, 1)
            self.assertTrue(torch.equal(actual, expected + 1))
            actual = mlp.forward(x, skip_allreduce=True)
            self.assertEqual(reduce.call_count, 1)
            self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_fused_packed_outputs_preserve_guards(self):
        from rtp_llm.models_py.kernels.cuda.sm120_fp4_swiglu import _compile

        up, down = self._fused_mlp_layers()
        w, ws = up._swiglu_weights
        for m in (1, 5, 32, 33, 40, 64):
            x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
            q, sf = fp4_quantize(x, up.input_scale_inv)
            output_storage = torch.full(
                (m * 4352 + 128,), 0xA5, device="cuda", dtype=torch.uint8
            )
            scale_storage = torch.full(
                (128 * 544 + 128,), 0xA5, device="cuda", dtype=torch.uint8
            )
            output = output_storage[64:-64].reshape(m, 4352)
            scales = scale_storage[64:-64]
            compiled = _compile(q.device, m <= 32)

            def call():
                compiled(
                    q,
                    w,
                    output,
                    scales,
                    down.input_scale_inv,
                    1,
                    136,
                    80,
                    sf.data_ptr(),
                    ws.data_ptr(),
                    up.alpha,
                )

            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(3):
                graph.replay()
            self.assertTrue(bool((output_storage[:64] == 0xA5).all()))
            self.assertTrue(bool((output_storage[-64:] == 0xA5).all()))
            self.assertTrue(bool((scale_storage[:64] == 0xA5).all()))
            self.assertTrue(bool((scale_storage[-64:] == 0xA5).all()))
            graph.reset()


if __name__ == "__main__":
    unittest.main()
