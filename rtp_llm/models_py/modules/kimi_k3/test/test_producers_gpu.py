"""Focused one-GPU correctness checks for the BF16 producer integration."""

import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton.language as tl
from torch import nn

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3 import producers
from rtp_llm.models_py.modules.kimi_k3.mla import KimiK3MLA
from rtp_llm.models_py.triton_kernels.kimi_k3 import bf16_producers as kernels
from rtp_llm.models_py.triton_kernels.kimi_kda.rms_norm_gate import (
    kimi_kda_rms_norm_sigmoid_gate,
)
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W

assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)
assert_near = partial(torch.testing.assert_close, rtol=1e-2, atol=1e-3)
randn = partial(torch.randn, device="cuda", dtype=torch.bfloat16)


def latent_fixture(rows, width):
    return randn(rows, width + 96)[:, 32 : 32 + width], randn(width)


def mla_fixture():
    module = KimiK3MLA.__new__(KimiK3MLA)
    nn.Module.__init__(module)
    module._fp8_enabled = False
    module.use_output_gate = True
    module.output_gate_op = producers.SigmoidGate()
    return module


def mla_fixture_for_role(role):
    config = SimpleNamespace(
        attn_config=SimpleNamespace(
            head_num=96,
            nope_head_dim=128,
            rope_head_dim=64,
            q_lora_rank=1536,
            kv_lora_rank=512,
            v_head_dim=128,
            kernel_tokens_per_block=64,
            is_sparse=False,
        ),
        k3_attention_quant_config=object(),
        quant_config=None,
        k3_runtime_config=SimpleNamespace(mla_use_nope=True, mla_use_output_gate=True),
    )
    parallel = SimpleNamespace(
        get_attn_tp_size=lambda: 8,
        decode_cp_q_replicated=False,
        role_type=role,
    )
    weights = {
        W.mla_q_a_ln_gamma: torch.ones(1536, device="cuda", dtype=torch.bfloat16),
        W.mla_kv_a_ln_gamma: torch.ones(512, device="cuda", dtype=torch.bfloat16),
        W.mla_fusedqkrope_w: torch.empty(0, device="cuda"),
        W.mla_fusedqkrope_s: torch.empty(0, device="cuda"),
        W.attn_gate_s: torch.empty(0, device="cuda"),
    }
    # Projection weights are irrelevant here: keep the real constructor and
    # native/Triton norms, replacing only the unused GEMM factory boundary.
    with patch.object(
        LinearFactory,
        "create_linear_from_weights",
        side_effect=lambda *a, **kw: nn.Identity(),
    ):
        return KimiK3MLA(config, parallel, weights)


def capture(call, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call()
    return graph, outputs


def kda_reference(x, gate, weight, eps):
    return (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * weight.float()
        * torch.sigmoid(gate.float())
    ).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeBf16ProducersCudaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260918)

    def test_int64_token_and_head_addressing(self):
        # One backing allocation exercises all three kernels. An int32 offset
        # regression wraps into the first 1 Ki elements instead of going OOB.
        stride = 2**30 + 128
        storage = torch.empty(2**32 + 1024, device="cuda", dtype=torch.bfloat16)

        latent = storage.as_strided((3, 512), (stride, 1))
        dense_latent = randn(3, 512)
        for row in range(3):
            latent[row].copy_(dense_latent[row])
        weight = randn(512)
        actual = kernels.latent_rmsnorm(latent, weight, 1e-6)
        assert_near(actual, RMSNorm(weight, 1e-6)(dense_latent))

        storage[:1024].zero_()
        for head_major in (False, True):
            shape = (1, 3, 128) if head_major else (3, 1, 128)
            strides = (128, stride, 1) if head_major else (stride, 128, 1)
            x = storage.as_strided(shape, strides, 2**31)
            gate = storage.as_strided(shape, strides, 2**31 + 512)
            dense_x, dense_gate = randn(shape), randn(shape)
            for index in range(3):
                view = (0, index) if head_major else (index, 0)
                x[view].copy_(dense_x[view])
                gate[view].copy_(dense_gate[view])

            expected_gate = (dense_x * torch.sigmoid(dense_gate)).reshape(shape[0], -1)
            assert_exact(kernels.sigmoid_gate(x, gate), expected_gate)

            if hasattr(tl, "gather"):
                gamma = randn(128)
                actual = producers.KdaOutputNorm(gamma, 1e-6)(x, gate, "decode")
                assert_near(actual, kda_reference(dense_x, dense_gate, gamma, 1e-6))

        del latent, x, gate, storage

    def test_noncontiguous_native_norm_and_gate_integration(self):
        module = mla_fixture()
        for width in (512, 1536):
            latent, weight = latent_fixture(32, width)
            reference = RMSNorm(weight, 1e-6)
            norm = producers.Bf16RMSNorm(weight, 1e-6)
            expected = reference(latent.contiguous())
            actual = norm(latent)
            assert_near(actual, expected)
            self.assertTrue(actual.is_contiguous())

        context = randn(12, 32, 144).transpose(0, 1)[..., :128]
        gate = randn(32, 1576)[:, 40:]
        prepared = module._prepare_output_layout(context, (32,), gate)
        self.assertIs(prepared, context)
        actual = module._apply_output_gate(prepared, gate)
        expected = context.reshape(32, -1) * torch.sigmoid(gate)
        self.assertTrue(actual.is_contiguous())
        assert_exact(actual, expected)

    def test_kda_decode_strides_extremes_and_rank_alias(self):
        x = randn(12, 3, 144).transpose(0, 1)[..., :128]
        gate = randn(3, 12, 136)[..., :128]
        x[0, 0].zero_()
        x[0, 1].fill_(1e-7)
        gate[..., :5] = torch.tensor([-100, -20, 0, 20, 100], device="cuda")
        weight = randn(128)
        producer = producers.KdaOutputNorm(weight, 1e-5)
        for rank4 in (False, True):
            source = x.unsqueeze(0) if rank4 else x
            gates = gate.unsqueeze(0) if rank4 else gate
            actual = producer(source, gates, "decode")
            assert_near(actual, kda_reference(source, gates, weight, 1e-5))

    def test_prefill_kda_preserves_original_arithmetic_bitwise(self):
        x = randn(32, 12, 144)[..., :128]
        gate = randn(32, 12, 136)[..., :128]
        weight = randn(128, dtype=torch.float32)
        expected = kimi_kda_rms_norm_sigmoid_gate(x, gate, weight, 1e-6)
        actual = producers.KdaOutputNorm(weight, 1e-6)(x, gate, "prefill")
        assert_exact(actual, expected)

    def test_target_verify_matches_prefill_arithmetic(self):
        x = randn(7, 96, 128)
        gate = randn(7, 96, 128)
        weight = randn(128, dtype=torch.float32)
        expected = kimi_kda_rms_norm_sigmoid_gate(x, gate, weight, 1e-6)
        actual = producers.KdaOutputNorm(weight, 1e-6)(x, gate, "target_verify")
        assert_near(actual, expected)

    def test_sigmoid_rounds_to_bf16_before_multiplication(self):
        x = torch.full((1, 12, 128), 1.5, device="cuda", dtype=torch.bfloat16)
        gate = torch.full_like(x, 0.2)
        actual = producers.SigmoidGate()(x, gate)
        # Fusing the FP32 sigmoid and multiply would instead yield 0.82421875.
        expected = torch.full((1, 1536), 0.828125, device="cuda", dtype=torch.bfloat16)
        assert_exact(actual, expected)

    def test_two_stream_graph_replay_reads_changing_values(self):
        weight, latent_weight = randn(128), randn(512)
        kda = producers.KdaOutputNorm(weight, 1e-6)
        native_norm = RMSNorm(latent_weight, 1e-5)
        norm = producers.Bf16RMSNorm(latent_weight, 1e-5)
        gate_op = producers.SigmoidGate()

        cases = []
        for _ in range(2):
            stream = torch.cuda.Stream()
            x = randn(12, 7, 136).transpose(0, 1)[..., :128]
            gate = randn(7, 12, 144)[..., :128]
            latent = randn(7, 560)[:, :512]

            def run():
                return kda(x, gate, "decode"), norm(latent), gate_op(x, gate)

            graph, outputs = capture(run, stream)
            cases.append((stream, graph, x, gate, latent, outputs))

        for replay in range(3):
            pending = []
            for index, (stream, graph, x, gate, latent, outputs) in enumerate(cases):
                with torch.cuda.stream(stream):
                    x.fill_(0.125 * (1 + index + replay))
                    gate.fill_(index - replay)
                    latent.fill_(((-1) ** (index + replay)) * 1e-4 * (1 + replay))
                    graph.replay()
                    expected = (
                        kda_reference(x, gate, weight, 1e-6),
                        native_norm(latent.contiguous()),
                        (x * torch.sigmoid(gate)).reshape(7, -1),
                    )
                    pending.append((stream, outputs, expected))
            # Both independent buffers are enqueued before either stream waits.
            for stream, outputs, expected in pending:
                stream.synchronize()
                assert_near(outputs[0], expected[0])
                assert_near(outputs[1], expected[1])
                assert_exact(outputs[2], expected[2])

        for stream, graph, *_ in cases:
            stream.synchronize()
            graph.reset()

    def test_fp8_decode_kv_norm_consumes_strided_latent_without_staging(self):
        optimized = mla_fixture_for_role(RoleType.DECODE)
        native_norm = RMSNorm(optimized._kv_a_norm, 1e-6)
        for rows in (1, 2, 4, 8, 16, 32):
            with self.subTest(rows=rows):
                latent = torch.randn(
                    rows, 1536 + 512 + 64, device="cuda", dtype=torch.bfloat16
                )[:, 1536:2048]
                reference = native_norm(latent.contiguous())
                call = lambda: optimized._normalize_latent(
                    optimized.kv_a_layernorm, latent
                )
                result = call()
                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.dtype, torch.bfloat16)
                torch.testing.assert_close(result, reference, rtol=1e-2, atol=1e-3)
                if rows == 32:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = call()
                    for _ in range(3):
                        latent.normal_()
                        graph.replay()
                        torch.testing.assert_close(captured, call(), rtol=0, atol=0)
                        reference = native_norm(latent.contiguous())
                        torch.testing.assert_close(
                            captured, reference, rtol=1e-2, atol=1e-3
                        )

    def test_fp8_prefill_keeps_quantized_and_retained_bf16_kv(self):
        module = mla_fixture_for_role(RoleType.PREFILL)
        latent = torch.randn(32, 576, device="cuda", dtype=torch.bfloat16)[:, :512]
        actual = module._normalize_latent(module.kv_a_layernorm, latent)
        expected = module.kv_a_layernorm(latent.contiguous())
        self.assertIsInstance(actual, QuantizedActivation)
        torch.testing.assert_close(
            actual.values.view(torch.uint8),
            expected.values.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual.scale_wire, expected.scale_wire, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.bf16, expected.bf16, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
