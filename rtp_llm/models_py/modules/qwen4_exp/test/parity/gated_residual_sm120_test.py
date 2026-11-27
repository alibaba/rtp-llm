"""Host admission checks plus optional SM120 numerical/Graph integration tests."""

import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.sm120_gr import DecodeGrMixer


def _module(weights, enabled, *, role="DECODE", gate=True, is_mtp=False, **kwargs):
    weights = weights if gate else (*weights[:3], None)
    # The unfused reference is test-only; production has no feature switch.
    if not enabled:
        with patch.object(DecodeGrMixer, "create", return_value=None):
            return Qwen4GatedResidual(
                *weights, hc_mult=4, norm_eps=1e-6, role=role, is_mtp=is_mtp, **kwargs
            )
    return Qwen4GatedResidual(
        *weights, hc_mult=4, norm_eps=1e-6, role=role, is_mtp=is_mtp, **kwargs
    )


class GatedResidualHostTest(unittest.TestCase):
    def setUp(self):
        self.weights = tuple(
            torch.zeros(shape, dtype=torch.bfloat16, device="cpu")
            for shape in ((10240,), (320, 10240), (10240, 320), (4, 10240))
        )

    def test_cpu_weights_keep_existing_path_without_dependency(self):
        with patch.dict(sys.modules, {"rtp_kernel.gr_mixer": None}):
            for gate in (True, False):
                with self.subTest(gate=gate):
                    module = _module(self.weights, True, gate=gate)
                    self.assertIsNone(module._decode_gr)
                    x = torch.ones(2, 10240, dtype=torch.bfloat16)
                    actual = module(x)
                    expected = _module(self.weights, False, gate=gate)(x)
                    self.assertIs(actual[1], x)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class GatedResidualSm120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("SM120 required")
        try:
            import rtp_kernel.gr_mixer as gr_mixer
        except ImportError as exc:
            raise unittest.SkipTest("rtp-kernel SM120 GR package required") from exc
        cls.kernel = gr_mixer

    def setUp(self):
        torch.manual_seed(61110)
        self.weights = (
            torch.ones(10240, device="cuda", dtype=torch.bfloat16),
            self.random(320, 10240),
            self.random(10240, 320),
            self.random(4, 10240),
        )

    @staticmethod
    def random(*shape):
        return (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.01
        ).contiguous()

    def assert_result(self, actual, expected):
        # The reference is the unfused BF16 implementation. FP64 is used only
        # for error norms; this is not a comparison against an FP64 forward.
        for index, (a, b) in enumerate(zip(actual, expected)):
            if b is None:
                self.assertIsNone(a)
                continue
            self.assertTrue(torch.isfinite(a).all().item())
            if index == 1:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                self.assertTrue(torch.isfinite(b).all().item())
                error = torch.linalg.vector_norm(a.double() - b.double())
                scale = torch.linalg.vector_norm(b.double())
                if scale.item() == 0:
                    self.assertEqual(error.item(), 0)
                else:
                    self.assertLess((error / scale).item(), 0.01)

    def test_all_rows_both_modes_and_residual_identity(self):
        for gate in (True, False):
            reference, fused = _module(self.weights, False, gate=gate), _module(
                self.weights, True, gate=gate
            )
            with patch.object(
                self.kernel, "fused_gr", wraps=self.kernel.fused_gr
            ) as call:
                for rows in range(1, 65):
                    x = self.random(rows, 10240)
                    actual = fused(x)
                    self.assertIs(actual[1], x)
                    self.assert_result(actual, reference(x))
                self.assertEqual(call.call_count, 64)

    def test_graph_replay_with_injection_and_without_gate(self):
        for gate in (True, False):
            reference = _module(self.weights, False, gate=gate)
            fused = _module(self.weights, True, gate=gate)
            forward = fused.forward_with_injection if gate else fused
            expected_forward = reference.forward_with_injection if gate else reference
            for rows in (1, 2, 4, 8, 16, 32, 64):
                with self.subTest(gate=gate, rows=rows):
                    inputs = [self.random(rows, 10240)]
                    if gate:
                        inputs += [self.random(rows, 2560), self.random(rows, 4)]
                    saved = forward(*inputs)
                    saved_mixed = saved[0].clone()
                    forward(*inputs)
                    # Match the runner: warm on one stream, capture on another.
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = forward(*inputs)
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(8):
                        for tensor in inputs:
                            tensor.copy_(self.random(*tensor.shape))
                        expected = expected_forward(*inputs)
                        graph.replay()
                        self.assert_result(output, expected)
                    torch.testing.assert_close(saved[0], saved_mixed, rtol=0, atol=0)

    def test_fallbacks_do_not_call_fused_kernel(self):
        for role, rows, enabled in (
            ("PREFILL", 8, True),
            ("PDFUSION", 8, True),
            ("DECODE", 65, True),
            ("DECODE", 8, False),
        ):
            with self.subTest(role=role, rows=rows, enabled=enabled):
                module = _module(self.weights, enabled, role=role)
                with patch.object(
                    self.kernel, "fused_gr", side_effect=AssertionError("wrong path")
                ):
                    x = self.random(rows, 10240)
                    actual = module(x)
                    expected = _module(self.weights, False, role=role)(x)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fp8_weights_keep_existing_path(self):
        norm, down, up, inject = self.weights
        weights = (norm, down, up.to(torch.float8_e4m3fn), inject)
        scales = (None, torch.ones((), device="cuda"), None)
        module = _module(weights, True, weight_scales=scales)
        reference = _module(weights, False, weight_scales=scales)
        self.assertIsNone(module._decode_gr)
        x = self.random(8, 10240)
        with patch.object(
            self.kernel, "fused_gr", side_effect=AssertionError("FP8 admitted")
        ):
            actual = module(x)
        for a, b in zip(actual, reference(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_model_builder_merged_views_reach_the_native_kernel(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
        from rtp_llm.utils.model_weight import W

        weights = {
            getattr(W, f"qwen4_hc_{branch}_{part}"): tensor.clone()
            for branch in ("attn", "mlp")
            for part, tensor in zip(
                ("norm", "mix_down", "mix_up", "inject"), self.weights
            )
        }
        owner = SimpleNamespace(_execution_role="DECODE")
        config = SimpleNamespace(hc_mult=4, layernorm_eps=1e-6, is_mtp=False)
        Qwen4ExpDecoderLayer._build_residual_modules(owner, config, weights)
        reference = _module(self.weights, False)
        for branch, unit in (
            ("attn", owner.attn_hyper_connection),
            ("mlp", owner.mlp_hyper_connection),
        ):
            self.assertIsNotNone(unit._decode_gr)
            self.assertTrue(
                getattr(
                    weights[getattr(W, f"qwen4_hc_{branch}_mix_up")],
                    "_rtp_requires_restart_for_update",
                    False,
                )
            )
            self.assertGreater(unit.inject.storage_offset(), 0)
            self.assertIs(
                unit.mix_down, weights[getattr(W, f"qwen4_hc_{branch}_mix_down")]
            )
            for rows in (8, 32, 64):
                x = self.random(rows, 10240)
                self.assert_result(unit(x), reference(x))

    def test_unwarmed_capture_fails_before_kernel_launch(self):
        fused = _module(self.weights, True)
        x = self.random(8, 10240)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with patch.object(
                self.kernel, "fused_gr", side_effect=AssertionError("launched")
            ):
                with self.assertRaisesRegex(RuntimeError, "must be warmed"):
                    fused._forward_normed(x, x)

    def test_model_shared_workspaces_preserve_layers_and_graph_replays(self):
        from rtp_llm.models_py.modules.qwen4_exp.sm120_gr import share_model_workspaces

        other_weights = tuple(w * 0.5 for w in self.weights)
        model = torch.nn.ModuleList(
            [
                _module(self.weights, True),
                _module(other_weights, True),
                _module(self.weights, True, gate=False),
            ]
        )
        independent = torch.nn.ModuleList([_module(self.weights, True)])
        share_model_workspaces(model)
        share_model_workspaces(independent)
        references = [
            _module(self.weights, False),
            _module(other_weights, False),
            _module(self.weights, False, gate=False),
        ]
        graphs = []
        for rows in (8, 16, 32, 48, 64):
            x = self.random(rows, 10240)
            for unit in model:
                unit(x)
            independent[0](x)
            shared = model[0]._decode_gr.workspaces[rows]
            self.assertIs(shared, model[1]._decode_gr.workspaces[rows])
            self.assertIsNot(shared, model[2]._decode_gr.workspaces[rows])
            self.assertIsNot(shared, independent[0]._decode_gr.workspaces[rows])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = [unit(x) for unit in model]
            torch.cuda.current_stream().wait_stream(stream)
            graphs.append((graph, x, outputs))
        # Alternate graphs and change inputs: generations must advance across
        # layers without overwriting another layer's separately owned output.
        for _ in range(8):
            for graph, x, outputs in reversed(graphs):
                x.copy_(self.random(*x.shape))
                expected = [unit(x) for unit in references]
                graph.replay()
                for actual, ref in zip(outputs, expected):
                    self.assert_result(actual, ref)


if __name__ == "__main__":
    unittest.main()
