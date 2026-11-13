"""Host admission checks plus optional SM120 numerical/Graph integration tests."""

import os
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.gated_residual import Qwen4GatedResidual
from rtp_llm.models_py.modules.qwen4_exp.sm120_gr import DecodeGrMixer


def _module(weights, enabled, *, role="DECODE", gate=True, is_mtp=False, **kwargs):
    weights = weights if gate else (*weights[:3], None)
    with patch.dict(os.environ, RTP_QWEN4_FUSED_GR=str(int(enabled))):
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

    def test_missing_dependency_is_optional_until_admitted(self):
        with patch.dict(sys.modules, {"rtp_kernel.gr_mixer": None}):
            with patch.object(
                DecodeGrMixer,
                "create",
                side_effect=AssertionError("disabled path admitted"),
            ):
                self.assertIsNone(_module(self.weights, False)._decode_gr)
                for role in ("PREFILL", "PDFUSION"):
                    self.assertIsNone(_module(self.weights, True, role=role)._decode_gr)
                with patch.dict(os.environ, RTP_QWEN4_FUSED_GR_MTP="0"):
                    self.assertIsNone(
                        _module(self.weights, True, is_mtp=True)._decode_gr
                    )
            # Only CUDA admission metadata is mocked. The package import and
            # actionable failure are exercised in the real factory on any host.
            tensors = []
            for shape in ((320, 10240), (10240, 320), (4, 10240)):
                tensor = Mock(
                    spec=torch.Tensor,
                    shape=shape,
                    is_cuda=True,
                    device=torch.device("cuda:0"),
                    dtype=torch.bfloat16,
                    requires_grad=False,
                )
                tensor.is_contiguous.return_value = True
                tensor.data_ptr.return_value = 256
                tensors.append(tensor)
            with patch.object(
                torch.cuda, "get_device_capability", return_value=(12, 0)
            ):
                with patch.object(torch.version, "hip", None):
                    with self.assertRaisesRegex(RuntimeError, "rtp-kernel"):
                        DecodeGrMixer.create(*tensors)

    def test_mtp_switch_does_not_disable_target(self):
        # This is Python admission policy; it must run without an SM120 GPU.
        for gate in (True, False):
            for enabled in (False, True):
                for mtp_enabled in ("0", "1", None):
                    with self.subTest(gate=gate, enabled=enabled, mtp=mtp_enabled):
                        with patch.dict(os.environ), patch.object(
                            DecodeGrMixer, "create", return_value=object()
                        ):
                            os.environ.pop("RTP_QWEN4_FUSED_GR_MTP", None)
                            if mtp_enabled is not None:
                                os.environ["RTP_QWEN4_FUSED_GR_MTP"] = mtp_enabled
                            target = _module(self.weights, enabled, gate=gate)
                            draft = _module(
                                self.weights, enabled, gate=gate, is_mtp=True
                            )
                        self.assertEqual(target._decode_gr is not None, enabled)
                        self.assertEqual(
                            draft._decode_gr is not None, enabled and mtp_enabled != "0"
                        )

    def test_model_builders_propagate_mtp_to_layer_and_output_mixers(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.model_desc.qwen4_exp import (
            Qwen4ExpDecoderLayer,
            Qwen4ExpModel,
        )
        from rtp_llm.utils.model_weight import W

        def weights(prefix, gate=True):
            parts = (
                ("norm", "mix_down", "mix_up", "inject")
                if gate
                else ("norm", "mix_down", "mix_up")
            )
            return {
                getattr(W, f"qwen4_hc_{prefix}_{part}"): tensor
                for part, tensor in zip(parts, self.weights)
            }

        globals_ = weights("mixer", gate=False)
        owner = SimpleNamespace(
            global_weights=globals_, get_global_weight=globals_.__getitem__
        )
        for is_mtp in (False, True):
            config = SimpleNamespace(is_mtp=is_mtp, hc_mult=4, layernorm_eps=1e-6)
            model = SimpleNamespace(
                _execution_role="DECODE",
                parallelism_config=SimpleNamespace(role_type="DECODE"),
            )
            with patch.object(
                DecodeGrMixer, "create", return_value=object()
            ), patch.dict(
                os.environ, RTP_QWEN4_FUSED_GR="1", RTP_QWEN4_FUSED_GR_MTP="0"
            ):
                Qwen4ExpDecoderLayer._build_residual_modules(
                    model, config, {**weights("attn"), **weights("mlp")}
                )
                Qwen4ExpModel._build_output_head(model, config, owner)
            for unit in (
                model.attn_hyper_connection,
                model.mlp_hyper_connection,
                model.hyper_connection_mixer,
            ):
                self.assertEqual(unit._decode_gr is not None, not is_mtp)


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

    def test_packed_up_is_static_and_restart_protected(self):
        fused = _module(self.weights, True)
        self.assertTrue(
            getattr(fused.mix_up, "_rtp_requires_restart_for_update", False)
        )
        with patch.object(
            self.kernel,
            "prepare_up",
            side_effect=AssertionError("repacking during forward"),
        ):
            fused(self.random(8, 10240))
            fused(self.random(32, 10240))

    def test_unwarmed_capture_fails_before_kernel_launch(self):
        fused = _module(self.weights, True)
        x = self.random(8, 10240)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with patch.object(
                self.kernel, "fused_gr", side_effect=AssertionError("launched")
            ):
                with self.assertRaisesRegex(RuntimeError, "must be warmed"):
                    fused._forward_normed(x, x)


if __name__ == "__main__":
    unittest.main()
