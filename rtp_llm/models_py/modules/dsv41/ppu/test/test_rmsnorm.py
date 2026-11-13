"""Phase-independent norm dispatch and explicit native replay contracts."""

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import ptpc_norm_quant, rmsnorm


def reference(value, gamma, eps):
    """Test-only arithmetic; production norm never imports this oracle."""
    x = value.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (y if gamma is None else y * gamma.float()).to(value.dtype)


def assert_bits(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


def source_path(name):
    normal = Path(__file__).resolve().parents[1] / name
    return normal if normal.exists() else Path(__file__).with_name(name)


def norm_class():
    """Load the real class without importing unrelated C++ model bindings."""
    tree = ast.parse(source_path("layers.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PPUSharedRMSNorm"
    )

    class BaseNorm(torch.nn.Module):
        def __init__(self, weight, eps):
            super().__init__()
            self.weight, self.variance_epsilon = weight, eps

    namespace = {
        "torch": torch,
        "BaseNorm": BaseNorm,
        "quant_mode": lambda: "ptpc",
        "MXFP4_FP8_PTPC": "ptpc",
        "rms_norm": rmsnorm.rms_norm,
        "try_rms_norm_quant": rmsnorm.try_rms_norm_quant,
    }
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(source_path("layers.py")),
            "exec",
        ),
        namespace,
    )
    return namespace["PPUSharedRMSNorm"], namespace


class RMSNormDispatchTest(unittest.TestCase):
    @torch.no_grad()
    def test_model_class_is_phase_independent_and_keeps_call_count(self):
        cls, namespace = norm_class()
        value = torch.ones(3, 8, dtype=torch.bfloat16)
        op = cls(torch.ones(8, dtype=torch.bfloat16))
        native = Mock(return_value=value)
        namespace["rms_norm"] = native
        for ptpc in (False, True):
            op._ptpc = ptpc
            for phase in ("prefill", "decode", "verify", "proposal"):
                op.runtime = phase
                with (
                    self.subTest(ptpc=ptpc, phase=phase),
                    patch.dict(
                        os.environ,
                        {
                            "RTP_V41_PPU_GRAPH_FULL_RMSNORM": "0",
                            "RTP_V41_PPU_CP8_FLASHINFER_RMSNORM": "0",
                            "RTP_V41_PPU_VERIFY_RMSNORM_EPILOGUE": "1",
                        },
                    ),
                ):
                    self.assertIs(op(value, allow_strided=True), value)
        self.assertEqual(op.calls, 8)
        self.assertEqual(native.call_count, 8)

    @torch.no_grad()
    def test_model_quant_request_has_priority_and_falls_back_once(self):
        cls, namespace = norm_class()
        value = torch.ones(3, 8, dtype=torch.bfloat16)
        op = cls(torch.ones(8, dtype=torch.bfloat16))
        payload, scale = object(), object()
        fused = Mock(return_value=(value, payload, scale))
        ordinary = Mock(return_value=value)
        namespace["try_rms_norm_quant"], namespace["rms_norm"] = fused, ordinary
        actual = op.forward_with_ptpc_input(value, allow_strided=True)
        self.assertIs(actual[0], value)
        self.assertEqual(actual[1], (payload, scale))
        ordinary.assert_not_called()
        self.assertEqual(op.calls, 1)
        fused.return_value = None
        actual = op.forward_with_ptpc_input(value)
        self.assertIs(actual[0], value)
        self.assertIsNone(actual[1])
        self.assertEqual(ordinary.call_count, 1)
        self.assertEqual(op.calls, 2)

    def test_production_wrappers_have_no_composed_norm_or_retired_flags(self):
        wanted = {
            "layers.py": {"forward", "forward_with_ptpc_input"},
            "ptpc_norm.py": {"ptpc_rms_norm"},
            "prefill.py": {"_rms_norm"},
            "graph_attention_layer.py": {"_normalize_graph_value"},
        }
        for name, methods in wanted.items():
            tree = ast.parse(source_path(name).read_text())
            nodes = list(tree.body)
            if name == "layers.py":
                nodes = next(
                    n.body
                    for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == "PPUSharedRMSNorm"
                )
            for node in nodes:
                if isinstance(node, ast.FunctionDef) and node.name in methods:
                    calls = {
                        n.func.attr
                        for n in ast.walk(node)
                        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    }
                    self.assertFalse(
                        calls & {"square", "mean", "rsqrt", "pow"}, (name, node.name)
                    )
        layers = source_path("layers.py").read_text()
        for flag in (
            "GRAPH_FULL_RMSNORM",
            "VERIFY_RMSNORM_EPILOGUE",
            "VERIFY_RMSNORM_WIDEN_SQUARE",
            "CP8_FLASHINFER_RMSNORM",
            "CP8_RMSNORM_EPILOGUE",
        ):
            self.assertNotIn("RTP_V41_PPU_" + flag, layers)

    @torch.no_grad()
    def test_dense_shapes_use_flashinfer_without_torch_norm_operations(self):
        for shape in ((8,), (3, 8), (2, 3, 8)):
            value = torch.randn(shape).bfloat16()
            gamma = torch.randn(8).bfloat16()
            expected = reference(value, gamma, 1e-6)
            provider = Mock(return_value=expected.view(-1, 8))
            with (
                self.subTest(shape=shape),
                patch.object(rmsnorm, "_flashinfer_rmsnorm", return_value=provider),
                patch.object(
                    rmsnorm,
                    "_strided_rmsnorm",
                    side_effect=AssertionError("dense input"),
                ),
                patch.object(
                    torch, "rsqrt", side_effect=AssertionError("composed norm")
                ),
                patch.object(
                    torch.Tensor, "square", side_effect=AssertionError("composed norm")
                ),
            ):
                actual = rmsnorm.rms_norm(value, gamma, 1e-6)
            assert_bits(self, actual, expected)
            rows, weight = provider.call_args.args
            self.assertTrue(torch._C._is_alias_of(rows, value))
            self.assertIs(weight, gamma)
            self.assertEqual(provider.call_args.kwargs, {"eps": 1e-6})

    @torch.no_grad()
    def test_gapped_rows_and_storage_offset_do_not_pack(self):
        for shape in ((3, 16), (2, 3, 16)):
            base = torch.randn(shape).bfloat16()
            value = base[..., 2:10]
            gamma = torch.randn(8).bfloat16()
            expected = reference(value, gamma, 1e-6)
            before = base.clone()
            provider = Mock(return_value=expected.view(-1, 8))
            with (
                self.subTest(shape=shape),
                patch.object(rmsnorm, "_strided_rmsnorm", return_value=provider),
                patch.object(
                    rmsnorm,
                    "_flashinfer_rmsnorm",
                    side_effect=AssertionError("packed input"),
                ),
                patch.object(
                    torch.Tensor,
                    "contiguous",
                    side_effect=AssertionError("strided copy"),
                ),
            ):
                actual = rmsnorm.rms_norm(value, gamma, 1e-6)
            rows = provider.call_args.args[0]
            self.assertTrue(torch._C._is_alias_of(rows, value))
            self.assertEqual(rows.data_ptr(), value.data_ptr())
            self.assertEqual(rows.stride(), (16, 1))
            assert_bits(self, actual, expected)
            assert_bits(self, base, before)

    @torch.no_grad()
    def test_column_strides_use_packed_library_input(self):
        value = torch.randn(3, 16).bfloat16()[:, ::2]
        gamma = torch.ones(8).bfloat16()
        expected = reference(value, gamma, 1e-6)
        provider = Mock(return_value=expected)
        with (
            patch.object(rmsnorm, "_flashinfer_rmsnorm", return_value=provider),
            patch.object(
                rmsnorm, "_strided_rmsnorm", side_effect=AssertionError("column stride")
            ),
        ):
            assert_bits(self, rmsnorm.rms_norm(value, gamma, 1e-6), expected)
        dense = provider.call_args.args[0]
        self.assertTrue(dense.is_contiguous())
        self.assertFalse(torch._C._is_alias_of(dense, value))
        assert_bits(self, dense, value)

    @torch.no_grad()
    def test_unweighted_norm_and_empty_input(self):
        value = torch.randn(3, 8).bfloat16()
        expected = reference(value, None, 1e-6)
        provider = Mock(return_value=expected)
        with patch.object(rmsnorm, "_flashinfer_rmsnorm", return_value=provider):
            assert_bits(self, rmsnorm.rms_norm(value, None, 1e-6), expected)
        self.assertIsNone(provider.call_args.args[1])
        with (
            patch.object(
                rmsnorm, "_flashinfer_rmsnorm", side_effect=AssertionError("empty")
            ),
            patch.object(
                rmsnorm, "_strided_rmsnorm", side_effect=AssertionError("empty")
            ),
        ):
            empty = torch.empty(2, 0, 8, dtype=torch.bfloat16)
            assert_bits(self, rmsnorm.rms_norm(empty, None, 1e-6), empty)

    @torch.no_grad()
    def test_provider_failures_propagate_without_retry(self):
        value = torch.randn(3, 16).bfloat16()[:, :8]
        gamma = torch.ones(8).bfloat16()
        broken = Mock(side_effect=RuntimeError("native failure"))
        with (
            patch.object(rmsnorm, "_strided_rmsnorm", return_value=broken),
            patch.object(
                rmsnorm, "_flashinfer_rmsnorm", side_effect=AssertionError("retry")
            ),
            self.assertRaisesRegex(RuntimeError, "native failure"),
        ):
            rmsnorm.rms_norm(value, gamma, 1e-6)
        self.assertEqual(broken.call_count, 1)

    def test_validation_and_autograd_do_not_fall_back(self):
        value = torch.ones(3, 8, dtype=torch.bfloat16)
        with patch.object(
            rmsnorm, "_flashinfer_rmsnorm", side_effect=AssertionError("invalid")
        ):
            for gamma, eps in ((torch.ones(7), 1e-6), (None, -1), (None, float("nan"))):
                with self.subTest(eps=eps), self.assertRaises(ValueError):
                    rmsnorm.rms_norm(value, gamma, eps)
            with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "inference"):
                rmsnorm.rms_norm(value.requires_grad_(), None, 1e-6)

    def test_library_dtype_contract_is_checked_before_native_dispatch(self):
        value = torch.ones(3, 8, dtype=torch.bfloat16)
        with patch.object(
            rmsnorm, "_flashinfer_rmsnorm", side_effect=AssertionError("dtype")
        ):
            with self.assertRaisesRegex(ValueError, "dtype"):
                rmsnorm.rms_norm(value, torch.ones(8, dtype=torch.float32), 1e-6)
            with self.assertRaisesRegex(ValueError, "BF16 or FP16"):
                rmsnorm.rms_norm(value.float(), None, 1e-6)

    @torch.no_grad()
    def test_quant_producer_accepts_strides_and_avoids_ordinary_norm(self):
        value = torch.randn(3, 2560).bfloat16()[:, :1280]
        gamma = torch.ones(1280).bfloat16()
        produced = (object(), object(), object())
        with (
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant.ptpc_fusion_enabled",
                return_value=True,
            ),
            patch.object(
                ptpc_norm_quant, "ptpc_rms_norm_quant_candidate", return_value=produced
            ) as native,
            patch.object(
                rmsnorm, "rms_norm", side_effect=AssertionError("unfused norm")
            ),
        ):
            self.assertIs(rmsnorm.try_rms_norm_quant(value, gamma, 1e-6), produced)
        native.assert_called_once_with(value, gamma, 1e-6, allow_strided=True)

    @torch.no_grad()
    def test_unsupported_quant_returns_before_launch(self):
        value = torch.ones(3, 8, dtype=torch.bfloat16)
        gamma = torch.ones(8).bfloat16()
        with (
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant.ptpc_fusion_enabled",
                return_value=True,
            ),
            patch.object(
                ptpc_norm_quant, "ptpc_rms_norm_quant_candidate", return_value=None
            ),
            patch.object(
                rmsnorm,
                "rms_norm",
                side_effect=AssertionError("ordinary consumer decides"),
            ),
        ):
            self.assertIsNone(rmsnorm.try_rms_norm_quant(value, gamma, 1e-6))
        with (
            patch(
                "rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant.ptpc_fusion_enabled",
                return_value=False,
            ),
            patch.object(
                ptpc_norm_quant,
                "ptpc_rms_norm_quant_candidate",
                side_effect=AssertionError("disabled"),
            ),
        ):
            self.assertIsNone(rmsnorm.try_rms_norm_quant(value, gamma, 1e-6))


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_RMSNORM_DEVICE") == "1", "explicit PPU norm test"
)
class RMSNormDeviceTest(unittest.TestCase):
    @torch.no_grad()
    def test_dense_matches_actual_flashinfer(self):
        from flashinfer.norm import rmsnorm as native

        for width in (128, 512, 513, 1280, 1281, 5120, 5121):
            for rows in (0, 1, 24, 32, 140):
                value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                expected = (
                    torch.empty_like(value)
                    if not rows
                    else native(value, gamma, eps=1e-6)
                )
                assert_bits(self, rmsnorm.rms_norm(value, gamma, 1e-6), expected)

    @torch.no_grad()
    def test_strided_replay_recomputes_changed_inputs_and_gamma(self):
        native = rmsnorm._strided_rmsnorm()
        for width in (512, 1280, 5120):
            backing = torch.randn(4, width * 2, device="cuda", dtype=torch.bfloat16)
            value = backing[:, :width]
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            rmsnorm.rms_norm(value, gamma, 1e-6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = rmsnorm.rms_norm(value, gamma, 1e-6)
            for magnitude in (0.0, 0.125, 1.0, 16.0):
                backing.copy_(torch.randn_like(backing) * magnitude)
                gamma.copy_(torch.randn_like(gamma))
                graph.replay()
                expected = native(value.contiguous(), gamma, 1e-6)
                assert_bits(self, actual, expected)

    @torch.no_grad()
    def test_strided_fused_quant_replays_without_ordinary_norm(self):
        for width in (512, 1280):
            backing = torch.randn(24, 2 * width, device="cuda", dtype=torch.bfloat16)
            value = backing[:, :width]
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}):
                self.assertIsNotNone(rmsnorm.try_rms_norm_quant(value, gamma, 1e-6))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = rmsnorm.try_rms_norm_quant(value, gamma, 1e-6)
                for magnitude in (0.0, 0.125, 1.0, 16.0):
                    backing.copy_(torch.randn_like(backing) * magnitude)
                    gamma.copy_(torch.randn_like(gamma))
                    graph.replay()
                    expected = ptpc_norm_quant.ptpc_rms_norm_quant_candidate(
                        value.contiguous(), gamma, 1e-6
                    )
                    self.assertIsNotNone(expected)
                    for left, right in zip(actual, expected):
                        assert_bits(self, left, right)


if __name__ == "__main__":
    unittest.main()
