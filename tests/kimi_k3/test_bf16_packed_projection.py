"""BF16 MLA packed projection without the optional MoE/DeepGEMM extension."""

import builtins
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.kimi_k3 import linear


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Bf16PackedProjectionTest(unittest.TestCase):
    def setUp(self):
        linear._optional_bf16_packed_gemm.cache_clear()
        self.addCleanup(linear._optional_bf16_packed_gemm.cache_clear)
        self.fallback = patch.object(
            linear, "_optional_bf16_packed_gemm", return_value=None
        )
        self.fallback.start()
        self.addCleanup(self.fallback.stop)

    def projection(self, heads=3, splits=(128, 64, 128), rank=512):
        torch.manual_seed(6504)
        width = heads * (splits[0] + splits[2])
        weight = torch.randint(-2, 3, (rank, width), device="cuda").bfloat16() / 8
        return linear.KimiK3Bf16Linear(weight)

    def assert_packed(self, actual, expected, splits, gap=7):
        left, middle, right = splits
        heads = expected.shape[1] // (left + right)
        shaped = actual.view(actual.shape[0], heads, sum(splits))
        expected = expected.view(expected.shape[0], heads, left + right)
        torch.testing.assert_close(
            shaped[..., :left], expected[..., :left], rtol=0, atol=0
        )
        torch.testing.assert_close(
            shaped[..., -right:], expected[..., left:], rtol=0, atol=0
        )
        self.assertTrue(torch.all(shaped[..., left : left + middle] == gap))

    def test_incident_shapes_with_bounded_scratch_and_preserved_rope(self):
        # The incident had BF16 [65536 or 63840, 512] and a [3072, 512]
        # transposed-contiguous weight. Repeated dyadic rows give an exact
        # FP64 oracle without materializing another full-size projection.
        splits = (128, 64, 128)
        projection = self.projection(heads=12, splits=splits)
        basis = torch.randint(-2, 3, (8, 512), device="cuda").bfloat16() / 8
        expected_basis = (basis.double() @ projection.weight.double().T).bfloat16()
        native = linear.bf16_linear
        for rows in (65536, 63840):
            with self.subTest(rows=rows):
                input = basis[torch.arange(rows, device="cuda") % 8]
                output = torch.full(
                    (rows, 12 * sum(splits)), 7, device="cuda", dtype=torch.bfloat16
                )
                calls = []

                def bounded(input, weight):
                    calls.append(input.shape[0])
                    self.assertLessEqual(
                        input.shape[0] * weight.shape[0] * input.element_size(),
                        linear._PACKED_PROJECTION_SCRATCH_BYTES,
                    )
                    return native(input, weight)

                with patch.object(linear, "bf16_linear", side_effect=bounded):
                    self.assertTrue(projection.supports_skip_head_mid(input, splits))
                    result = projection.forward_skip_head_mid(
                        input, splits, output=output
                    )
                self.assertIs(result, output)
                self.assertGreater(len(calls), 1)
                self.assertEqual(sum(calls), rows)
                for start in range(0, rows, 1024):
                    stop = min(start + 1024, rows)
                    expected = expected_basis[
                        torch.arange(start, stop, device="cuda") % 8
                    ]
                    self.assert_packed(result[start:stop], expected, splits)
                del input, output, result

    def test_unaligned_head_layout_and_strided_input_weight(self):
        splits = (3, 2, 5)
        projection = self.projection(heads=2, splits=splits, rank=17)
        # Exercise the row-major weight path and an input requiring a copy.
        projection.weight = projection.weight.contiguous()
        input = torch.randint(-2, 3, (13, 34), device="cuda").bfloat16()[:, ::2] / 8
        input = input.T.contiguous().T
        output = torch.full(
            (13, 2 * sum(splits)), 7, device="cuda", dtype=torch.bfloat16
        )
        with patch.object(linear, "_PACKED_PROJECTION_SCRATCH_BYTES", 2 * 16 * 4):
            projection.forward_skip_head_mid(input, splits, output=output)
        expected = (input.double() @ projection.weight.double().T).bfloat16()
        self.assert_packed(output, expected, splits)

    def test_graph_replay_updates_kv_and_keeps_rope(self):
        splits = (128, 64, 128)
        projection = self.projection(heads=1, splits=splits)
        input = torch.randint(-2, 3, (7, 512), device="cuda").bfloat16() / 8
        original = input.clone()
        output = torch.full((7, sum(splits)), 7, device="cuda", dtype=torch.bfloat16)
        with patch.object(linear, "_PACKED_PROJECTION_SCRATCH_BYTES", 2 * 256 * 3):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    projection.forward_skip_head_mid(input, splits, output=output)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = projection.forward_skip_head_mid(input, splits, output=output)
            for factor in (1, -1, 0, 2):
                input.copy_(original * factor)
                graph.replay()
                expected = (input.double() @ projection.weight.double().T).bfloat16()
                self.assert_packed(result, expected, splits)

    def test_default_path_without_optional_package(self):
        self.fallback.stop()
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "deep_gemm":
                raise ImportError("serving package has no optional DeepGEMM")
            if name.startswith("k3_native_deep_gemm") or name.endswith("moe_backend"):
                self.fail("MLA must not load the MoE backend")
            return original_import(name, *args, **kwargs)

        splits = (128, 64, 128)
        projection = self.projection(heads=12, splits=splits)
        input = torch.randint(-2, 3, (17, 512), device="cuda").bfloat16() / 8
        output = torch.full(
            (17, 12 * sum(splits)), 7, device="cuda", dtype=torch.bfloat16
        )
        with patch("builtins.__import__", side_effect=guarded):
            self.assertTrue(projection.supports_skip_head_mid(input, splits))
            projection.forward_skip_head_mid(input, splits, output=output)
        self.assert_packed(
            output, (input.double() @ projection.weight.double().T).bfloat16(), splits
        )

    def test_zero_inner_dimension_does_not_select_fused_backend(self):
        splits = (128, 64, 128)
        projection = linear.KimiK3Bf16Linear(
            torch.empty(0, 256, device="cuda", dtype=torch.bfloat16)
        )
        input = torch.empty(5, 0, device="cuda", dtype=torch.bfloat16)
        output = torch.full((5, sum(splits)), 7, device="cuda", dtype=torch.bfloat16)
        projection.forward_skip_head_mid(input, splits, output=output)
        linear._optional_bf16_packed_gemm.assert_not_called()
        self.assert_packed(
            output, torch.zeros(5, 256, device="cuda", dtype=torch.bfloat16), splits
        )

    def test_empty_invalid_and_aliasing_buffers(self):
        splits = (3, 2, 5)
        projection = self.projection(heads=2, splits=splits, rank=20)
        input = torch.zeros((2, 20), device="cuda", dtype=torch.bfloat16)
        empty = projection.forward_skip_head_mid(input[:0], splits)
        self.assertEqual(tuple(empty.shape), (0, 20))
        for invalid in ((3, 0, 5), (3, 2, 4), (3, True, 5), (3, 2)):
            self.assertFalse(projection.supports_skip_head_mid(input, invalid))
        with self.assertRaisesRegex(ValueError, "output buffer mismatch"):
            projection.forward_skip_head_mid(input, splits, output=input)
        with self.assertRaisesRegex(ValueError, "output buffer mismatch"):
            projection.forward_skip_head_mid(
                input, splits, output=projection.weight.T[:2]
            )
        with self.assertRaisesRegex(ValueError, "output buffer mismatch"):
            projection.forward_skip_head_mid(
                input, splits, output=torch.empty_like(input).float()
            )
        with self.assertRaisesRegex(ValueError, "input or head layout"):
            projection.forward_skip_head_mid(input.float(), splits)


class OptionalPackedBackendTest(unittest.TestCase):
    def setUp(self):
        linear._optional_bf16_packed_gemm.cache_clear()
        self.addCleanup(linear._optional_bf16_packed_gemm.cache_clear)

    def test_missing_standard_package_does_not_import_moe_backend(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "deep_gemm":
                raise ImportError("optional package absent")
            if name.startswith("k3_native_deep_gemm") or name.endswith("moe_backend"):
                self.fail("MLA must not load the MoE backend")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            self.assertIsNone(linear._optional_bf16_packed_gemm())

    def test_missing_optional_symbol_and_standard_symbol(self):
        with patch.dict(sys.modules, {"deep_gemm": SimpleNamespace()}):
            self.assertIsNone(linear._optional_bf16_packed_gemm())
        linear._optional_bf16_packed_gemm.cache_clear()
        fused = lambda *args, **kwargs: None
        with patch.dict(
            sys.modules,
            {"deep_gemm": SimpleNamespace(bf16_gemm_nt_skip_head_mid=fused)},
        ):
            self.assertIs(linear._optional_bf16_packed_gemm(), fused)


if __name__ == "__main__":
    unittest.main(verbosity=2)
