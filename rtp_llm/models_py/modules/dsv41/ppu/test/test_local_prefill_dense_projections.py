"""Packed PTPC Q-a/KV-a contracts and explicit strided-norm byte gates."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, PropertyMock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_qkv as qkv_module
from rtp_llm.models_py.modules.dsv41.ppu import ptpc_norm as norm_module
from rtp_llm.models_py.modules.dsv41.ppu import ptpc_norm_quant as norm_quant_module
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import PreparedQKV
from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import quantize_ptpc_activation


def make_linear(n, k, device="cpu"):
    weight = torch.randint(-2, 3, (n, k), device=device).bfloat16().to(torch.float8_e4m3fn)
    scale = torch.ones((n, 1), dtype=torch.float32, device=device)
    return PPUPTPCLinear(weight, scale)


def assert_bits(test, actual, expected, label):
    test.assertIsNotNone(actual, label)
    test.assertEqual(actual.dtype, expected.dtype, label)
    test.assertEqual(actual.shape, expected.shape, label)
    if not actual.numel():
        return
    left = actual.contiguous().view(torch.uint8)
    right = expected.contiguous().view(torch.uint8)
    if not torch.equal(left, right):
        # Keep both exact operands on failure. Never replace this gate with a
        # tolerance merely because a different N selected a different tuner.
        root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="qkv-byte-mismatch-", dir=root))
        evidence = destination / "operands.pt"
        torch.save(
            {
                "label": label,
                "actual": actual.detach().cpu(),
                "expected": expected.detach().cpu(),
                "actual_stride": actual.stride(),
                "expected_stride": expected.stride(),
            },
            evidence,
        )
        test.fail(f"{label}: exact bytes differ; evidence: {evidence}")


def reference_norm(value, gamma, eps=1e-6):
    values = value.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * inverse * gamma.float()).to(value.dtype)


def reference_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


class LocalPrefillDenseProjectionCPU(unittest.TestCase):
    def prepare(self, first=None, second=None):
        first = make_linear(7, 16) if first is None else first
        second = make_linear(3, 16) if second is None else second
        with patch.object(qkv_module, "_physical_pair_supported", return_value=True):
            helper = PreparedQKV.try_prepare(first, second)
        self.assertIsNotNone(helper)
        return first, second, helper

    def test_pack_existing_bytes_and_channel_scales_once(self):
        first, second = make_linear(7, 16), make_linear(3, 16)
        # All byte patterns, including signed zero/NaN, are packing canaries.
        first.weight.view(torch.uint8).copy_(torch.arange(112, dtype=torch.uint8).view(7, 16))
        second.weight.view(torch.uint8).copy_(torch.arange(208, 256, dtype=torch.uint8).view(3, 16))
        first.scale.copy_(torch.arange(1, 8).float()[:, None])
        second.scale.copy_(torch.arange(21, 24).float()[:, None])
        originals = tuple(t.clone() for op in (first, second) for t in (op.weight, op.scale))
        with patch.object(torch, "cat", wraps=torch.cat) as concatenate:
            _, _, helper = self.prepare(first, second)
        self.assertEqual(concatenate.call_count, 2)
        packed = helper._linear
        for actual, expected, label in (
            (packed.weight[:7], first.weight, "Q weight"),
            (packed.weight[7:], second.weight, "KV weight"),
            (packed.scale[:7], first.scale, "Q scale"),
            (packed.scale[7:], second.scale, "KV scale"),
        ):
            assert_bits(self, actual, expected, label)
        for op in (first, second):
            self.assertFalse(torch._C._is_alias_of(packed.weight, op.weight))
            self.assertFalse(torch._C._is_alias_of(packed.scale, op.scale))
        for actual, expected in zip(
            (t for op in (first, second) for t in (op.weight, op.scale)), originals
        ):
            assert_bits(self, actual, expected, "source unchanged")

    @torch.no_grad()
    def test_one_quantizer_one_projection_and_raw_outputs_are_views(self):
        torch.manual_seed(4150)
        first, second, helper = self.prepare()
        for rows in (0, 1, 5, 140):
            with self.subTest(rows=rows):
                value = torch.randint(-2, 3, (rows, 16)).bfloat16()
                pair = first.quantize_input(value)
                expected = tuple(op.forward_quantized(*pair) for op in (first, second))
                with patch.object(
                    helper._linear, "quantize_input", wraps=helper._linear.quantize_input
                ) as quantize, patch.object(
                    helper._linear, "forward_quantized", wraps=helper._linear.forward_quantized
                ) as project, patch.object(
                    torch, "cat", side_effect=AssertionError("no forward concatenation")
                ):
                    actual = helper.project(value)
                self.assertIsNotNone(actual)
                self.assertEqual(quantize.call_count, 1)
                self.assertEqual(project.call_count, 1)
                for name, left, right in zip(("Q", "KV"), actual, expected):
                    assert_bits(self, left, right, name)
                    self.assertEqual(left.stride(), (10, 1))
                self.assertTrue(torch._C._is_alias_of(*actual))
                self.assertFalse(torch._C._is_alias_of(actual[0], value))
                self.assertEqual(actual[1].storage_offset() - actual[0].storage_offset(), 7)
                if rows > 1:
                    self.assertFalse(actual[0].is_contiguous())
                    self.assertFalse(actual[1].is_contiguous())
                with patch.object(
                    helper._linear, "quantize_input",
                    side_effect=AssertionError("prepared input must not quantize"),
                ):
                    again = helper.project(value, quantized_input=pair)
                for left, right in zip(again, expected):
                    assert_bits(self, left, right, "prepared projection")

    @torch.no_grad()
    def test_declines_before_quantization_for_invalid_input_and_pair(self):
        first, _, helper = self.prepare()
        value = torch.ones((4, 16), dtype=torch.bfloat16)
        pair = first.quantize_input(value)
        with patch.object(
            helper._linear, "quantize_input", side_effect=AssertionError("prelaunch guard")
        ), patch.object(
            helper._linear, "forward_quantized", side_effect=AssertionError("prelaunch guard")
        ):
            for invalid in (
                None, value.float(), value[0], value[None], value[:, :15],
                torch.ones((4, 32), dtype=torch.bfloat16)[:, ::2],
                value.clone().requires_grad_(), torch._neg_view(value),
                torch.empty((4, 16), dtype=torch.bfloat16, device="meta"),
            ):
                self.assertIsNone(helper.project(invalid))
            for invalid in (
                (), (pair[0],), (pair[0], pair[1], pair[1]),
                (pair[0].float(), pair[1]), (pair[0], pair[1].half()),
                (pair[0][:3], pair[1]), (pair[0], pair[1].flatten()),
                (pair[0], torch.ones(4, 2)[:, ::2]),
                (pair[0], torch.empty((4, 1), device="meta")),
                (pair[0], torch._neg_view(pair[1])),
            ):
                self.assertIsNone(helper.project(value, quantized_input=invalid))
        with torch.enable_grad():
            self.assertIsNone(helper.project(value))

    @torch.no_grad()
    def test_weight_writes_alias_writes_rebinding_and_metadata_invalidate(self):
        mutations = (
            lambda op: op.weight.view(torch.uint8).add_(1),
            lambda op: op.scale.add_(1),
            lambda op: op.scale.view(-1).add_(1),
            lambda op: setattr(op, "weight", op.weight.clone()),
            lambda op: setattr(op.weight, "data", op.weight.clone()),
            lambda op: setattr(op, "scale", op.scale.clone()),
            lambda op: setattr(op, "output_dtype", torch.float32),
            lambda op: setattr(op, "quantization", "changed"),
            lambda op: setattr(op, "_gemm", lambda *args: None),
            lambda op: op.scale.requires_grad_(True),
        )
        for mutate in mutations:
            for source in (0, 1):
                with self.subTest(mutation=mutate, source=source):
                    first, second, helper = self.prepare()
                    mutate((first, second)[source])
                    with patch.object(
                        helper._linear, "quantize_input",
                        side_effect=AssertionError("stale weight quantized"),
                    ):
                        self.assertIsNone(helper.project(torch.ones(4, 16).bfloat16()))

    def test_prepare_unsupported_is_metadata_only_and_native_errors_propagate(self):
        first, second = make_linear(7, 16), make_linear(3, 16)
        class DerivedLinear(PPUPTPCLinear):
            pass
        derived = DerivedLinear(first.weight, first.scale)
        with patch.object(torch, "cat", side_effect=AssertionError("must decline before packing")):
            self.assertIsNone(PreparedQKV.try_prepare(first, second))  # CPU
            for left, right in ((None, second), (first, object()), (derived, second)):
                self.assertIsNone(PreparedQKV.try_prepare(left, right))
            with patch.object(qkv_module, "_physical_pair_supported", return_value=True):
                self.assertIsNone(PreparedQKV.try_prepare(first, make_linear(3, 8)))
        _, _, helper = self.prepare()
        error = RuntimeError("native GEMM failed")
        with torch.no_grad(), patch.object(
            helper._linear, "forward_quantized", side_effect=error
        ):
            with self.assertRaisesRegex(RuntimeError, "native GEMM failed"):
                helper.project(torch.ones(4, 16).bfloat16())

    @torch.inference_mode()
    def test_inference_weight_owner_contract_does_not_read_missing_version(self):
        first, second, helper = self.prepare()
        self.assertTrue(torch.is_inference(first.weight))
        self.assertIs(helper._sources[0], first)
        self.assertIs(helper._sources[1], second)
        self.assertIsNotNone(helper.project(torch.ones(1, 16).bfloat16()))


class StridedPTPCNormCPU(unittest.TestCase):
    @torch.no_grad()
    def test_explicit_layout_admission_and_default_rejection(self):
        with patch.object(torch, "__version__", "2.11.0"), patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(norm_module.importlib.util, "find_spec", return_value=object()), patch.dict(
            os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}
        ):
            for width, offset in ((1280, 0), (512, 1280)):
                value = torch.ones((5, 1792), dtype=torch.bfloat16)[:, offset:offset + width]
                gamma = torch.ones(width, dtype=torch.bfloat16)
                self.assertFalse(norm_module.can_fuse_ptpc_rms_norm(value, gamma))
                self.assertFalse(norm_quant_module.is_supported(value, gamma, 1e-6))
                self.assertTrue(norm_module.can_fuse_ptpc_rms_norm(value, gamma, allow_strided=True))
                self.assertTrue(norm_quant_module.is_supported(value, gamma, 1e-6, allow_strided=True))
                invalid = (
                    value[:, ::2], torch._neg_view(value),
                    torch.as_strided(value, (5, width), (width - 1, 1)),
                    value[None],
                )
                for item in invalid:
                    self.assertFalse(norm_quant_module.is_supported(item, gamma, 1e-6, allow_strided=True))
                large = torch.ones((5, 5121), dtype=torch.bfloat16)[:, :5120]
                self.assertFalse(norm_quant_module.is_supported(
                    large, torch.ones(5120).bfloat16(), 1e-6, allow_strided=True
                ))

    @torch.no_grad()
    def test_candidate_passes_runtime_row_stride_and_allocates_dense_outputs(self):
        calls = []

        class Kernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append((grid, args, kwargs))
                return launch

        class Triton:
            @staticmethod
            def next_power_of_2(value):
                return 1 << (value - 1).bit_length()

        for rows in (0, 1, 140):
            for width, offset in ((1280, 0), (512, 1280)):
                value = torch.ones((rows + 2, 1792), dtype=torch.bfloat16)[1:rows + 1, offset:offset + width]
                gamma = torch.ones(width, dtype=torch.bfloat16)
                with patch.object(norm_quant_module, "is_supported", return_value=True), patch.object(
                    norm_quant_module, "_kernels", return_value=(Triton(), Kernel())
                ):
                    before = len(calls)
                    output = norm_quant_module.ptpc_rms_norm_quant_candidate(
                        value, gamma, allow_strided=True
                    )
                for item in output:
                    self.assertTrue(item.is_contiguous())
                    self.assertFalse(torch._C._is_alias_of(item, value))
                self.assertEqual(output[0].shape, (rows, width))
                self.assertEqual(output[1].dtype, torch.float8_e4m3fn)
                self.assertEqual(output[2].shape, (rows, 1))
                self.assertEqual(len(calls) - before, int(rows > 0))
                if rows:
                    _, args, kwargs = calls[-1]
                    self.assertIs(args[0], value)
                    # A one-row view is itself contiguous; both row strides
                    # address exactly the same sole logical row.
                    self.assertEqual(args[5], width if value.is_contiguous() else 1792)
                    self.assertEqual(kwargs["WIDTH"], width)

    @torch.no_grad()
    def test_declined_strided_candidate_never_uses_dense_only_verify_provider(self):
        value = torch.randn(4, 1792).bfloat16()[:, 1280:]
        gamma = torch.randn(512).bfloat16()
        expected = object()
        with patch.object(norm_module, "rms_norm", return_value=expected) as shared, patch.object(
            norm_quant_module, "ptpc_rms_norm_candidate",
            side_effect=AssertionError("ordinary norm must use shared dispatch"),
        ):
            actual = norm_module.ptpc_rms_norm(value, gamma, 1e-6, allow_strided=True)
        self.assertIs(actual, expected)
        shared.assert_called_once_with(value, gamma, 1e-6)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test"
)
class LocalPrefillDenseProjectionDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requested PPU runtime is absent")
        self.flags = patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"})
        self.flags.start()
        self.addCleanup(self.flags.stop)
        native_only = patch(
            "rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant.quantize_ptpc_activation_reference",
            side_effect=AssertionError("production PTPC must use the native quantizer"),
        )
        native_only.start()
        self.addCleanup(native_only.stop)

    def prepare(self):
        first, second = make_linear(1280, 5120, "cuda"), make_linear(512, 5120, "cuda")
        # Include actual per-channel FP32 scales rather than an all-one GEMM.
        first.scale.copy_(torch.rand_like(first.scale) * 0.01 + 0.001)
        second.scale.copy_(torch.rand_like(second.scale) * 0.01 + 0.001)
        self.assertIs(first._gemm, second._gemm)
        # Bind the same transparent provider spy into both original modules
        # and the packed module. Production guards remain active throughout.
        first._gemm = second._gemm = Mock(wraps=first._gemm)
        helper = PreparedQKV.try_prepare(first, second)
        self.assertIsNotNone(helper)
        return first, second, helper

    def check_outputs(self, first, second, helper, value, *, label):
        pair = first.quantize_input(value)
        expected = tuple(op.forward_quantized(*pair) for op in (first, second))
        gemm = helper._linear._gemm
        gemm.reset_mock()
        actual = helper.project(value, quantized_input=pair)
        self.assertIsNotNone(actual)
        self.assertEqual(gemm.call_count, int(value.shape[0] > 0))
        self.assertTrue(torch._C._is_alias_of(*actual))
        for index, (raw, old_raw) in enumerate(zip(actual, expected)):
            assert_bits(self, raw, old_raw, f"{label}/raw-{index}")
            gamma = torch.randn(raw.shape[-1], dtype=torch.bfloat16, device="cuda")
            old_norm = norm_module.ptpc_rms_norm(old_raw, gamma, 1e-6)
            new_norm = norm_module.ptpc_rms_norm(raw, gamma, 1e-6, allow_strided=True)
            self.assertTrue(new_norm.is_contiguous())
            assert_bits(self, new_norm, old_norm, f"{label}/norm-{index}")
            produced = norm_quant_module.ptpc_rms_norm_quant_candidate(
                raw, gamma, allow_strided=True
            )
            self.assertIsNotNone(produced, "strided native producer declined")
            # Layout-only extension must match the existing contiguous fused
            # provider, including its established nonfinite payload encoding.
            original = norm_quant_module.ptpc_rms_norm_quant_candidate(old_raw, gamma)
            self.assertIsNotNone(original)
            portable = reference_quant(old_norm)
            differs = [left.numel() > 0 and not torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
                       for left, right in zip(original[1:], portable)]
            if any(differs) and os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
                root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
                evidence = Path(tempfile.mkdtemp(prefix="existing-quant-provider-difference-", dir=root))
                torch.save({"label": label, "old_raw": old_raw.cpu(), "gamma": gamma.cpu(),
                            "contiguous_fused": tuple(value.cpu() for value in original),
                            "portable_quant": tuple(value.cpu() for value in portable),
                            "differs": differs}, evidence / "operands.pt")
            for name, left, right in zip(
                ("norm", "payload", "scale"), produced, original
            ):
                self.assertTrue(left.is_contiguous())
                assert_bits(self, left, right, f"{label}/{name}-{index}")
            # Production uses norm-only plus the unchanged native quantizer.
            for name, left, right in zip(
                ("payload", "scale"), quantize_ptpc_activation(new_norm),
                quantize_ptpc_activation(old_norm),
            ):
                assert_bits(self, left, right, f"{label}/production-{name}-{index}")
        return actual

    @torch.inference_mode()
    def test_shared_fp8_native_packed_projection_and_strided_norm_bytes(self):
        torch.manual_seed(4151)
        first, second, helper = self.prepare()
        for rows in (0, 1, 140, 4096):
            with self.subTest(rows=rows):
                value = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
                self.check_outputs(first, second, helper, value, label=f"normal-M{rows}")

    @torch.inference_mode()
    def test_extreme_and_nonfinite_values_preserve_existing_norm_and_scales(self):
        torch.manual_seed(4152)
        first, second, helper = self.prepare()
        raw_bits = torch.tensor(
            [0, -32768, 1, -32767, 127, -32641, 128, -32640,
             16255, 16256, 16257, -16513, -16512, -16511],
            dtype=torch.int16,
        )
        finite = raw_bits.repeat((5120 + raw_bits.numel() - 1) // raw_bits.numel())[:5120]
        finite = finite.view(torch.bfloat16).cuda()[None].repeat(140, 1)
        self.check_outputs(first, second, helper, finite, label="BF16-boundaries")
        for magnitude in (0.0, 2.0**-70, 2.0**30, torch.finfo(torch.bfloat16).max):
            value = torch.full((1, 5120), magnitude, dtype=torch.bfloat16, device="cuda")
            value[:, 1::2].neg_()
            self.check_outputs(first, second, helper, value, label=f"magnitude-{magnitude}")
        for special in (float("inf"), -float("inf"), float("nan")):
            value = torch.zeros((1, 5120), dtype=torch.bfloat16, device="cuda")
            value[:, 0] = special
            self.check_outputs(first, second, helper, value, label=f"nonfinite-{special}")

    @torch.inference_mode()
    def test_strided_norm_value_bits_match_dense_provider_without_input_copy(self):
        patterns = (
            [0, -32768, 1, -32767, 128, -32640, 16255, 16256, 16257],
            [0, 32639, -129, 16256, -16512],
            [0, 32640, -128, 32704],
        )
        for rows in (0, 1, 140, 4096):
            for pattern_index, pattern in enumerate(patterns):
                bits = torch.tensor(pattern, dtype=torch.int16)
                source = bits.repeat((1792 + bits.numel() - 1) // bits.numel())[:1792]
                source = source.view(torch.bfloat16).cuda()
                # Nonzero storage offset and sibling-channel canaries exercise
                # the exact same layouts as the packed GEMM output.
                slab = source[None].repeat(rows + 2, 1)
                before = slab.clone()
                for width, offset in ((1280, 0), (512, 1280)):
                    value = slab[1:rows + 1, offset:offset + width]
                    dense = value.contiguous()
                    gamma = torch.ones(width, dtype=torch.bfloat16, device="cuda")
                    original = norm_quant_module.ptpc_rms_norm_quant_candidate(dense, gamma)
                    actual = norm_quant_module.ptpc_rms_norm_quant_candidate(
                        value, gamma, allow_strided=True
                    )
                    self.assertIsNotNone(original)
                    self.assertIsNotNone(actual)
                    for name, left, right in zip(("norm", "payload", "scale"), actual, original):
                        assert_bits(
                            self, left, right,
                            f"strided-{name}-M{rows}-W{width}-pattern{pattern_index}",
                        )
                        self.assertTrue(left.is_contiguous())
                        self.assertFalse(torch._C._is_alias_of(left, value))
                assert_bits(self, slab, before, "norm source and canaries unchanged")

    @torch.inference_mode()
    def test_changed_input_gamma_graph_matches_independent_projections(self):
        torch.manual_seed(4153)
        first, second, helper = self.prepare()
        value = torch.randn(140, 5120, dtype=torch.bfloat16, device="cuda")
        gamma_q = torch.randn(1280, dtype=torch.bfloat16, device="cuda")
        gamma_kv = torch.randn(512, dtype=torch.bfloat16, device="cuda")

        def body():
            raw = helper.project(value)
            self.assertIsNotNone(raw)
            normalized = []
            for part, gamma in zip(raw, (gamma_q, gamma_kv)):
                norm = norm_module.ptpc_rms_norm(part, gamma, 1e-6, allow_strided=True)
                normalized.append((norm, *quantize_ptpc_activation(norm)))
            return raw, normalized

        body()  # Compile/allocate before capture.
        first(value)
        second(value)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_raw, actual_norm = body()
        for magnitude in (0.0, 0.125, 2.0):
            value.copy_(torch.randn_like(value) * magnitude)
            gamma_q.copy_(torch.randn_like(gamma_q))
            gamma_kv.copy_(torch.randn_like(gamma_kv))
            graph.replay()
            pair = first.quantize_input(value)
            for index, (op, gamma) in enumerate(zip((first, second), (gamma_q, gamma_kv))):
                expected_raw = op.forward_quantized(*pair)
                assert_bits(self, actual_raw[index], expected_raw, f"Graph raw-{index}")
                expected_norm = norm_module.ptpc_rms_norm(expected_raw, gamma, 1e-6)
                for name, left, right in zip(
                    ("norm", "payload", "scale"), actual_norm[index],
                    (expected_norm, *quantize_ptpc_activation(expected_norm)),
                ):
                    assert_bits(self, left, right, f"Graph {name}-{index}")


if __name__ == "__main__":
    unittest.main()
