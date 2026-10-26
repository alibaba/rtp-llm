"""Native DeepGEMM grouped WO_A: quantization, ownership and graph contracts."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _inverse_rope_bf16(o, freqs):
    """Independent eager complex rotation, with the required BF16 boundary."""
    result = o.clone()
    rows = o.numel() // (o.shape[-2] * 512)
    flat = result.reshape(rows, o.shape[-2], 512)
    expanded = freqs.repeat_interleave(rows // freqs.shape[0], dim=0)
    complex_tail = torch.view_as_complex(
        flat[..., -64:].float().reshape(rows, -1, 32, 2)
    )
    rotated = torch.view_as_real(complex_tail * expanded.conj()[:, None]).flatten(-2)
    flat[..., -64:].copy_(rotated.to(torch.bfloat16))
    return flat


def _independent_quantize(grouped, *, backend):
    rows, groups, width = grouped.shape
    use_v2 = backend == "v2" or (backend == "auto" and rows * width >= 4 * 1024 * 1024)
    blocks = grouped.float().reshape(rows, groups, width // 32, 32)
    maximum = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    raw = maximum * (1.0 / 448.0)
    if use_v2:
        raw = torch.where(raw < torch.finfo(torch.float32).tiny, 0.0, raw)
    else:
        raw = raw.clamp_min(1e-10)
    exponent = torch.where(raw > 0, torch.ceil(torch.log2(raw)) + 127, 0).to(
        torch.int64
    )
    scale = torch.exp2(exponent.float() - 127)
    payload = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    shifts = torch.arange(4, device=grouped.device, dtype=torch.int64) * 8
    packed = (
        (exponent.reshape(rows, groups, width // 128, 4) << shifts)
        .sum(-1)
        .to(torch.int32)
    )
    return payload.reshape(rows, groups, width), packed


def _fp32_oracle(q, packed, weight, block_scales):
    """Independent exponent extraction, 32x32 weight expansion, FP32 GEMM."""
    rows, groups, width = q.shape
    shifts = torch.arange(4, device=q.device, dtype=torch.int64) * 8
    exponents = ((packed.to(torch.int64)[..., None] >> shifts) & 255).reshape(
        rows, groups, width // 32
    )
    scales = torch.exp2(exponents.float() - 127)
    inputs = (
        q.float().reshape(rows, groups, width // 32, 32) * scales[..., None]
    ).flatten(-2)
    outputs = []
    for group in range(groups):
        weight_scales = (
            block_scales[group]
            .float()
            .repeat_interleave(32, 0)
            .repeat_interleave(32, 1)
        )
        decoded = weight[group].float() * weight_scales[: weight.shape[1], :width]
        outputs.append((inputs[:, group] @ decoded.T).to(torch.bfloat16))
    return torch.stack(outputs, dim=1).flatten(1)


def _metrics(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    return dict(
        finite=bool(finite.all()),
        max_abs=float(delta.max()),
        rms_abs=float(delta.square().mean().sqrt()),
        violations=int(
            ((~finite) | (delta > 0.02 + 0.015 * expected.float().abs())).sum()
        ),
    )


def _fp64_oracle(q, packed, weight, block_scales, *, device):
    """Diagnostic higher-precision reduction; never an acceptance bypass."""
    rows, groups, width = q.shape
    payload = q.to(device=device, dtype=torch.float64)
    words = packed.to(device=device, dtype=torch.int64)
    shifts = torch.arange(4, device=device, dtype=torch.int64) * 8
    exponent = ((words[..., None] >> shifts) & 255).reshape(rows, groups, width // 32)
    scales = torch.exp2(exponent.to(torch.float64) - 127)
    inputs = (
        payload.reshape(rows, groups, width // 32, 32) * scales[..., None]
    ).flatten(-2)
    outputs = []
    for group in range(groups):
        values = weight[group].to(device=device, dtype=torch.float64)
        block = block_scales[group].to(device=device, dtype=torch.float64)
        scale = block.repeat_interleave(32, 0).repeat_interleave(32, 1)
        decoded = values * scale[: weight.shape[1], :width]
        outputs.append(inputs[:, group] @ decoded.T)
    return torch.stack(outputs, dim=1).flatten(1)


def _byte_differences(actual, expected):
    left = actual.contiguous().view(torch.uint8)
    right = expected.contiguous().view(torch.uint8)
    return int((left != right).sum())


def _input(rows, groups=8, width=4096, *, strided=False, shared_freqs=False):
    heads = groups * (width // 512)
    if strided:
        storage = torch.randn(rows, heads + 2, 544, device="cuda", dtype=torch.bfloat16)
        o = storage[:, :heads, :512]
    else:
        o = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
    freq_rows = rows // 2 if shared_freqs else rows
    angle = torch.randn(freq_rows, 64 if strided else 32, device="cuda")
    frequencies = torch.polar(torch.ones_like(angle), angle)
    if strided:
        frequencies = frequencies[:, ::2]
    return o, frequencies


def _capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output



class SM90OutputProjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 GPU required")

    def setUp(self):
        torch.manual_seed(419331)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def fixture(self, groups=8, n=256, k=512, packed=False):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import Sm90GroupedOutputProjection
        # Scaled random data keeps the strict independent FP32 oracle separate
        # from the known amplified-input FP8 WGMMA stress limitation.
        weight = (torch.randn(groups, n, k, device="cuda") * .125).to(torch.float8_e4m3fn)
        scales = torch.randint(125, 128, (groups, n // 32, k // 32), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        source = scales
        if packed:
            source = scales.view(torch.uint8).repeat_interleave(32, dim=1).contiguous().view(torch.int32)
        module = Sm90GroupedOutputProjection(weight, source)
        self.assertEqual(module.backend, "deepgemm")
        self.assertEqual(module.weight.data_ptr(), weight.data_ptr())
        self.assertEqual(module.weight.numel(), weight.numel())
        self.assertEqual(module.weight_scale.numel(), scales.numel())
        return module, weight, scales

    def test_groups_and_actual_model_shape_against_fp32(self):
        for groups, n, k in ((1, 256, 512), (2, 256, 512), (8, 1024, 4096)):
            module, weight, scales = self.fixture(groups, n, k)
            weight_before = weight.view(torch.uint8).clone()
            scales_before = scales.view(torch.uint8).clone()
            for rows in (1, 7, 17, 128):
                o, freqs = _input(rows, groups, k)
                o.mul_(.125)
                q, packed = module.quantize(o, freqs)
                actual = module.forward_quantized(q, packed)
                expected = _fp32_oracle(q, packed, weight, scales)
                self.assertEqual(_metrics(actual, expected)["violations"], 0, (groups, rows, _metrics(actual, expected)))
                self.assertTrue(torch.equal(actual, module(o, freqs)), (groups, rows))
            self.assertTrue(torch.equal(weight.view(torch.uint8), weight_before))
            self.assertTrue(torch.equal(scales.view(torch.uint8), scales_before))

    def test_graph_observes_changed_inputs_and_scales(self):
        module, _, _ = self.fixture()
        for rows in (1, 16, 17, 128):
            o, freqs = _input(rows, 8, 512)
            module(o, freqs)
            graph, captured = _capture(lambda: module(o, freqs))
            for factor in (.03125, 1., 8.):
                o.copy_(torch.randn_like(o) * factor)
                freqs.copy_(torch.polar(torch.ones_like(freqs.real), torch.randn_like(freqs.real)))
                expected = module(o, freqs)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(captured, expected), (rows, factor))

    def test_strided_inputs_shared_frequencies_and_output(self):
        module, _, _ = self.fixture()
        o, freqs = _input(18, 8, 512, strided=True, shared_freqs=True)
        out = torch.empty((18, 8 * 256), device="cuda", dtype=torch.bfloat16)
        self.assertIs(module(o, freqs, out=out), out)
        self.assertTrue(torch.equal(out, module(o.contiguous(), freqs.contiguous())))
        q, packed = module.quantize(o, freqs)
        self.assertTrue(torch.equal(out, module.forward_quantized(q.contiguous(), packed.contiguous())))

    def test_packed_weight_scale_abi_and_rejection(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import Sm90GroupedOutputProjection
        module, weight, scales = self.fixture(packed=True)
        raw = Sm90GroupedOutputProjection(weight, scales)
        o, f = _input(17, 8, 512)
        self.assertTrue(torch.equal(module(o, f), raw(o, f)))
        expanded = scales.view(torch.uint8).repeat_interleave(32, dim=1).contiguous()
        expanded[0, 1, 0] = 120
        with self.assertRaisesRegex(ValueError, "equal scales"):
            Sm90GroupedOutputProjection(weight, expanded.view(torch.int32))

    def test_empty_and_output_alias_contract(self):
        module, _, _ = self.fixture(1)
        o = torch.empty((0, 1, 512), device="cuda", dtype=torch.bfloat16)
        freqs = torch.empty((0, 32), device="cuda", dtype=torch.complex64)
        self.assertEqual(tuple(module(o, freqs).shape), (0, 256))
        o, freqs = _input(1, 1, 512)
        with self.assertRaisesRegex(ValueError, "alias"):
            module(o, freqs, out=o.view(-1)[:256].view(1, 256))

    def test_batched_matches_per_group_bitwise(self):
        """One batched GEMM must reproduce the per-group loop exactly."""
        for groups, n, k in ((1, 256, 512), (2, 256, 512), (8, 1024, 4096)):
            module, _, _ = self.fixture(groups, n, k)
            self.assertIsNotNone(module._einsum)
            for rows in (1, 7, 17, 128):
                o, freqs = _input(rows, groups, k)
                o.mul_(.125)
                q, packed = module.quantize(o, freqs)
                batched = module.forward_quantized(q, packed)
                # Same operands and recipe, only the launch count differs.
                module._einsum = None
                try:
                    per_group = module.forward_quantized(q, packed)
                finally:
                    module._einsum = getattr(
                        __import__("deep_gemm"), "fp8_einsum", None
                    )
                self.assertTrue(
                    torch.equal(batched, per_group), (groups, rows, n, k)
                )

    def test_missing_einsum_falls_back_with_warning(self):
        """An older wheel must warn once and keep the per-group path working."""
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import Sm90GroupedOutputProjection
        weight = (torch.randn(8, 256, 512, device="cuda") * .125).to(torch.float8_e4m3fn)
        scales = torch.randint(125, 128, (8, 8, 16), device="cuda", dtype=torch.uint8).view(torch.float8_e8m0fnu)
        with patch("deep_gemm.fp8_einsum", create=True, new=None), self.assertLogs(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection",
            level="WARNING",
        ) as logs:
            module = Sm90GroupedOutputProjection(weight, scales)
        self.assertIsNone(module._einsum)
        self.assertIn("fp8_einsum is unavailable", logs.output[0])
        o, freqs = _input(17, 8, 512)
        o.mul_(.125)
        self.assertEqual(tuple(module(o, freqs).shape), (17, 8 * 256))


if __name__ == "__main__":
    unittest.main()
