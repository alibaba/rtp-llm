"""SM90 grouped WO_A: original G32 A8 contract before clean graph timing.

This target requires a parent-owned SM90 GPU. Each timing input must pass its
independent FP32 gate first; no BF16/A16 backend is an eligible replacement.
Graph equivalence is strictly bitwise versus eager and the original per-group
Humming path. Amplified-input FP32 stress acceptance is a separate opt-in gate:
its existing .015/.02 failures are recorded, never promoted to numerical PASS
merely because the replacement preserves the original A8 arithmetic exactly.
"""

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
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            9,
            0,
        ):
            raise RuntimeError(
                "This dedicated target requires a parent-locked SM90 GPU"
            )
        cls.fixtures = {}

    def setUp(self):
        env = patch.dict(
            os.environ, {"DSV41_SM90_NATIVE_A8": "1", "DSV4_FP8_QUANT_KERNEL": "auto"}
        )
        env.start()
        self.addCleanup(env.stop)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.manual_seed(419331)

    def _fixture(self, groups=8, n=1024, k=4096):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import (
            Sm90GroupedOutputProjection,
        )
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        key = (groups, n, k)
        if key not in self.fixtures:
            generator = torch.Generator(device="cuda").manual_seed(
                419331 + groups + n + k
            )
            weight = torch.randn(groups, n, k, device="cuda", generator=generator).to(
                torch.float8_e4m3fn
            )
            exponent = torch.randint(
                124, 130, (groups, n // 32, k // 32), device="cuda", generator=generator
            )
            scales = exponent.to(torch.uint8).view(torch.float8_e8m0fnu)
            candidate = Sm90GroupedOutputProjection(weight, scales)
            baseline = [
                V41MXFP8Linear(weight[g].contiguous(), scales[g].contiguous())
                for g in range(groups)
            ]
            for adapter in (candidate, *(linear._sm90_linear for linear in baseline)):
                config = json.loads(adapter.config.to_str())
                self.assertEqual(config["mma_type"], "wgmma")
                self.assertEqual(config["a_dtype"], "float8e4m3")
                self.assertEqual(config["b_dtype"], "float8e4m3")
                self.assertEqual(config["input_scale_group_size"], 32)
                self.assertEqual(config["weight_scale_group_size"], 32)
                self.assertFalse(config["use_fused_e8m0_scale"])
                self.assertFalse(config["use_int_weight_scale"])
                compute = json.loads(adapter.compute_config)
                self.assertTrue(compute["use_batch_invariant"])
                self.assertFalse(compute["use_f16_accum"])
            self.assertEqual(
                json.loads(candidate.compute_config)["gemm_type"],
                "dense" if groups == 1 else "grouped_contiguous",
            )
            self.assertEqual(
                json.loads(candidate.config.to_str())["num_experts"],
                0 if groups == 1 else groups,
            )
            self.fixtures[key] = (weight, scales, candidate, baseline)
        return self.fixtures[key]

    def _legacy(self, o, freqs, linears, *, clone=True):
        from rtp_llm.models_py.modules.dsv41.rope import apply_rotary_emb

        source = o.clone() if clone else o
        rows, heads = source.shape[:2]
        expanded = (
            freqs
            if rows == freqs.shape[0]
            else freqs.repeat_interleave(rows // freqs.shape[0], dim=0)
        )
        apply_rotary_emb(source[..., -64:].unsqueeze(0), expanded, inverse=True)
        grouped = source.reshape(rows, len(linears), -1)
        return torch.stack(
            [
                linear(grouped[:, group].contiguous())
                for group, linear in enumerate(linears)
            ],
            dim=1,
        ).flatten(1)

    def _check_pair(self, o, freqs, fixture):
        weight, scales, candidate, baseline = fixture
        groups, _, width = weight.shape
        rotated = _inverse_rope_bf16(o, freqs).reshape(o.shape[0], groups, width)
        expected_q, expected_s = _independent_quantize(
            rotated, backend=os.environ["DSV4_FP8_QUANT_KERNEL"]
        )
        q, packed = candidate.quantize(o, freqs)
        self.assertEqual(q.shape, expected_q.shape)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(packed.shape, expected_s.shape)
        self.assertEqual(packed.dtype, torch.int32)
        torch.testing.assert_close(
            q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(packed, expected_s, rtol=0, atol=0)
        for group, linear in enumerate(baseline):
            old_q, old_s = linear._quantize_input(rotated[:, group].contiguous())
            torch.testing.assert_close(
                q[:, group].view(torch.uint8), old_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(packed[:, group], old_s, rtol=0, atol=0)
        return q, packed

    def _check_outputs(self, o, freqs, fixture):
        weight, scales, candidate, baseline = fixture
        q, packed = self._check_pair(o, freqs, fixture)
        expected = _fp32_oracle(q, packed, weight, scales)
        actual = candidate.forward(o, freqs)
        old = self._legacy(o, freqs, baseline)
        metrics = dict(
            candidate=_metrics(actual, expected), baseline=_metrics(old, expected)
        )
        print(
            "SM90_OUTPUT_PROJECTION_CORRECTNESS "
            + json.dumps(
                dict(
                    M=o.shape[0],
                    G=weight.shape[0],
                    N=weight.shape[1],
                    K=weight.shape[2],
                    tolerance=dict(rtol=0.015, atol=0.02),
                    **metrics,
                )
            ),
            flush=True,
        )
        torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.02)
        torch.testing.assert_close(old, expected, rtol=0.015, atol=0.02)
        torch.testing.assert_close(
            actual.view(torch.int16), old.view(torch.int16), rtol=0, atol=0
        )
        return actual, expected

    @torch.no_grad()
    def test_decode_m1_minimal_exact_geometry(self):
        o, freqs = _input(1)
        self._check_outputs(o, freqs, self._fixture())

    @torch.no_grad()
    def test_real_attention_project_output_uses_grouped_adapter(self):
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8

        fixture = self._fixture()
        candidate = fixture[2]
        o, freqs = _input(2)
        saved_o, saved_freqs = o.clone(), freqs.clone()
        expected, _ = self._check_outputs(o, freqs, fixture)
        calls = []

        # Only downstream WO_B is replaced: WO_A runs the real adapter and
        # the actual Attention method. This verifies dispatch and output
        # ownership, not WO_B GEMM numerics or full-model construction.
        def wo_b(projected, *, out=None):
            calls.append((projected.clone(), out))
            if out is None:
                return projected
            out.copy_(projected)
            return out

        owner = SimpleNamespace(
            n_heads=64,
            head_dim=512,
            _sm90_wo_a=candidate,
            wo_b=wo_b,
        )
        for supplied_out in (False, True):
            with self.subTest(supplied_out=supplied_out):
                out = torch.empty_like(expected) if supplied_out else None
                returned = AttentionV41FP8._project_output(
                    owner,
                    o.reshape(1, 2, 64, 512),
                    freqs,
                    out=out,
                )
                projected, observed_out = calls[-1]
                self.assertIs(observed_out, out)
                if supplied_out:
                    self.assertIs(returned, out)
                torch.testing.assert_close(
                    projected.view(torch.int16),
                    expected.view(torch.int16),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    returned.view(torch.int16),
                    expected.view(torch.int16),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    o.view(torch.int16), saved_o.view(torch.int16), rtol=0, atol=0
                )
                torch.testing.assert_close(freqs, saved_freqs, rtol=0, atol=0)
        self.assertEqual(len(calls), 2)

    @torch.no_grad()
    def test_quantization_payload_scales_rope_rounding_and_tail_rows(self):
        fixture = self._fixture()
        for backend in ("auto", "legacy", "v2"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": backend}):
                for rows in (1, 2, 3, 4, 7, 8, 15, 16, 17, 128, 1024):
                    with self.subTest(backend=backend, M=rows):
                        o, freqs = _input(rows, strided=rows in (3, 17))
                        # Include all-zero G32 groups, tiny inputs and signed zero.
                        o[:, 0, :32] = 0
                        o[:, 0, 32:64] = -0.0
                        o[:, 0, 64:96] *= 1e-30
                        self._check_pair(o, freqs, fixture)

    @torch.no_grad()
    def test_real_geometry_correctness_and_output_buffer(self):
        fixture = self._fixture()
        candidate = fixture[2]
        for rows in (1, 2, 4, 8, 16, 128, 1024):
            with self.subTest(M=rows):
                o, freqs = _input(rows)
                saved_o, saved_freqs = o.clone(), freqs.clone()
                actual, expected = self._check_outputs(o, freqs, fixture)
                out = torch.empty_like(actual)
                returned = candidate.forward(o, freqs, out=out)
                self.assertIs(returned, out)
                torch.testing.assert_close(out, expected, rtol=0.015, atol=0.02)
                torch.testing.assert_close(o, saved_o, rtol=0, atol=0)
                torch.testing.assert_close(freqs, saved_freqs, rtol=0, atol=0)

    @torch.no_grad()
    def test_multiple_groups_strides_tails_and_packed_weight_scales(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_output_projection import (
            Sm90GroupedOutputProjection,
        )

        for groups, n, k, rows in (
            (1, 256, 512, 3),
            (2, 256, 1024, 7),
            (8, 256, 512, 17),
        ):
            with self.subTest(G=groups, N=n, K=k, M=rows):
                fixture = self._fixture(groups, n, k)
                weight, scales, _, baseline = fixture
                o, freqs = _input(rows, groups, k, strided=True)
                self._check_outputs(o, freqs, fixture)
                expanded = (
                    scales.view(torch.uint8)
                    .repeat_interleave(32, dim=1)
                    .to(torch.int64)
                )
                shifts = torch.arange(4, device=o.device, dtype=torch.int64) * 8
                words = (
                    (expanded.reshape(groups, n, k // 128, 4) << shifts)
                    .sum(-1)
                    .to(torch.int32)
                )
                packed_candidate = Sm90GroupedOutputProjection(weight, words)
                self._check_outputs(
                    o, freqs, (weight, scales, packed_candidate, baseline)
                )

        fixture = self._fixture()
        for rows in (4, 16):
            with self.subTest(shared_freq_rows=True, M=rows):
                o, freqs = _input(rows, strided=True, shared_freqs=True)
                self._check_outputs(o, freqs, fixture)

    @torch.no_grad()
    def test_decode_gate_and_rejected_output_contracts(self):
        candidate = self._fixture()[2]
        for rows in (1, 16, 17, 128):
            o, freqs = _input(rows)
            self.assertEqual(candidate.supports(o, freqs), rows <= 16)
        o, freqs = _input(1)
        self.assertFalse(candidate.supports(o.float(), freqs))
        self.assertFalse(candidate.supports(o, freqs.conj()))
        q, packed = candidate.quantize(o, freqs)
        with self.assertRaises((ValueError, TypeError)):
            candidate.forward_quantized(
                q, packed, out=torch.empty(1, 8192, device="cuda")
            )
        with self.assertRaises((ValueError, TypeError)):
            aliased = q.permute(1, 0, 2).view(torch.bfloat16).reshape(1, -1)[:, :8192]
            candidate.forward_quantized(q, packed, out=aliased)
        empty = torch.empty((0, 64, 512), device="cuda", dtype=torch.bfloat16)
        empty_freqs = torch.empty((0, 32), device="cuda", dtype=torch.complex64)
        output = candidate.forward(empty, empty_freqs)
        self.assertEqual(output.shape, (0, 8192))

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_OUTPUT_PROJECTION_SASS") == "1",
        "explicit registered-kernel binary audit opt-in required",
    )
    @torch.no_grad()
    def test_registered_grouped_kernels_use_native_fp8_sass(self):
        import hashlib
        import re
        import shutil
        import subprocess
        from pathlib import Path

        from humming.kernel.humming import HummingKernel

        destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        self.assertTrue(
            destination, "SASS audit requires persistent Bazel undeclared outputs"
        )
        directory = Path(destination) / "wo_a_grouped_native_a8"
        directory.mkdir(parents=True, exist_ok=True)
        table = []
        for groups in (1, 8):
            candidate = self._fixture(groups)[2]
            entries = (
                HummingKernel.prepare_kernels(
                    candidate.config.to_str(),
                    candidate.compute_config,
                    None,
                    candidate.weight.device,
                )
                .reshape(-1, 4)
                .tolist()
            )
            table.extend((groups, *entry) for entry in entries)
        records = []
        cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        # Audit every registered interval, not an assumed row-count selector
        # for grouped or single-group dense GEMM. Any launcher interval must
        # satisfy the same native E4M3 instruction contract.
        for groups, low, high, kernel_id, num_sms in table:
            kernel = HummingKernel._id2kernel[kernel_id]
            binary = Path(kernel.kernel_filename)
            target = directory / f"g{groups}_interval_{low}_{high}_{kernel_id}.cubin"
            shutil.copy2(binary, target)
            result = subprocess.run(
                [cuobjdump, "--dump-sass", str(target)],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            target.with_suffix(".sass").write_text(result.stdout)
            instructions = re.findall(r"\bQGMMA\.[^;\n]+", result.stdout)
            self.assertTrue(instructions)
            self.assertTrue(all(".F32.E4M3.E4M3 " in value for value in instructions))
            self.assertNotRegex(result.stdout, r"\b(?:HGMMA|HMMA)\.")
            records.append(
                dict(
                    G=groups,
                    interval=[low, high],
                    kernel_id=kernel_id,
                    num_sms=num_sms,
                    name=kernel.kernel_name,
                    cubin=str(target),
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    instructions=list(dict.fromkeys(instructions)),
                )
            )
        self.assertTrue(records)
        (directory / "registered_kernel_audit.json").write_text(
            json.dumps(records, indent=2)
        )

    @torch.no_grad()
    def test_graph_replays_changed_inputs_frequencies_and_quant_scales(self):
        self._check_graph_replay_contract(require_fp32_acceptance=False)

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_OUTPUT_PROJECTION_STRESS") == "1",
        "separate amplified-input FP32 numerical acceptance; known baseline A8 failures remain failures",
    )
    @torch.no_grad()
    def test_amplified_input_fp32_stress_acceptance(self):
        self._check_graph_replay_contract(require_fp32_acceptance=True)

    def _check_graph_replay_contract(self, *, require_fp32_acceptance):
        fixture = self._fixture()
        weight, scales, candidate, baseline = fixture
        for rows in (1, 7, 16):
            with self.subTest(M=rows):
                o, freqs = _input(rows, strided=rows == 7)
                self._check_outputs(o, freqs, fixture)
                initial_q, initial_s = self._check_pair(o, freqs, fixture)
                initial_q, initial_s = initial_q.clone(), initial_s.clone()
                graph, captured = _capture(lambda: candidate.forward(o, freqs))
                for multiplier in (16.0, 0.03125):
                    o.copy_(torch.randn_like(o) * multiplier)
                    angles = torch.randn(freqs.shape, device=freqs.device)
                    freqs.copy_(torch.polar(torch.ones_like(angles), angles))
                    q, packed = self._check_pair(o, freqs, fixture)
                    self.assertTrue(
                        bool((q.view(torch.uint8) != initial_q.view(torch.uint8)).any())
                    )
                    self.assertTrue(bool((packed != initial_s).any()))
                    expected = _fp32_oracle(q, packed, weight, scales)
                    graph.replay()
                    eager = candidate.forward(o, freqs)
                    original = self._legacy(o, freqs, baseline)
                    observations = dict(
                        graph=_metrics(captured, expected),
                        eager=_metrics(eager, expected),
                        original=_metrics(original, expected),
                    )
                    numerical_gate = (
                        "FAIL"
                        if any(v["violations"] for v in observations.values())
                        else "PASS"
                    )
                    print(
                        "SM90_OUTPUT_PROJECTION_GRAPH_CONTRACT "
                        + json.dumps(
                            dict(
                                M=rows,
                                multiplier=multiplier,
                                numerical_gate=numerical_gate,
                                tolerance=dict(rtol=0.015, atol=0.02),
                                observations=observations,
                                acceptance_scope=(
                                    "bitwise equivalence to existing native-A8 path; not full FP32 stress acceptance"
                                ),
                                graph_vs_eager_bytes=_byte_differences(captured, eager),
                                graph_vs_original_bytes=_byte_differences(
                                    captured, original
                                ),
                            )
                        ),
                        flush=True,
                    )
                    torch.testing.assert_close(
                        captured.view(torch.int16),
                        eager.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        captured.view(torch.int16),
                        original.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )
                    self.assertTrue(
                        all(value["finite"] for value in observations.values())
                    )
                    if require_fp32_acceptance:
                        torch.testing.assert_close(
                            captured, expected, rtol=0.015, atol=0.02
                        )
                        torch.testing.assert_close(
                            original, expected, rtol=0.015, atol=0.02
                        )
                    for _ in range(3):
                        graph.replay()
                        torch.testing.assert_close(
                            captured.view(torch.int16),
                            eager.view(torch.int16),
                            rtol=0,
                            atol=0,
                        )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_OUTPUT_PROJECTION_DIAG") == "1",
        "explicit rounding/accumulation diagnostic opt-in required",
    )
    @torch.no_grad()
    def test_diagnose_rounding_and_scaled_graph(self):
        from pathlib import Path

        destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        self.assertTrue(
            destination, "Diagnostics must retain identical input artifacts"
        )
        directory = Path(destination) / "wo_a_rounding_diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        fixture = self._fixture()
        weight, scales, candidate, baseline = fixture
        torch.save(
            dict(
                weight_bytes=weight.view(torch.uint8).cpu(),
                scale_bytes=scales.view(torch.uint8).cpu(),
                weight_dtype=str(weight.dtype),
                scale_dtype=str(scales.dtype),
            ),
            directory / "shared_weights.pt",
        )
        records = []

        # Reproduce the unchanged M1/7/16 graph campaign and its x16 first
        # mutation. Additional diagnostics consume no RNG draws between cases.
        for rows in (1, 7, 16):
            o, freqs = _input(rows, strided=rows == 7)
            candidate.forward(o, freqs)
            graph, captured = _capture(lambda: candidate.forward(o, freqs))
            o.copy_(torch.randn_like(o) * 16.0)
            angles = torch.randn(freqs.shape, device=freqs.device)
            freqs.copy_(torch.polar(torch.ones_like(angles), angles))
            q, packed = self._check_pair(o, freqs, fixture)
            graph.replay()
            graph_output = captured.clone()
            eager_output = candidate.forward(o, freqs)
            baseline_output = self._legacy(o, freqs, baseline)
            expected32 = _fp32_oracle(q, packed, weight, scales)
            expected64_gpu = _fp64_oracle(q, packed, weight, scales, device=o.device)
            expected64_cpu = _fp64_oracle(
                q, packed, weight, scales, device=torch.device("cpu")
            )
            cpu_reference = expected64_cpu.to(device=o.device, dtype=torch.bfloat16)
            gpu_reference = expected64_gpu.to(torch.bfloat16)
            record = dict(
                stage="changed_input_graph",
                M=rows,
                multiplier=16.0,
                status="DIAGNOSTIC_ONLY_NOT_ACCEPTANCE",
                tolerance=dict(rtol=0.015, atol=0.02),
                versus_fp32={
                    "graph": _metrics(graph_output, expected32),
                    "eager": _metrics(eager_output, expected32),
                    "eight_humming": _metrics(baseline_output, expected32),
                },
                versus_gpu_fp64={
                    "graph": _metrics(graph_output, gpu_reference),
                    "eager": _metrics(eager_output, gpu_reference),
                    "eight_humming": _metrics(baseline_output, gpu_reference),
                },
                versus_cpu_fp64={
                    "graph": _metrics(graph_output, cpu_reference),
                    "eager": _metrics(eager_output, cpu_reference),
                    "eight_humming": _metrics(baseline_output, cpu_reference),
                },
                byte_differences=dict(
                    graph_vs_eager=_byte_differences(graph_output, eager_output),
                    graph_vs_baseline=_byte_differences(graph_output, baseline_output),
                    eager_vs_baseline=_byte_differences(eager_output, baseline_output),
                    fp32_vs_gpu64=_byte_differences(expected32, gpu_reference),
                    gpu64_vs_cpu64=_byte_differences(gpu_reference, cpu_reference),
                ),
                unrounded_fp64_gpu_cpu_max_abs=float(
                    (expected64_gpu.cpu() - expected64_cpu).abs().max()
                ),
            )
            torch.save(
                dict(
                    o=o.cpu(),
                    freqs=freqs.cpu(),
                    input_stride=o.stride(),
                    quantized_bytes=q.contiguous().view(torch.uint8).cpu(),
                    packed_scales=packed.cpu(),
                    graph=graph_output.cpu(),
                    eager=eager_output.cpu(),
                    eight_humming=baseline_output.cpu(),
                    expected_fp32=expected32.cpu(),
                    expected_gpu_fp64=expected64_gpu.cpu(),
                    expected_cpu_fp64=expected64_cpu,
                ),
                directory / f"changed_input_m{rows}.pt",
            )
            records.append(record)
            print("SM90_OUTPUT_PROJECTION_DIAGNOSTIC " + json.dumps(record), flush=True)
            (directory / "diagnostics.json").write_text(json.dumps(records, indent=2))

        # Reproduce the original benchmark's RNG sequence up to M1024.
        torch.manual_seed(419331)
        for rows in (1, 2, 4, 8, 16, 128, 1024):
            o, freqs = _input(rows)
        rotated = _inverse_rope_bf16(o, freqs).reshape(1024, 8, 4096)
        reference_q, reference_s = _independent_quantize(rotated, backend="auto")
        legacy_pairs = [
            linear._quantize_input(rotated[:, group].contiguous())
            for group, linear in enumerate(baseline)
        ]
        legacy_q = torch.stack([pair[0] for pair in legacy_pairs], dim=1)
        legacy_s = torch.stack([pair[1] for pair in legacy_pairs], dim=1)
        for imaginary_first in (False, True):
            q, packed = candidate.quantize(
                o, freqs, imaginary_fma_first=imaginary_first
            )
            differences = q.view(torch.uint8) != reference_q.view(torch.uint8)
            coordinates = differences.nonzero()[:16]
            examples = []
            for token, group, column in coordinates.cpu().tolist():
                head, dim = group * 8 + column // 512, column % 512
                examples.append(
                    dict(
                        token=token,
                        group=group,
                        column=column,
                        head=head,
                        dim=dim,
                        input_value=float(o[token, head, dim]),
                        input_partner=float(o[token, head, dim ^ 1]),
                        cos=(
                            float(freqs[token, (dim - 448) // 2].real)
                            if dim >= 448
                            else None
                        ),
                        sin=(
                            float(freqs[token, (dim - 448) // 2].imag)
                            if dim >= 448
                            else None
                        ),
                        rotated_bf16=float(rotated[token, group, column]),
                        candidate_byte=int(q.view(torch.uint8)[token, group, column]),
                        reference_byte=int(
                            reference_q.view(torch.uint8)[token, group, column]
                        ),
                    )
                )
            record = dict(
                stage="rope_quant",
                M=1024,
                imaginary_fma_first=imaginary_first,
                status="DIAGNOSTIC_ONLY_NOT_ACCEPTANCE",
                payload_vs_oracle=_byte_differences(q, reference_q),
                payload_vs_legacy=_byte_differences(q, legacy_q),
                scales_vs_oracle=_byte_differences(packed, reference_s),
                scales_vs_legacy=_byte_differences(packed, legacy_s),
                legacy_payload_vs_oracle=_byte_differences(legacy_q, reference_q),
                examples=examples,
            )
            records.append(record)
            print("SM90_OUTPUT_PROJECTION_DIAGNOSTIC " + json.dumps(record), flush=True)
            (directory / "diagnostics.json").write_text(json.dumps(records, indent=2))
        torch.save(
            dict(o=o.cpu(), freqs=freqs.cpu()), directory / "rope_quant_m1024.pt"
        )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_OUTPUT_PROJECTION_BENCH") == "1",
        "explicit correctness-gated component benchmark opt-in required",
    )
    @torch.no_grad()
    def test_benchmark_complete_chains_after_correctness(self):
        def timing(graph, count=100):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(count):
                graph.replay()
            stop.record()
            stop.synchronize()
            return start.elapsed_time(stop) / count

        cases = [(8, rows) for rows in (1, 2, 4, 8, 16, 128, 1024, 4096)]
        cases += [(1, rows) for rows in (1, 128, 1024, 4096)]
        for groups, rows in cases:
            fixture = self._fixture(groups, 1024, 4096)
            candidate, baseline = fixture[2:]
            o, freqs = _input(rows, groups=groups)
            self._check_outputs(o, freqs, fixture)
            old_input, new_input = torch.empty_like(o), torch.empty_like(o)

            def old_chain():
                old_input.copy_(o)
                return self._legacy(old_input, freqs, baseline, clone=False)

            def new_chain():
                new_input.copy_(o)
                return candidate.forward(new_input, freqs)

            old_graph, old_output = _capture(old_chain)
            new_graph, new_output = _capture(new_chain)
            old_graph.replay()
            new_graph.replay()
            expected = _fp32_oracle(*self._check_pair(o, freqs, fixture), *fixture[:2])
            torch.testing.assert_close(old_output, expected, rtol=0.015, atol=0.02)
            torch.testing.assert_close(new_output, expected, rtol=0.015, atol=0.02)
            baseline_label = "eight_humming_ms" if groups == 8 else "one_humming_ms"
            measured = {baseline_label: [], "grouped_humming_ms": []}
            for repeat in range(5):
                order = ((baseline_label, old_graph), ("grouped_humming_ms", new_graph))
                for label, graph in order if repeat % 2 == 0 else reversed(order):
                    measured[label].append(timing(graph))
            print(
                "SM90_OUTPUT_PROJECTION_BENCH "
                + json.dumps(
                    dict(
                        M=rows,
                        G=groups,
                        N=1024,
                        K=4096,
                        parallel_shape=(
                            "TP1 local WO_A" if groups == 8 else "TP8 local WO_A"
                        ),
                        default_decode_gate=candidate.supports(o, freqs),
                        baseline_gemm_count=groups,
                        precision="W8A8 G32, inverse RoPE rounded BF16 before quantization",
                        scope="WO_A only; inverse RoPE/quant/scale conversion/GEMM/stack; excludes WO_B",
                        staging="Both arms include identical source-to-scratch copy to reset in-place legacy RoPE",
                        frequency_handling="Legacy directly uses per-token frequencies; no factor-one repeat copy",
                        eligibility="Both arms passed independent FP32 .015/.02 gate before timing",
                        stress_acceptance="Not implied: amplified-input native-A8 FP32 stress remains separately reported",
                        profile="No profiler active during clean CUDA graph event timing",
                        **measured,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
