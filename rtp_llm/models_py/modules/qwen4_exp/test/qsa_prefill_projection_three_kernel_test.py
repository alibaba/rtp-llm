"""SM90 prefill gates for the real two-quant + one-library-GEMM QSA projection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base.common import qsa_fp8_pair
from rtp_llm.models_py.modules.base.common.fp8_linear import (
    Fp8Linear,
    prepare_fp8_pair,
    project_fp8_pair,
)
from rtp_llm.models_py.modules.base.common.qsa_fp8_pair import (
    quantize_fp8_prefill_input,
)


def _rms_error(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt().clamp_min(1.0e-12)
    return (error / reference).item()


def _metrics(actual, expected):
    return {
        "max_absolute_error": (actual.float() - expected.float()).abs().max().item(),
        "relative_rms_error": _rms_error(actual, expected),
        "different_elements": (actual != expected).count_nonzero().item(),
        "elements": actual.numel(),
    }


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class QsaPrefillProjectionThreeKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (9, 0):
            raise unittest.SkipTest("SM90 prefill projection target")

    def setUp(self):
        torch.manual_seed(110103)

    def _pair(self, scales=(0.119713, 0.03125)):
        def projection(width, scale):
            weight = (torch.randn(width, 2560, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            return Fp8Linear(
                weight,
                torch.tensor([scale], device="cuda"),
                role="PREFILL",
            )

        first = projection(3072, scales[0])
        second = projection(640, scales[1])
        pair = prepare_fp8_pair(first, second)
        self.assertIsNotNone(pair)
        return pair, first, second

    def _assert_quant_matches_native(self, x, quantized=None):
        q, scales = quantized or quantize_fp8_prefill_input(x)
        old_m = (x.shape[0] + 15) // 16 * 16
        old_q = torch.empty((old_m, 2560), dtype=torch.float8_e4m3fn, device=x.device)
        expected_q, expected_scale = scaled_fp8_per_tensor_quant(x, output=old_q)
        self.assertEqual(q.shape[0], old_m)
        self.assertEqual(tuple(scales.shape), (q.shape[0], 1))
        self.assertTrue(
            torch.equal(
                q[: x.shape[0]].view(torch.uint8),
                expected_q[: x.shape[0]].view(torch.uint8),
            ),
            "real-token FP8 bytes changed",
        )
        self.assertTrue(torch.all(q[x.shape[0] :].view(torch.uint8) == 0).item())
        self.assertTrue(torch.all(scales == expected_scale).item())

    def test_quant_native_bytes_real_domain_and_padding(self):
        for rows in (
            1,
            15,
            16,
            127,
            128,
            129,
            1371,
            1383,
            1576,
            2048,
            8192,
            8193,
            32768,
            32769,
        ):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
                self._assert_quant_matches_native(x)

    def test_quant_zero_tiny_nan_inf_and_decreasing_amax(self):
        for gain in (300.0, 0.001, 0.0, 1.0e-12, 1.0e-40):
            with self.subTest(gain=gain):
                x = torch.randn(129, 2560, dtype=torch.bfloat16, device="cuda") * gain
                self._assert_quant_matches_native(x)
        x = torch.randn(129, 2560, dtype=torch.bfloat16, device="cuda")
        x[0, 0] = float("nan")
        self._assert_quant_matches_native(x)
        x[0, 1] = float("inf")
        x[0, 2] = -float("inf")
        self._assert_quant_matches_native(x)
        x.fill_(float("nan"))
        self._assert_quant_matches_native(x)

    def test_per_call_scratch_isolated_on_two_streams(self):
        xs = [
            torch.randn(1371, 2560, dtype=torch.bfloat16, device="cuda") * gain
            for gain in (300.0, 0.001)
        ]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for stream, x in zip(streams, xs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                results.append(quantize_fp8_prefill_input(x))
        for stream in streams:
            stream.synchronize()
        for x, result in zip(xs, results):
            self._assert_quant_matches_native(x, result)
        self.assertNotEqual(results[0][1].data_ptr(), results[1][1].data_ptr())

    def test_outputs_independent_scales_old_pair_and_bf16_reference(self):
        pair, first, second = self._pair()
        for rows in (
            1,
            127,
            128,
            1371,
            1383,
            1576,
            2048,
            4096,
            8192,
            8193,
            16384,
            32768,
            32769,
        ):
            x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
            expected = project_fp8_pair(x, first, second)
            with patch.object(
                qsa_fp8_pair,
                "_project_fp8_prefill_native",
                wraps=qsa_fp8_pair._project_fp8_prefill_native,
            ) as native:
                actual = pair.project(x, first=first, second=second)
                self.assertEqual(native.call_count, 1)
                self.assertIs(native.call_args.args[0], x)
                self.assertIs(native.call_args.args[1], pair.weight_t)
                self.assertIs(native.call_args.args[2], pair.weight_scales)
                self.assertIs(native.call_args.args[3], pair._source_guard)
                self.assertTrue(
                    all(
                        a is b
                        for a, b in zip(native.call_args.args[4], pair._source_tensors)
                    )
                )
                self.assertEqual(tuple(native.call_args.args[1].shape), (2560, 3712))
            self.assertIsNotNone(actual)
            for name, value, baseline, projection in zip(
                ("gate", "qk"), actual, expected, (first, second)
            ):
                self.assertEqual(value.dtype, torch.bfloat16)
                self.assertEqual(value.stride(), (3712, 1))
                self.assertEqual(value.is_contiguous(), rows == 1)
                torch.testing.assert_close(value, baseline, rtol=1.0e-2, atol=1.0e-3)
                # Independent unquantized BF16-input reference, including the
                # checkpoint weight dequantization's BF16 rounding boundary.
                reference = F.linear(
                    x, (projection.weight.float() * projection.scale).to(torch.bfloat16)
                )
                old_error = _rms_error(baseline, reference)
                new_error = _rms_error(value, reference)
                self.assertLessEqual(new_error, old_error * 1.10 + 1.0e-4)
                self.assertLess(new_error, 0.04)
                print(
                    json.dumps(
                        {
                            "metric": "qsa_three_kernel_numeric",
                            "rows": rows,
                            "projection": name,
                            "candidate_vs_old": _metrics(value, baseline),
                            "candidate_vs_bf16": _metrics(value, reference),
                            "old_vs_bf16": _metrics(baseline, reference),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        torch.testing.assert_close(
            pair.weight_scales[:, :3072], first.scale.expand(1, 3072), rtol=0, atol=0
        )
        torch.testing.assert_close(
            pair.weight_scales[:, 3072:3712],
            second.scale.expand(1, 640),
            rtol=0,
            atol=0,
        )

    def test_runtime_three_kernels_without_copy_or_quantization_clear(self):
        from torch.profiler import ProfilerActivity, profile

        pair, first, second = self._pair()
        trace_dir = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
            or tempfile.mkdtemp(prefix="qsa_three_kernel_")
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        for rows in (1, 127, 1371, 8193, 32769):
            x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
            for _ in range(3):
                pair.project(x, first=first, second=second)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                output = pair.project(x, first=first, second=second)
                torch.cuda.synchronize()
            self.assertIsNotNone(output)
            path = trace_dir / f"three-kernel-{rows}.json"
            prof.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
            kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
            self.assertEqual(len(kernels), 3, kernels)
            self.assertIn("qsa_fp8_partial_amax_kernel", kernels[0])
            self.assertIn("qsa_fp8_quantize_from_partials_kernel", kernels[1])
            # The native entry uses accurate rowwise scaling for every input size.
            self.assertTrue(
                any(
                    marker in kernels[2].lower()
                    for marker in ("cutlass", "nvjet", "gemm", "xmma")
                ),
                kernels[2],
            )
            self.assertFalse([e for e in events if e.get("cat") == "gpu_memcpy"])
            # Retain and account for the cuBLAS workspace clears. Quantization
            # owns fully overwritten scratch and must not introduce a clear.
            memsets = [e for e in events if e.get("cat") == "gpu_memset"]
            self.assertLessEqual(len(memsets), 2, memsets)
            self.assertTrue(all(e["args"]["bytes"] == 4 for e in memsets), memsets)
            cpu_ops = [e["name"] for e in events if e.get("cat") == "cpu_op"]
            for operation in ("aten::cat", "aten::copy_", "aten::zero_", "aten::fill_"):
                self.assertNotIn(operation, cpu_ops)
            print(
                json.dumps(
                    {
                        "metric": "qsa_three_kernel_launches",
                        "rows": rows,
                        "kernels": kernels,
                        "library_gemm_kernel": kernels[2],
                        "library_memsets": [
                            {"bytes": e["args"]["bytes"], "duration_us": e["dur"]}
                            for e in memsets
                        ],
                    }
                ),
                flush=True,
            )

    def test_owner_dispatch_guards_decline_before_launch(self):
        pair, first, second = self._pair()
        x = torch.randn(1371, 2560, dtype=torch.bfloat16, device="cuda")
        module = "rtp_llm.models_py.modules.base.common.qsa_fp8_pair"
        with patch(module + "._project_fp8_prefill_native") as native:
            self.assertIsNone(pair.project(x, first=second, second=first))
            with patch.object(first, "forward", wraps=first.forward):
                self.assertIsNone(pair.project(x, first=first, second=second))
            handle = second.register_forward_hook(lambda module, args, output: output)
            try:
                self.assertIsNone(pair.project(x, first=first, second=second))
            finally:
                handle.remove()
            native.assert_not_called()
        pair, first, second = self._pair()
        first.weight = first.weight.clone()
        self.assertIsNone(pair.project(x, first=first, second=second))

    def test_native_guard_rejects_invalid_inputs_or_sources_before_device_work(self):
        from contextlib import nullcontext

        from torch.profiler import ProfilerActivity, profile

        for change in (
            "scale_version",
            "weight_storage",
            "weight_shape",
            "scale_dtype",
            "scale_grad",
            "inference_shape",
            "input_empty",
            "input_dtype",
            "input_strided",
            "input_transpose",
            "input_unaligned",
            "input_grad",
            "input_cpu",
        ):
            with self.subTest(change=change):
                context = (
                    torch.inference_mode()
                    if change == "inference_shape"
                    else nullcontext()
                )
                with context:
                    pair, first, second = self._pair()
                    x = torch.randn(1371, 2560, dtype=torch.bfloat16, device="cuda")
                    self.assertIsNotNone(pair.project(x, first=first, second=second))
                    if change == "scale_version":
                        first.scale.mul_(2)
                    elif change == "weight_storage":
                        version = first.weight._version
                        first.weight.data = first.weight.clone()
                        self.assertEqual(first.weight._version, version)
                    elif change == "weight_shape":
                        first.weight.transpose_(0, 1)
                    elif change == "scale_dtype":
                        first.scale.data = first.scale.double()
                    elif change == "scale_grad":
                        first.scale.requires_grad_(True)
                    elif change == "inference_shape":
                        first.scale.resize_(1, 1)
                    elif change == "input_empty":
                        x = x[:0]
                    elif change == "input_dtype":
                        x = x.float()
                    elif change == "input_strided":
                        x = x[:, ::2]
                    elif change == "input_transpose":
                        x = x.t()
                    elif change == "input_unaligned":
                        x = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[
                            1:
                        ].reshape_as(x)
                    elif change == "input_grad":
                        x.requires_grad_(True)
                    else:
                        x = x.cpu()
                    torch.cuda.synchronize()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                    ) as prof:
                        actual = pair.project(x, first=first, second=second)
                        torch.cuda.synchronize()
                    self.assertIsNone(actual)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "rejected-source.json"
                        prof.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                    self.assertFalse(
                        [
                            e
                            for e in events
                            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
                        ]
                    )

    def test_inference_owners_are_frozen_and_rebinding_declines(self):
        with torch.inference_mode():
            pair, first, second = self._pair()
            x = torch.randn(128, 2560, dtype=torch.bfloat16, device="cuda")
            self.assertIsNotNone(pair.project(x, first=first, second=second))
            second.scale = second.scale.clone()
            self.assertIsNone(pair.project(x, first=first, second=second))


if __name__ == "__main__":
    unittest.main()
