"""Qualify the existing DeepGEMM HBR-input/BHD-output wo_a consumer."""

import json
import os
import statistics
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import linear as linear_module
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if actual.numel():
        test.assertTrue(torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        ))


def reference_group_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scales = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.permute(1, 0, 2), scales.permute(1, 0, 2)


def make_woa():
    weight = torch.randn(8192, 4096, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.rand(8192, 1, dtype=torch.float32, device="cuda") * 0.01 + 0.001
    op = PPUPTPCLinear(weight, scale)
    op.prepare_grouped(8, 1024)
    return op


class PTPCWoaDirectCPU(unittest.TestCase):
    def test_qualified_default_and_rollback(self):
        with patch.dict(os.environ):
            os.environ.pop(linear_module.PTPC_WOA_DIRECT_ENV, None)
            self.assertTrue(linear_module.ptpc_woa_direct_enabled())
        for value, expected in (("0", False), ("1", True)):
            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: value}):
                self.assertEqual(linear_module.ptpc_woa_direct_enabled(), expected)

    def test_missing_provider_declines_before_activation_work(self):
        with patch.object(linear_module.importlib, "import_module", side_effect=ModuleNotFoundError):
            self.assertIsNone(linear_module._load_direct_woa_gemm())
        for module in (SimpleNamespace(), SimpleNamespace(fp8_bmm=17)):
            with patch.object(linear_module.importlib, "import_module", return_value=module):
                self.assertIsNone(linear_module._load_direct_woa_gemm())
        with patch.object(linear_module.importlib, "import_module", side_effect=RuntimeError("provider failed")):
            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                linear_module._load_direct_woa_gemm()

    def test_python_jit_build_lock_is_installed_before_returning_provider(self):
        operation = lambda *args: None
        with patch(
            "rtp_llm.models_py.kernels.ppu.runtime.install_deep_gemm_build_lock"
        ) as install, patch.object(
            linear_module.importlib, "import_module", return_value=SimpleNamespace(fp8_bmm=operation)
        ):
            self.assertIs(linear_module._load_direct_woa_gemm(), operation)
            install.assert_called_once_with()


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU device test"
)
class PTPCWoaDirectDevice(unittest.TestCase):
    @torch.no_grad()
    def test_native_payload_output_and_changed_input_graph_exact(self):
        torch.manual_seed(4151)
        op = make_woa()
        self.assertIsNotNone(op._group_gemm, "public native einsum is required as the independent path")
        self.assertIsNotNone(op._group_direct_gemm, "installed Python JIT BMM must be present for qualification")
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}):
            for rows in (1, 8, 144, 4096):
                with self.subTest(rows=rows):
                    value = torch.randn(rows, 8, 4096, dtype=torch.bfloat16, device="cuda")
                    with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "0"}):
                        expected = op.forward_grouped(value, prefill=True)
                    provider = op._group_direct_gemm
                    prepared = []

                    def record(payload, scales, weight, weight_scale, output):
                        prepared[:] = (payload, scales)
                        self.assertTrue(payload.is_contiguous())
                        self.assertTrue(scales.is_contiguous())
                        self.assertIs(weight, op._group_weight)
                        self.assertIs(weight_scale, op._group_scale)
                        return provider(payload, scales, weight, weight_scale, output)

                    with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "1"}), patch.object(
                        op, "_group_direct_gemm", side_effect=record
                    ):
                        actual = op.forward_grouped(value, prefill=True)
                        assert_bits(self, actual, expected)
                        for actual_part, expected_part in zip(prepared, reference_group_quant(value)):
                            assert_bits(self, actual_part, expected_part)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = op.forward_grouped(value, prefill=True)
                        for magnitude in (0.0, 0.125, 2.0):
                            value.copy_(torch.randn_like(value) * magnitude)
                            graph.replay()
                            for actual_part, expected_part in zip(prepared, reference_group_quant(value)):
                                assert_bits(self, actual_part, expected_part)
                            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "0"}):
                                expected = op.forward_grouped(value, prefill=True)
                            assert_bits(self, captured, expected)
                            self.assertTrue(captured.is_contiguous())
            # Absence is an admission fallback; a runtime failure after a
            # selected consumer writes must propagate instead of retrying.
            value = torch.randn(1, 8, 4096, dtype=torch.bfloat16, device="cuda")
            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "1"}), patch.object(
                op, "_group_direct_gemm", None
            ):
                fallback = op.forward_grouped(value, prefill=True)
            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "0"}):
                assert_bits(self, fallback, op.forward_grouped(value, prefill=True))

            def fail_after_write(payload, scale, weight, weight_scale, output):
                output.zero_()
                raise RuntimeError("selected provider failed after write")

            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "1"}), patch.object(
                op, "_group_direct_gemm", side_effect=fail_after_write
            ), patch.object(op, "_group_gemm", side_effect=AssertionError("must not retry public einsum")):
                with self.assertRaisesRegex(RuntimeError, "failed after write"):
                    op.forward_grouped(value, prefill=True)

    @torch.no_grad()
    def test_actual_two_launch_profile_and_component_latency(self):
        root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if root:
            report = profile_woa_direct(Path(root) / "ptpc-woa-direct")
        else:
            with tempfile.TemporaryDirectory(prefix="ptpc-woa-direct-") as directory:
                report = profile_woa_direct(Path(directory))
        for case in report["cases"]:
            baseline, candidate = case["public"], case["direct"]
            self.assertEqual(baseline["kernel_count"], 3)
            self.assertEqual(candidate["kernel_count"], 2)
            self.assertEqual(baseline["physical_permute_kernel_count"], 1)
            self.assertEqual(candidate["physical_permute_kernel_count"], 0)
            self.assertEqual(baseline["batched_gemm_kernel_count"], 1)
            self.assertEqual(candidate["batched_gemm_kernel_count"], 1)
            if "ordinary" in case:
                self.assertEqual(case["ordinary"]["kernel_count"], 3)
                self.assertEqual(case["ordinary"]["physical_permute_kernel_count"], 1)
                self.assertEqual(case["ordinary"]["batched_gemm_kernel_count"], 1)
        print("PTPC_WOA_DIRECT_PROFILE=" + json.dumps(report, sort_keys=True))


@torch.no_grad()
def profile_woa_direct(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(4152)
    op = make_woa()
    if op._group_gemm is None or op._group_direct_gemm is None:
        raise AssertionError("both installed provider consumers are required for qualification")
    report = {
        "geometry": {"groups": 8, "n": 1024, "k": 4096},
        "warmup": 30, "measure": 100, "cases": [],
        "event_metric": "whole_call_stream_latency_us_including_host_launch_gaps",
        "profile_metric": "sum_actual_GPU_kernel_durations_us_diagnostic_only",
    }
    with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}):
        for rows in (1, 8, 144, 4096):
            value = torch.randn(rows, 8, 4096, dtype=torch.bfloat16, device="cuda")
            case = {"rows": rows, "output_byte_exact": True}
            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "0"}):
                expected = op.forward_grouped(value, prefill=True)
            with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: "1"}):
                actual = op.forward_grouped(value, prefill=True)
            if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                raise AssertionError(f"wo_a rows={rows}: output bytes differ before measurement")
            variants = [("public", "0", True), ("direct", "1", True)]
            if rows == 8:
                variants.append(("ordinary", "1", False))
            for name, enabled, prefill in variants:
                with patch.dict(os.environ, {linear_module.PTPC_WOA_DIRECT_ENV: enabled}):
                    for _ in range(30):
                        output = op.forward_grouped(value, prefill=prefill)
                    torch.cuda.synchronize()
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    samples = []
                    for _ in range(100):
                        start.record()
                        output = op.forward_grouped(value, prefill=prefill)
                        stop.record()
                        stop.synchronize()
                        samples.append(start.elapsed_time(stop) * 1000)
                    path = directory / f"m{rows}-{name}.json"
                    if path.exists():
                        raise FileExistsError(path)
                    torch.cuda.synchronize()
                    with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=True,
                    ) as profiler:
                        # The ordinary variant omits the opt-in entirely.
                        output = op.forward_grouped(value, prefill=True) if prefill else op.forward_grouped(value)
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(path))
                if not torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
                    raise AssertionError(f"wo_a rows={rows}: measured output bytes differ")
                events = json.loads(path.read_text())["traceEvents"]
                kernels = [event for event in events if event.get("cat") == "kernel" and event.get("ph") == "X"]
                if not kernels:
                    raise AssertionError("wo_a trace contains no actual GPU kernel events")
                case[name] = {
                    "trace": str(path), "kernel_count": len(kernels),
                    "physical_permute_kernel_count": sum("fused_permute" in event["name"] for event in kernels),
                    "batched_gemm_kernel_count": sum("gemm" in event["name"].lower() for event in kernels),
                    "gpu_kernel_sum_us": sum(event["dur"] for event in kernels),
                    "stream_call_median_us": statistics.median(samples), "stream_call_samples_us": samples,
                    "kernels": [{"name": event["name"], "duration_us": event["dur"]} for event in kernels],
                }
            report["cases"].append(case)
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    unittest.main()
