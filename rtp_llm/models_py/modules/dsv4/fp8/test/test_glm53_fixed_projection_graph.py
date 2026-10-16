"""Fixed-row graph must preserve native FP32 GEMM shapes and owned outputs."""

import json
import os
import statistics
import time
import unittest
from unittest import mock

import torch
from rtp_llm.models_py.modules.dsv4.fp8.indexer import _linear_fixed_rows
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.glm53_fixed_projection_graph import (
    fixed_projection_graph,
    projection_graph_stats,
)
from rtp_llm.models_py.modules.glm53_router import Glm53FP32Router


class FixedProjectionGraphTest(unittest.TestCase):
    def setUp(self):
        from rtp_llm.models_py.modules import glm53_fixed_projection_graph as impl
        from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA

        patch = mock.patch.object(impl, "_COMPILED_DISPATCH", True)
        patch.start()
        self.addCleanup(patch.stop)
        token = _FORWARD_METADATA.set({})
        self.addCleanup(_FORWARD_METADATA.reset, token)

    def test_compiled_dispatch_dtype_alignment_device_and_stream(self):
        from rtp_llm.models_py.modules import glm53_fixed_projection_graph as impl

        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.backends.cuda.matmul.allow_tf32 = False
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
            for device in range(min(2, torch.cuda.device_count())):
                with torch.cuda.device(device):
                    for dtype in (torch.bfloat16, torch.float16, torch.float32):
                        for n in (1, 13, 16, 81, 513):
                            x = torch.randn(n, 129, device="cuda", dtype=dtype)
                            w = torch.randn(7, 129, device="cuda") * 0.01
                            with mock.patch.object(impl, "_COMPILED_DISPATCH", False):
                                expected = fixed_projection_graph(x, w, 512)
                            actual = fixed_projection_graph(x, w, 512)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # Contiguous tensors may have unaligned storage offsets.
                    x = torch.randn(81 * 129 + 1, device="cuda", dtype=torch.bfloat16)[
                        1:
                    ].view(81, 129)
                    w = torch.randn(7 * 129 + 1, device="cuda")[1:].view(7, 129)
                    with mock.patch.object(impl, "_COMPILED_DISPATCH", False):
                        expected = fixed_projection_graph(x, w, 512)
                    with mock.patch.object(
                        impl,
                        "_pack_runner",
                        side_effect=AssertionError("unaligned warmup"),
                    ):
                        actual = fixed_projection_graph(x, w, 512)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # An odd fixed row count also misaligns the second slab,
                    # even when the input tensors' initial pointers are aligned.
                    x = torch.randn(8193, 129, device="cuda", dtype=torch.bfloat16)
                    w = torch.randn(7, 129, device="cuda") * 0.01
                    with mock.patch.object(impl, "_COMPILED_DISPATCH", False):
                        expected = fixed_projection_graph(x, w, 8191)
                    with mock.patch.object(
                        impl,
                        "_pack_runner",
                        side_effect=AssertionError("unaligned slab warmup"),
                    ):
                        actual = fixed_projection_graph(x, w, 8191)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        x = torch.randn(81, 129, device="cuda", dtype=torch.bfloat16)
                        w = torch.randn(7, 129, device="cuda") * 0.01
                        expected = fixed_projection_graph(x, w, 512)
                    stream.synchronize()
                    pending = torch.cuda.Event()
                    with torch.cuda.stream(stream):
                        torch.cuda._sleep(200_000_000)
                        pending.record()
                        actual = fixed_projection_graph(x, w, 512)
                        self.assertFalse(
                            pending.query(), "wrapper synchronized the caller stream"
                        )
                    stream.synchronize()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    # The public runner must use the new caller stream, even
                    # though compiled code was first obtained on another stream.
                    with mock.patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=True
                    ):
                        self.assertIsNone(fixed_projection_graph(x, w, 512))
        self.assertLessEqual(impl._pack_runner.cache_info().currsize, 128)
        self.assertLessEqual(impl._copy_runner.cache_info().currsize, 128)

    def test_compiled_dispatch_full_wrapper_benchmark(self):
        from rtp_llm.models_py.modules import glm53_fixed_projection_graph as impl

        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        impl._CACHE.clear()
        impl._STATS.clear()
        records = []
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
            for n, outputs in ((81, 32), (509, 256), (5101, 32), (9889, 256)):
                x = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(outputs, 8192, device="cuda") * 0.01
                for enabled in (False, True):
                    impl._COMPILED_DISPATCH = enabled
                    for _ in range(5):
                        fixed_projection_graph(x, w, 512)
                    torch.cuda.synchronize()
                # Alternate order, time the entire Python wrapper, and keep
                # synchronization outside each CPU interval. No speed threshold
                # is imposed on a shared CI machine; record both distributions.
                samples = {False: [], True: []}
                gpu_windows = {False: [], True: []}
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                end.record()
                end.synchronize()
                for repeat in range(60):
                    for enabled in (
                        (False, True) if repeat % 2 == 0 else (True, False)
                    ):
                        impl._COMPILED_DISPATCH = enabled
                        torch.cuda.synchronize()
                        begin.record()
                        start = time.perf_counter_ns()
                        out = fixed_projection_graph(x, w, 512)
                        samples[enabled].append((time.perf_counter_ns() - start) / 1000)
                        end.record()
                        end.synchronize()
                        gpu_windows[enabled].append(begin.elapsed_time(end) * 1000)
                        self.assertEqual(out.shape, (n, outputs))
                impl._COMPILED_DISPATCH = False
                expected = fixed_projection_graph(x, w, 512)
                impl._COMPILED_DISPATCH = True
                actual = fixed_projection_graph(x, w, 512)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                records.append(
                    {
                        "rows": n,
                        "outputs": outputs,
                        # Includes Python submission gaps, not kernel-only GPU time.
                        "gpu_window_us": {
                            str(enabled): {
                                "median": statistics.median(values),
                                "samples": values,
                            }
                            for enabled, values in gpu_windows.items()
                        },
                        "cpu_us": {
                            str(enabled): {
                                "median": statistics.median(values),
                                "samples": values,
                            }
                            for enabled, values in samples.items()
                        },
                    }
                )
        print("PROJECTION_DISPATCH_BENCH", json.dumps(records), flush=True)
        path = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if path:
            with open(os.path.join(path, "projection_dispatch.json"), "w") as output:
                json.dump(records, output, indent=2)

    def test_compact_middle_exact_budget_and_benchmark(self):
        self._check_compact_templates(
            "_COMPACT_MIDDLE",
            (512, 513, 1601, 2027, 2048, 2049, 16383),
            (513, 1601, 2027, 2049),
            "projection_middle.json",
        )

    def test_compact_tail_exact_budget_and_benchmark(self):
        self._check_compact_templates(
            "_COMPACT_TAIL",
            (8192, 8193, 8704, 8705, 9889, 10240, 10241, 16383),
            (8193, 8705, 9889, 10241),
            "projection_tail.json",
        )

    def test_compact_actual_model_shapes(self):
        # The exercised GLM53 checkpoint has hidden_size=4096 and 288 experts.
        # Keep the earlier 8192-wide shapes, but never infer model coverage from
        # those synthetic shapes alone.
        self._check_compact_templates(
            "_COMPACT_MIDDLE",
            (512, 513, 1601, 2027, 2048, 2049, 16383),
            (513, 1601, 2027, 2049),
            "projection_middle_actual.json",
            hidden=4096,
            output_widths=(32, 288),
        )
        self._check_compact_templates(
            "_COMPACT_TAIL",
            (8192, 8193, 8704, 8705, 9889, 10240, 10241, 16383),
            (8193, 8705, 9889, 10241),
            "projection_tail_actual.json",
            hidden=4096,
            output_widths=(32, 288),
        )

    def _check_compact_templates(
        self, feature, cases, timed, filename, hidden=8192, output_widths=(32, 256)
    ):
        from rtp_llm.models_py.modules import glm53_fixed_projection_graph as impl

        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        impl._CACHE.clear()
        impl._STATS.clear()
        records = []
        with mock.patch.dict(
            os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}
        ), mock.patch.object(impl, feature, True), mock.patch.object(
            impl, "_COMPACT_MIDDLE", True
        ):
            weights = [
                torch.randn(n, hidden, device="cuda") * 0.01 for n in output_widths
            ]
            # Warm all three capacities for both shapes. The prototype does
            # not yet prewarm the extra family in the model startup path.
            for w in weights:
                for m in (81, 1601, 5101):
                    x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
                    fixed_projection_graph(x, w, 512)
            warm = projection_graph_stats()
            self.assertEqual(warm["captures"], 6)
            self.assertEqual(warm.get("evictions", 0), 0)
            self.assertLessEqual(warm["bytes"], impl._MAX_BYTES)
            for w in weights:
                for m in cases:
                    x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
                    with mock.patch.object(impl, feature, False):
                        old = fixed_projection_graph(x, w, 512)
                    new = fixed_projection_graph(x, w, 512)
                    torch.testing.assert_close(new, old, rtol=0, atol=0)
                    with mock.patch.object(impl, "_COMPILED_DISPATCH", False):
                        jit = fixed_projection_graph(x, w, 512)
                    torch.testing.assert_close(jit, old, rtol=0, atol=0)
                    frozen = new.clone()
                    w.mul_(0.99)
                    fixed_projection_graph(x, w, 512)
                    torch.testing.assert_close(new, frozen, rtol=0, atol=0)
                    if m not in timed:
                        continue
                    cpu = {False: [], True: []}
                    gpu = {False: [], True: []}
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    end.record()
                    end.synchronize()
                    for repeat in range(40):
                        for enabled in ((False, True) if repeat % 2 else (True, False)):
                            with mock.patch.object(impl, feature, enabled):
                                torch.cuda.synchronize()
                                begin.record()
                                start = time.perf_counter_ns()
                                result = fixed_projection_graph(x, w, 512)
                                cpu[enabled].append(
                                    (time.perf_counter_ns() - start) / 1000
                                )
                                end.record()
                                end.synchronize()
                                gpu[enabled].append(begin.elapsed_time(end) * 1000)
                            self.assertEqual(result.shape, old.shape)
                    records.append(
                        {
                            "rows": m,
                            "hidden": hidden,
                            "outputs": w.shape[0],
                            "cpu_us": {
                                str(k): {"median": statistics.median(v), "samples": v}
                                for k, v in cpu.items()
                            },
                            "gpu_window_us": {
                                str(k): {"median": statistics.median(v), "samples": v}
                                for k, v in gpu.items()
                            },
                        }
                    )
            after = projection_graph_stats()
            self.assertEqual(after["captures"], warm["captures"])
            self.assertEqual(after.get("evictions", 0), 0)
            self.assertLessEqual(after["bytes"], impl._MAX_BYTES)
        print(
            feature,
            json.dumps({"stats": after, "cases": records}),
            flush=True,
        )
        path = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if path:
            with open(os.path.join(path, filename), "w") as output:
                json.dump({"stats": after, "cases": records}, output, indent=2)

    def test_decode_keeps_original_projection_path(self):
        from rtp_llm.models_py.modules.dsv4.forward_metadata import _FORWARD_METADATA

        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.backends.cuda.matmul.allow_tf32 = False
        x = torch.randn(17, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(32, 4096, device="cuda") * 0.01
        router = Glm53FP32Router(torch.randn(4096, 256, device="cuda") * 0.01)
        token = _FORWARD_METADATA.set(None)
        try:
            with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}):
                expected = _linear_fixed_rows(x, w, 512)
                expected_router = fixed_m_linear(router, x, 512)
            before = projection_graph_stats().get("replays", 0)
            with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
                actual = _linear_fixed_rows(x, w, 512)
                actual_router = fixed_m_linear(router, x, 512)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_router, expected_router, rtol=0, atol=0)
            self.assertEqual(projection_graph_stats().get("replays", 0), before)
        finally:
            _FORWARD_METADATA.reset(token)

    def test_hot_model_templates_stay_within_existing_budget(self):
        from rtp_llm.models_py.modules import glm53_fixed_projection_graph as impl

        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.synchronize()
        impl._CACHE.clear()
        impl._STATS.clear()
        weights = [torch.randn(n, 8192, device="cuda") * 0.01 for n in (32, 256)]
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
            for w in weights:
                for n in (81, 16383):
                    x = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16)
                    self.assertIsNotNone(fixed_projection_graph(x, w, 512))
            warm = projection_graph_stats()
            self.assertLessEqual(warm["bytes"], 512 * 1024**2)
            self.assertEqual(warm.get("evictions", 0), 0)
            for n in (81, 509, 513, 1601, 5102, 7711, 13931, 16383):
                x = torch.randn(n, 8192, device="cuda", dtype=torch.bfloat16)
                for w in weights:
                    actual = fixed_projection_graph(x, w, 512)
                    with mock.patch.dict(
                        os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}
                    ):
                        expected = _linear_fixed_rows(x, w, 512)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            after = projection_graph_stats()
            self.assertEqual(after["captures"], warm["captures"])
            self.assertEqual(after.get("evictions", 0), 0)
            self.assertLessEqual(after["bytes"], 512 * 1024**2)
            print("Bounded hot projection templates:", after)

    def test_exact_indexer_router_and_changed_weights(self):
        self.assertTrue(torch.cuda.is_available(), "GPU validation must not skip")
        torch.manual_seed(531013)
        torch.backends.cuda.matmul.allow_tf32 = False
        for outputs, hidden in ((32, 4096), (256, 4096), (7, 129)):
            w = torch.randn(outputs, hidden, device="cuda") * 0.01
            for n in (1, 17, 509, 512, 513, 1601, 7711, 16383):
                x = torch.randn(n, hidden, device="cuda", dtype=torch.bfloat16)
                with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}):
                    expected = _linear_fixed_rows(x, w, 512)
                with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
                    actual = _linear_fixed_rows(x, w, 512)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    frozen = actual.clone()
                    w.mul_(0.99)
                    changed = fixed_projection_graph(x, w, 512)
                    self.assertIsNotNone(changed)
                    with mock.patch.dict(
                        os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}
                    ):
                        new_expected = _linear_fixed_rows(x, w, 512)
                    torch.testing.assert_close(changed, new_expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual, frozen, rtol=0, atol=0)
        router = Glm53FP32Router(torch.randn(4096, 256, device="cuda") * 0.01)
        x = torch.randn(13931, 4096, device="cuda", dtype=torch.bfloat16)
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "0"}):
            old = fixed_m_linear(router, x, 512)
        with mock.patch.dict(os.environ, {"GLM53_FIXED_PROJECTION_GRAPH": "1"}):
            new = fixed_m_linear(router, x, 512)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        print("Projection graph stats:", projection_graph_stats())


if __name__ == "__main__":
    unittest.main()
