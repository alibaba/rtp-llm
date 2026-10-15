import json
import os
import statistics
import unittest
from collections import Counter
from pathlib import Path

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.triton_kv_scatter import (
    prepare_kv_scatter_indices,
    triton_kv_scatter,
)


class TritonKvScatterTest(unittest.TestCase):
    def test_trace_shape_launch_count_and_stream_latency(self) -> None:
        # Given: trace-sized BF16 inputs and caller-owned output buffers.
        total_q, n, heads, dim = 1252, 1247, 64, 512
        ids = torch.arange(n, device="cuda", dtype=torch.int64) + 2
        src = torch.randn((n, heads, dim), device="cuda", dtype=torch.bfloat16)
        baseline_out = torch.empty(
            (total_q, heads, dim), device="cuda", dtype=src.dtype
        )
        prepared_out = torch.empty_like(baseline_out)
        reverse_indices = prepare_kv_scatter_indices(ids, total_q)
        cases = {
            "original_fallback": lambda: triton_kv_scatter(
                src, ids, total_q, out=baseline_out
            ),
            "prepared_map": lambda: triton_kv_scatter(
                src, ids, total_q, out=prepared_out, reverse_indices=reverse_indices
            ),
        }
        for action in cases.values():
            for _ in range(30):
                action()
        torch.cuda.synchronize()
        expected = torch.zeros_like(baseline_out)
        expected[ids] = src
        torch.testing.assert_close(baseline_out, expected, rtol=0, atol=0)
        torch.testing.assert_close(prepared_out, expected, rtol=0, atol=0)

        # When: time warmed wrappers, then independently profile each path.
        report = {
            "shape": {"total_q": total_q, "N": n, "H": heads, "D": dim},
            "dtype": "bfloat16",
            "warmup_iterations": 30,
            "measurement_iterations": 100,
            "timing_source": "CUDA events: wrapper end-to-end stream latency",
            "includes_host_launch_gaps": True,
            "input_and_output_allocation_timed": False,
            "preparation_in_per_layer_timing": False,
            "compile_warmup_timed": False,
            "torch_compile_included": False,
            "scope": "single scatter wrapper; not model E2E or pure-kernel speedup",
            "correctness": "exact_equal_to_torch_scatter",
        }
        latencies = {}
        for name, action in cases.items():
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(100)
            ]
            torch.cuda.synchronize()
            for start, end in events:
                start.record()
                action()
                end.record()
            torch.cuda.synchronize()
            samples = [start.elapsed_time(end) * 1000 for start, end in events]
            latencies[name] = {
                "stream_latency_us": {
                    "median": statistics.median(samples),
                    "p90": sorted(samples)[89],
                    "min": min(samples),
                    "max": max(samples),
                },
                "samples_us": samples,
            }
        report["timings"] = latencies
        profiles = {}
        profile_actions = {
            **cases,
            "plan_preparation": lambda: prepare_kv_scatter_indices(ids, total_q),
        }
        for name, action in profile_actions.items():
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                action()
                torch.cuda.synchronize()
            kernels = [
                event
                for event in profile.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            profiles[name] = {
                "calls": 1,
                "launches_per_call": len(kernels),
                "kernel_names": dict(Counter(event.name for event in kernels)),
                "kernel_sum_us_per_call": sum(
                    event.device_time_total for event in kernels
                ),
            }
        report["profiles"] = profiles
        output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scatter_plan_launch_latency.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

        # Then: each prepared layer saves one launch; plan setup costs two.
        for name, count in (
            ("original_fallback", 2),
            ("prepared_map", 1),
            ("plan_preparation", 2),
        ):
            self.assertEqual(profiles[name]["launches_per_call"], count, profiles)
            names = profiles[name]["kernel_names"]
            self.assertEqual(
                sum(
                    count
                    for kernel, count in names.items()
                    if "_populate_rev_kernel" in kernel
                ),
                0 if name == "prepared_map" else 1,
                profiles,
            )
            self.assertEqual(
                sum(
                    count
                    for kernel, count in names.items()
                    if "_fused_scatter_kernel" in kernel
                ),
                0 if name == "plan_preparation" else 1,
                profiles,
            )
        torch.testing.assert_close(baseline_out, prepared_out, rtol=0, atol=0)

    def test_prepared_map_reuses_indices_across_layers(self) -> None:
        # Given: padded destination rows, strided int32 ids and strided sources.
        ids = torch.tensor([6, 99, 1, 99, 4, 99], device="cuda", dtype=torch.int32)[::2]
        reverse_indices = prepare_kv_scatter_indices(ids, 8)
        expected_map = torch.tensor(
            [-1, 1, -1, -1, 2, -1, 0, -1], device="cuda", dtype=torch.int32
        )
        torch.testing.assert_close(reverse_indices, expected_map, rtol=0, atol=0)

        for dtype in (torch.float32, torch.bfloat16):
            for layer in range(3):
                with self.subTest(dtype=dtype, layer=layer):
                    src = torch.arange(42, device="cuda", dtype=dtype).reshape(3, 7, 2)
                    src = (src + layer).transpose(1, 2)
                    out = torch.full(
                        (8, 2, 7), float("nan"), device="cuda", dtype=dtype
                    )
                    expected = torch.zeros_like(out)
                    expected[ids.long()] = src

                    # When: all layers consume the same prepared map.
                    result = triton_kv_scatter(
                        src, ids, 8, out=out, reverse_indices=reverse_indices
                    )

                    # Then: the caller's buffer contains copies and zeroed holes.
                    self.assertIs(result, out)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_new_plan_does_not_mutate_previous_map(self) -> None:
        # Given: ids change in place between plans of equal shape.
        ids = torch.tensor([1, 5, 7], device="cuda", dtype=torch.int64)
        src = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 2, 4)
        previous_ids = ids.clone()
        previous_map = prepare_kv_scatter_indices(ids, 9)
        ids.copy_(torch.tensor([0, 3, 8], device="cuda", dtype=torch.int64))
        current_map = prepare_kv_scatter_indices(ids, 9)

        # When: both independently owned plans remain in use.
        previous = triton_kv_scatter(src, previous_ids, 9, reverse_indices=previous_map)
        current = triton_kv_scatter(src, ids, 9, reverse_indices=current_map)

        # Then: each output reflects its own plan, without stale destinations.
        expected_previous = torch.zeros_like(previous)
        expected_previous[previous_ids] = src
        expected_current = torch.zeros_like(current)
        expected_current[ids] = src
        torch.testing.assert_close(previous, expected_previous, rtol=0, atol=0)
        torch.testing.assert_close(current, expected_current, rtol=0, atol=0)

    def test_empty_source_zeroes_destination(self) -> None:
        # Given: no rank-owned tokens but nonempty padded output.
        src = torch.empty((0, 2, 7), device="cuda", dtype=torch.bfloat16)
        ids = torch.empty(0, device="cuda", dtype=torch.int64)

        # When: scatter has no selected rows.
        result = triton_kv_scatter(src, ids, 5)

        # Then: every destination row is zero.
        torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_fallback_graph_replay_restores_fixed_indices(self) -> None:
        # Given: a graph with fixed ids and other calls sharing the fallback map.
        ids = torch.tensor([1, 5, 7], device="cuda", dtype=torch.int64)
        src = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 2, 4)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            triton_kv_scatter(src, ids, 9)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = triton_kv_scatter(src, ids, 9)
        graph.replay()
        torch.cuda.synchronize()
        other_ids = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int64)
        triton_kv_scatter(src, other_ids, 9)

        # When: replay rebuilds its own generation-tagged entries.
        graph.replay()

        # Then: the other call's destinations are zero.
        expected = torch.zeros_like(result)
        expected[ids] = src
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_prepared_graph_replay_uses_refreshed_plan(self) -> None:
        # Given: a capture-stable reverse map refreshed between requests.
        ids = torch.tensor([1, 5, 7], device="cuda", dtype=torch.int64)
        src = torch.arange(24, device="cuda", dtype=torch.float32).reshape(3, 2, 4)
        reverse_indices = prepare_kv_scatter_indices(ids, 9)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            triton_kv_scatter(src, ids, 9, reverse_indices=reverse_indices)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = triton_kv_scatter(src, ids, 9, reverse_indices=reverse_indices)
        graph.replay()
        torch.cuda.synchronize()
        ids.copy_(torch.tensor([0, 3, 8], device="cuda", dtype=torch.int64))
        reverse_indices.copy_(prepare_kv_scatter_indices(ids, 9))

        # When: the same graph consumes updated plan contents.
        graph.replay()

        # Then: it reflects new destinations and clears old ones.
        expected = torch.zeros_like(result)
        expected[ids] = src
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_independent_streams_keep_separate_maps(self) -> None:
        # Given: two independent requests with overlapping destination ranges.
        ids_a = torch.arange(0, 2050, 2, device="cuda", dtype=torch.int64)
        ids_b = torch.arange(1, 2050, 2, device="cuda", dtype=torch.int64)
        src = torch.arange(1025 * 62, device="cuda", dtype=torch.float32).reshape(
            1025, 2, 31
        )
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        stream_a.wait_stream(torch.cuda.current_stream())
        stream_b.wait_stream(torch.cuda.current_stream())

        # When: map preparation and scatter overlap on distinct streams.
        with torch.cuda.stream(stream_a):
            map_a = prepare_kv_scatter_indices(ids_a, 2050)
        with torch.cuda.stream(stream_b):
            map_b = prepare_kv_scatter_indices(ids_b, 2050)
        with torch.cuda.stream(stream_a):
            result_a = triton_kv_scatter(src, ids_a, 2050, reverse_indices=map_a)
        with torch.cuda.stream(stream_b):
            result_b = triton_kv_scatter(src, ids_b, 2050, reverse_indices=map_b)
        torch.cuda.current_stream().wait_stream(stream_a)
        torch.cuda.current_stream().wait_stream(stream_b)

        # Then: neither map contains rows from the other request.
        expected_a = torch.zeros_like(result_a)
        expected_b = torch.zeros_like(result_b)
        expected_a[ids_a] = src
        expected_b[ids_b] = src
        torch.testing.assert_close(result_a, expected_a, rtol=0, atol=0)
        torch.testing.assert_close(result_b, expected_b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
