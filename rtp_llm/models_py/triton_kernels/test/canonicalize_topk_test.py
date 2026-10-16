"""Exact integer canonicalization, including padding, duplicates and boundaries."""

import json
import os
import statistics
import time
import unittest
from pathlib import Path

import torch
from rtp_llm.models_py.triton_kernels.common.canonicalize_topk import (
    _canonicalize,
    canonicalize_topk_,
)


def reference(out):
    sentinel = torch.iinfo(out.dtype).max
    sortable = torch.where(out >= 0, out, torch.full_like(out, sentinel))
    sorted_indices = torch.sort(sortable, dim=-1).values
    out.copy_(
        torch.where(
            sorted_indices == sentinel,
            torch.full_like(sorted_indices, -1),
            sorted_indices,
        )
    )


class TestCanonicalizeTopk(unittest.TestCase):
    def check(self, out):
        expected = out.clone()
        reference(expected)
        pointer = out.data_ptr()
        self.assertTrue(canonicalize_topk_(out))
        self.assertEqual(pointer, out.data_ptr())
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_exact_shapes_boundaries_and_fallback(self):
        torch.manual_seed(5313)
        for dtype in (torch.int32, torch.int64):
            for width in (1, 7, 127, 512, 1024, 1537, 2048, 4096):
                for rows in (0, 1, 13, 81):
                    out = torch.randint(
                        -7, 100003, (rows, width), device="cuda", dtype=dtype
                    )
                    if rows:
                        out[0].fill_(-1)
                        out[-1, ::3] = torch.iinfo(dtype).max
                        out[-1, 1::3] = torch.iinfo(dtype).max - 1
                    self.check(out)
        unsupported = [
            torch.zeros(2, 4, dtype=torch.int32),
            torch.zeros(2, 4, device="cuda"),
            torch.zeros(4, 2, device="cuda", dtype=torch.int32).T,
        ]
        for out in unsupported:
            self.assertFalse(canonicalize_topk_(out))

    def test_warp_tuning_with_unsorted_inputs(self):
        records = []
        torch.manual_seed(331)
        for tokens in (81, 419, 1021, 4093, 9889, 16381, 32749):
            source = torch.randint(
                -5, 100003, (tokens, 2048), device="cuda", dtype=torch.int32
            )
            expected = source.clone()
            reference(expected)
            out = source.clone()
            row = {"tokens": tokens}
            for warps in (0, 2, 4, 8, 16):

                def candidate():
                    if warps:
                        _canonicalize[(tokens,)](
                            out, 2048, 2147483647, 2048, num_warps=warps
                        )
                    else:
                        reference(out)

                def restored():
                    out.copy_(source)
                    candidate()

                restored()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                for _ in range(20):
                    restored()
                cpu, stream = [], []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    before = time.perf_counter()
                    for _ in range(100):
                        restored()
                    cpu.append((time.perf_counter() - before) * 1e4)
                    end.record()
                    end.synchronize()
                    stream.append(start.elapsed_time(end) * 10)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(10):
                        restored()
                graph.replay()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA]
                ) as profile:
                    graph.replay()
                    torch.cuda.synchronize()
                events = [
                    e
                    for e in profile.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
                row["eager" if not warps else f"warps{warps}"] = {
                    "cpu_us_including_restore": statistics.median(cpu),
                    "stream_us_including_restore": statistics.median(stream),
                    "kernel_sum_us_including_restore": sum(
                        e.device_time_total for e in events
                    )
                    / 10,
                    "launches_including_restore": len(events) / 10,
                }
            records.append(row)
            Path(
                os.environ["TEST_UNDECLARED_OUTPUTS_DIR"],
                "canonicalize_warp_tuning.json",
            ).write_text(json.dumps(records, indent=2))

    def test_glm53_width_tuning(self):
        records = []
        for width in (512, 1024):
            for tokens in (81, 419, 4093, 9889, 16381, 32749):
                source = torch.randint(
                    -5, 100003, (tokens, width), device="cuda", dtype=torch.int32
                )
                expected = source.clone()
                reference(expected)
                out = source.clone()
                row = {"tokens": tokens, "width": width}
                for warps in (0, 2, 4, 8, 16):

                    def restored():
                        out.copy_(source)
                        if warps:
                            _canonicalize[(tokens,)](
                                out, width, 2147483647, width, num_warps=warps
                            )
                        else:
                            reference(out)

                    restored()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    for _ in range(20):
                        restored()
                    cpu = []
                    for _ in range(5):
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        for _ in range(100):
                            restored()
                        cpu.append((time.perf_counter() - before) * 1e4)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(10):
                            restored()
                    graph.replay()
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    gpu = []
                    for _ in range(5):
                        start.record()
                        for _ in range(20):
                            graph.replay()
                        end.record()
                        end.synchronize()
                        gpu.append(start.elapsed_time(end) * 5)
                    row["eager" if not warps else f"warps{warps}"] = {
                        "cpu_us_including_restore": statistics.median(cpu),
                        "graph_us_including_restore": statistics.median(gpu),
                    }
                records.append(row)
                Path(
                    os.environ["TEST_UNDECLARED_OUTPUTS_DIR"],
                    "canonicalize_glm53_width_tuning.json",
                ).write_text(json.dumps(records, indent=2))

    def test_async_and_submission(self):
        rows = []
        for tokens in (1, 81, 419, 1021, 4093):
            source = torch.randint(
                -5, 100003, (tokens, 2048), device="cuda", dtype=torch.int32
            )
            self.check(source.clone())
            row = {"tokens": tokens, "width": 2048}
            for name, function in (("eager", reference), ("fused", canonicalize_topk_)):
                out = source.clone()
                for _ in range(20):
                    function(out)
                cpu, stream = [], []
                for _ in range(5):
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    before = time.perf_counter()
                    for _ in range(100):
                        function(out)
                    cpu.append((time.perf_counter() - before) * 1e4)
                    end.record()
                    end.synchronize()
                    stream.append(start.elapsed_time(end) * 10)
                row[name] = {
                    "cpu_us": statistics.median(cpu),
                    "stream_us": statistics.median(stream),
                }
            rows.append(row)
        torch.cuda.synchronize()
        torch.cuda._sleep(300_000_000)
        pending = torch.cuda.Event()
        pending.record()
        canonicalize_topk_(out)
        self.assertFalse(pending.query())
        torch.cuda.synchronize()
        Path(
            os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], "canonicalize_topk.json"
        ).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    unittest.main()
