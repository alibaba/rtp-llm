"""Eager wrapper and kernel-sum timings on real MoE captures after exact checks."""

import json
import os
import statistics
import unittest
from collections import Counter
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_shared_act_quant_sm120 as shared_test,
)
from rtp_llm.models_py.modules.dsv4.moe._v41_shared_act_quant_triton import (
    shared_act_quant,
)
from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_return import V41ReturnLayout, pack_return


def percentile(samples, fraction):
    samples = sorted(samples)
    index = (len(samples) - 1) * fraction
    lower = int(index)
    return samples[lower] + (
        samples[min(lower + 1, len(samples) - 1)] - samples[lower]
    ) * (index - lower)


def measure(function):
    for _ in range(30):
        function()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for _ in range(100):
        start.record()
        function()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000)
    peak = torch.cuda.max_memory_allocated() - base
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(10):
            function()
        torch.cuda.synchronize()
    kernels = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    counts = Counter(event.name for event in kernels)
    return {
        "event_us": {
            "median": statistics.median(samples),
            "p90": percentile(samples, 0.9),
            "min": min(samples),
            "max": max(samples),
            "samples": samples,
        },
        "kernel_sum_us_per_iter": sum(event.device_time_total for event in kernels)
        / 10,
        "launches_per_iter": len(kernels) / 10,
        "kernel_names_per_iter": {name: count / 10 for name, count in counts.items()},
        "peak_temporary_bytes": peak,
    }


def exact(actual, expected):
    if isinstance(actual, tuple):
        for a, b in zip(actual, expected):
            exact(a, b)
    else:
        if actual.dtype == torch.float8_e4m3fn:
            actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class V41FusionBenchmark(unittest.TestCase):
    def test_real_wrappers(self):
        directory = Path(os.environ["DSV41_MOE_FIXTURES"])
        paths = sorted(directory.glob("rank*-layer*-chunk*.pt"))
        self.assertEqual(len(paths), 8)
        report = {
            "warmups": 30,
            "event_samples": 100,
            "profile_samples": 10,
            "compile_timed": False,
            "input_allocation_timed": False,
            "mode": "eager; warm cache; wrapper allocations included",
            "cases": [],
        }
        for path in paths:
            tensors = torch.load(path, weights_only=True, map_location="cpu")["tensors"]
            gate_up = tensors["shared_gate_up"].cuda()
            partial = tensors["local_partial"].cuda()
            x = tensors["moe_input"].cuda()
            layout = V41DispatchLayout(5120, 6, 384)
            packet = tensors["wire0_send"][: x.shape[0]].cuda().contiguous()
            _, _, weights, ids = layout.views(packet)
            weights, ids = weights.contiguous(), ids.contiguous()
            valid = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            ret = V41ReturnLayout(5120, "mxfp8")
            cases = [
                (
                    "shared",
                    lambda: shared_test.reference(gate_up, 10.0),
                    lambda: shared_act_quant(gate_up, clamp_limit=10.0),
                ),
                (
                    "dispatch",
                    lambda: pack_dispatch(x, weights, ids, valid, layout),
                    lambda: pack_dispatch(x, weights, ids, valid, layout, fused=True),
                ),
                (
                    "return",
                    lambda: pack_return(partial, ret),
                    lambda: pack_return(partial, ret, fused=True),
                ),
            ]
            for name, baseline, candidate in cases:
                exact(candidate(), baseline())
                samples = {
                    "baseline_before": measure(baseline),
                    "candidate": measure(candidate),
                    "baseline_after": measure(baseline),
                }
                names = samples["candidate"]["kernel_names_per_iter"]
                target = {
                    "shared": "_shared_act_quant",
                    "dispatch": "_dispatch",
                    "return": "_return",
                }[name]
                matches = [count for kernel, count in names.items() if target in kernel]
                self.assertEqual(sum(matches), 1)
                if name != "dispatch":
                    self.assertEqual(samples["candidate"]["launches_per_iter"], 1)
                report["cases"].append(
                    {
                        "fixture": path.name,
                        "boundary": name,
                        "exact": True,
                        "candidate_kernel_verified": True,
                        **samples,
                    }
                )
        output = Path(os.environ["DSV41_BENCHMARK_OUTPUT"])
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
