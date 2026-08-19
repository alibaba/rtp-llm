from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules import GroupTopK


DEFAULT_TOKEN_COUNTS = [1, 7, 17, 64, 128, 257, 1024, 4096, 6954, 8192, 16384]
PROFILE_TOKEN_COUNTS = {1, 128, 1024, 6954, 16384}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _event_metrics(
    function: Callable[[], None], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": statistics.median(samples_us),
        "p90_us": _percentile(samples_us, 0.90),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _kernel_metrics(
    function: Callable[[], None], warmup: int, iterations: int
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for _ in range(iterations):
            function()
            profiler.step()
    torch.cuda.synchronize()

    kernel_time_us = 0.0
    kernel_names: Counter[str] = Counter()
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        kernel_time_us += float(event.device_time_total)
        kernel_names[event.name] += 1
    return {
        "kernel_sum_us_per_iter": kernel_time_us / iterations,
        "launches_per_iter": sum(kernel_names.values()) / iterations,
        "kernel_names": dict(kernel_names),
    }


def _token_counts() -> list[int]:
    value = os.environ.get("M_LIST", "")
    if not value:
        return DEFAULT_TOKEN_COUNTS
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    warmup = int(os.environ.get("WARMUP", "30"))
    iterations = int(os.environ.get("ITERS", "100"))
    profile_iterations = int(os.environ.get("PROFILE_ITERS", "20"))
    torch.manual_seed(20260814)

    group_topk = GroupTopK(use_fused=True)
    results = []
    for tokens in _token_counts():
        logits = torch.randn((tokens, 256), dtype=torch.bfloat16, device="cuda")
        correction_bias = torch.randn((256,), dtype=torch.float32, device="cuda")
        outputs = {}
        functions = {}
        for candidate, implementation in (
            ("baseline", "forward_legacy"),
            ("fused", "forward_fused"),
        ):
            values = torch.empty((tokens, 8), dtype=torch.float32, device="cuda")
            indices = torch.empty((tokens, 8), dtype=torch.int64, device="cuda")

            def run(
                implementation: str = implementation,
                values: torch.Tensor = values,
                indices: torch.Tensor = indices,
            ) -> None:
                getattr(group_topk, implementation)(
                    values,
                    indices,
                    logits,
                    correction_bias,
                    1,
                    1,
                    8,
                    True,
                    2.5,
                )

            run()
            functions[candidate] = run
            outputs[candidate] = (values.clone(), indices.clone())

        torch.testing.assert_close(
            outputs["fused"][0], outputs["baseline"][0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            outputs["fused"][1], outputs["baseline"][1], rtol=0, atol=0
        )

        row = {"tokens": tokens}
        for candidate, function in functions.items():
            metrics: dict[str, object] = _event_metrics(
                function, warmup, iterations
            )
            if tokens in PROFILE_TOKEN_COUNTS:
                metrics.update(
                    _kernel_metrics(function, warmup, profile_iterations)
                )
            row[candidate] = metrics
        row["event_speedup"] = (
            row["baseline"]["median_us"] / row["fused"]["median_us"]
        )
        if tokens in PROFILE_TOKEN_COUNTS:
            row["kernel_speedup"] = (
                row["baseline"]["kernel_sum_us_per_iter"]
                / row["fused"]["kernel_sum_us_per_iter"]
            )
        results.append(row)

    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(),
            "dtype": "BF16",
            "shape": "[T,256]",
            "n_group": 1,
            "topk_group": 1,
            "topk": 8,
            "warmup": warmup,
            "iterations": iterations,
            "profile_iterations": profile_iterations,
            "input_output_allocations_timed": False,
            "implementation_internal_allocations_in_event_timing": True,
            "torch_compile": False,
        },
        "results": results,
    }
    print("GROUP_TOPK_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
