from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from typing import Callable

import torch
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.models_py.triton_kernels.common.strided_slice_copy import (
    strided_slice_copy_,
)


HEADS = 64
NOPE = 192
ROPE = 64
KV_LORA = 512
Q_OUT = KV_LORA + ROPE
TOKEN_COUNTS = [1, 17, 128, 257, 1024, 4096, 6954, 8192, 16384]
PROFILE_TOKEN_COUNTS = {1, 128, 1024, 6954, 16384}


def _cos_sin_cache(device: torch.device) -> torch.Tensor:
    inv = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE, 2, device=device, dtype=torch.float32)
            / ROPE
        )
    )
    positions = torch.arange(16384.0, device=device)
    return torch.cat(
        [torch.outer(positions, inv).cos(), torch.outer(positions, inv).sin()],
        dim=-1,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _event_metrics(
    function: Callable[[], None], warmup: int, iterations: int
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return {
        "median_us": statistics.median(samples),
        "p90_us": _percentile(samples, 0.90),
        "min_us": min(samples),
        "max_us": max(samples),
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
    names: Counter[str] = Counter()
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        kernel_time_us += float(event.device_time_total)
        names[event.name] += 1
    return {
        "kernel_sum_us_per_iter": kernel_time_us / iterations,
        "launches_per_iter": sum(names.values()) / iterations,
        "kernel_names": dict(names),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    warmup = int(os.environ.get("WARMUP", "30"))
    iterations = int(os.environ.get("ITERS", "100"))
    profile_iterations = int(os.environ.get("PROFILE_ITERS", "20"))
    raw_tokens = os.environ.get("M_LIST", "")
    token_counts = (
        [int(item) for item in raw_tokens.split(",") if item.strip()]
        if raw_tokens
        else TOKEN_COUNTS
    )
    device = torch.device("cuda")
    cache = _cos_sin_cache(device)
    rope_op = NewMlaRotaryEmbeddingOp(cache, is_neox_style=False)
    torch.manual_seed(20260814)
    results = []

    for tokens in token_counts:
        positions = torch.randint(
            0, 16383, (tokens,), dtype=torch.int32, device=device
        )
        q_baseline = torch.randn(
            (tokens, HEADS, ROPE), dtype=torch.bfloat16, device=device
        )
        q_direct = q_baseline.clone()
        k_baseline = torch.randn(
            (tokens, ROPE), dtype=torch.bfloat16, device=device
        )
        k_direct = k_baseline.clone()
        output_baseline = torch.empty(
            (tokens, HEADS, Q_OUT), dtype=torch.bfloat16, device=device
        )
        output_direct = torch.empty_like(output_baseline)

        def baseline() -> None:
            rope_op.forward(
                q_baseline,
                k_baseline,
                None,
                precomputed_pos_ids=positions,
            )
            strided_slice_copy_(output_baseline, q_baseline, KV_LORA)

        def direct() -> None:
            rope_op.forward(
                q_direct,
                k_direct,
                None,
                precomputed_pos_ids=positions,
                q_rope_output=output_direct[..., KV_LORA:],
            )

        row: dict[str, object] = {"tokens": tokens}
        for name, function in (("baseline", baseline), ("direct", direct)):
            metrics: dict[str, object] = _event_metrics(
                function, warmup, iterations
            )
            if tokens in PROFILE_TOKEN_COUNTS:
                metrics.update(
                    _kernel_metrics(function, warmup, profile_iterations)
                )
            row[name] = metrics
        row["event_speedup"] = (
            row["baseline"]["median_us"] / row["direct"]["median_us"]
        )
        if tokens in PROFILE_TOKEN_COUNTS:
            row["kernel_speedup"] = (
                row["baseline"]["kernel_sum_us_per_iter"]
                / row["direct"]["kernel_sum_us_per_iter"]
            )
        results.append(row)

    payload = {
        "meta": {
            "device": torch.cuda.get_device_name(),
            "dtype": "BF16",
            "q_shape": "[T,64,64]",
            "output_shape": "[T,64,576]",
            "is_neox_style": False,
            "warmup": warmup,
            "iterations": iterations,
            "profile_iterations": profile_iterations,
            "input_output_allocations_timed": False,
            "torch_compile": False,
        },
        "results": results,
    }
    print("Q_ROPE_DIRECT_WRITE_BENCHMARK_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
