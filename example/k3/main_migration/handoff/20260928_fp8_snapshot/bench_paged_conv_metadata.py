#!/usr/bin/env python3
"""Measure warmed host planning plus device transfer for paged convolution."""

import json
import statistics
import time

import torch

from rtp_llm.models_py.triton_kernels.causal_conv1d.paged_short_conv_prefill import (
    prepare_paged_short_conv_metadata,
)


def stable(recent):
    median = statistics.median(recent)
    return all(abs(value / median - 1) <= 0.05 for value in recent)


def measure(lengths):
    cu_host = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    device = torch.device("cuda:0")
    samples = []

    def one():
        torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        meta = prepare_paged_short_conv_metadata(cu_host, device)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        assert meta.total_chunks == sum((length + 63) // 64 for length in lengths)
        return elapsed

    one()  # CUDA initialization and lazy runtime state
    warmups = []
    for _ in range(60):
        warmups.append(one())
        if len(warmups) >= 10 and stable(warmups[-3:]):
            break
    else:
        raise RuntimeError("metadata warmup did not converge")
    for _ in range(100):
        samples.append(one())
    return {
        "lengths": lengths,
        "warmup_count": len(warmups),
        "warmup_last3_ms": warmups[-3:],
        "sample_count": len(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "single": measure([8192]),
        "multi": measure([4096, 4096]),
    }, indent=2))
