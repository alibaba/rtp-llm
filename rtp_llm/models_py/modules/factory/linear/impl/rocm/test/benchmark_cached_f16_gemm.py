"""Standalone warm eager microbenchmark; run on an isolated ROCm test worker.

No profiler, graph capture, service requests, or runtime configuration writes.
Reports submission and end-to-end per-call times, NOT pure GPU kernel time.
"""

import argparse
import json
import statistics
import time

import torch
from aiter import hipb_create_extension, hipb_mm

from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.swizzle_utils import swizzle_tensor


def measure(fn, iterations):
    for _ in range(10):
        fn()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            fn()
        submitted = time.perf_counter_ns()
        torch.cuda.synchronize()
        finished = time.perf_counter_ns()
        samples.append(
            (
                (submitted - start) / iterations / 1000,
                (finished - start) / iterations / 1000,
            )
        )
    return {
        "host_submit_us": statistics.median(s[0] for s in samples),
        "end_to_end_us": statistics.median(s[1] for s in samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU required")
    hipb_create_extension()
    with torch.inference_mode():
        for k, n in ((2560, 6144), (4096, 2560), (2560, 19456), (9728, 2560)):
            weight = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.01
            weight = swizzle_tensor(weight.t(), False, MiM=16).t()
            for q in (1, 2, 3, 4, 14, 15, 16):
                x = torch.randn(q, k, device="cuda", dtype=weight.dtype)
                old = lambda: hipb_mm(x, weight, -1, bpreshuffle=True)
                new = lambda: rtp_llm_ops.rocm_cached_f16_gemm(x, weight, None, True)
                actual = new()
                if actual is None:
                    raise RuntimeError("benchmark unexpectedly fell back")
                torch.testing.assert_close(actual, old(), atol=0, rtol=0)
                # Run both orders to expose clock/order bias rather than hide it.
                for order in ("old-new", "new-old"):
                    fns = {"old": old, "new": new}
                    timings = {
                        name: measure(fns[name], args.iterations)
                        for name in order.split("-")
                    }
                    print(
                        json.dumps(
                            {
                                "q": q,
                                "k": k,
                                "n": n,
                                "order": order,
                                **timings,
                                "cache": rtp_llm_ops.rocm_f16_gemm_cache_stats(),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
