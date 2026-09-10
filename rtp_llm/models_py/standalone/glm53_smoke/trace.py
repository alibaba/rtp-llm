"""Synchronize profiler startup and reject traces dominated by instrumentation."""

from contextlib import nullcontext
import json
from pathlib import Path

import torch
import torch.distributed as dist


def gpu_span_ms(path):
    events = json.loads(Path(path).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise ValueError("trace contains no CUDA kernels")
    return (
        max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)
    ) / 1000


def profile_block(fn, restore, phase, output, median_ms):
    # CUPTI can initialize lazily on the first replay, with very different
    # delays across ranks. Discard a complete profiled warmup, rendezvous on
    # CPU, and check the exported trace against independent event timing.
    group = dist.new_group(backend="gloo")
    attempts = []
    accepted = False
    try:
        for attempt in range(4):
            restore()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            context = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                )
                if dist.get_rank() == 0
                else nullcontext()
            )
            # Only the exported rank needs CUPTI. Profiling every rank makes
            # communication wait on other ranks' instrumentation overhead.
            with context as prof:
                dist.barrier(group=group)
                with torch.profiler.record_function(f"glm53_smoke.{phase}.four_layers"):
                    fn()
                torch.cuda.synchronize()
            status = [None]
            if dist.get_rank() == 0:
                path = output / f"{phase}_profile_attempt{attempt}.json"
                prof.export_chrome_trace(str(path))
                span = gpu_span_ms(path)
                accepted = attempt > 0 and 0.8 * median_ms <= span <= 1.2 * median_ms
                attempts.append(
                    dict(
                        attempt=attempt,
                        gpu_span_ms=span,
                        warmup=attempt == 0,
                        accepted=accepted,
                    )
                )
                if accepted:
                    path.rename(output / f"{phase}.json")
                status[0] = accepted
            dist.broadcast_object_list(status, src=0, group=group)
            if status[0]:
                accepted = True
                break
        if dist.get_rank() == 0:
            (output / "trace_review.json").write_text(
                json.dumps(
                    dict(
                        accepted=accepted,
                        event_median_ms=median_ms,
                        attempts=attempts,
                        criterion="post-warmup GPU span within 20% of independent event median",
                    ),
                    indent=2,
                )
            )
        if not accepted:
            raise AssertionError(
                "profiler distorted every trace; timing samples remain in performance_rankN.json"
            )
    finally:
        dist.destroy_process_group(group)
