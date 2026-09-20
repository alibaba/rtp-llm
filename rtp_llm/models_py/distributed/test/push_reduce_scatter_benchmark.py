"""Measure the RTP push reduce-scatter against NCCL with GPU graph events.

Requires an RTP build exporting push_reduce_scatter and eight NVLink GPUs.
python -m torch.distributed.run --standalone --nproc-per-node=8 \
    rtp_llm/models_py/distributed/test/push_reduce_scatter_benchmark.py --output /tmp/push-rs-results.json
"""

import argparse
import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.push_reduce_scatter import create_push_reduce_scatter


def measure(fn, control):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=control)
    start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
    sync = torch.ones(1, device="cuda")
    dist.all_reduce(sync)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dist.all_reduce(sync)
        start.record()
        for _ in range(64):
            fn()
        end.record()
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        dist.barrier(group=control)
        elapsed = 0
        for _ in range(5):
            graph.replay()
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end)
        samples.append(elapsed * 1000 / (5 * 64))
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, samples, group=control)
    worst = [max(values[i] for values in ranks) for i in range(7)]
    return {
        "median_us": statistics.median(worst),
        "min_us": min(worst),
        "max_us": max(worst),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", default="8,128,504,512,1024,8192")
    args = parser.parse_args()
    sizes = [int(m) for m in args.sizes.split(",")]
    if not sizes or any(m <= 0 or m % 8 for m in sizes):
        parser.error("--sizes must contain positive multiples of 8")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=3))
    group = dist.group.WORLD
    control = dist.new_group(backend="gloo")
    assert group.size() == 8
    push = create_push_reduce_scatter(group, device, max_m=max(sizes), n=7168)
    assert push is not None, "Push RS is unavailable on this topology/device"
    results = []
    for m in sizes:
        x = torch.randn(m, 7168, dtype=torch.bfloat16, device=device)
        out = torch.empty(m // 8, 7168, dtype=x.dtype, device=device)
        # Both backends use the production operator with preallocated output.
        row = {"m": m}
        row["nccl"] = measure(
            lambda: dist.reduce_scatter_tensor(out, x, group=group), control
        )
        row["rtp_push"] = measure(lambda: push.reduce_scatter(x, out), control)
        row["speedup"] = row["nccl"]["median_us"] / row["rtp_push"]["median_us"]
        results.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "device": props.name,
                    "capability": [props.major, props.minor],
                    "nccl": torch.cuda.nccl.version(),
                    "tp": 8,
                    "hidden": 7168,
                    "dtype": "BF16",
                    "max_m": max(sizes),
                    "results": results,
                    "timing": "median of maximum-rank CUDA graph event times; 64 ops, 5 replays, 7 repeats; repeated hot input",
                },
                indent=2,
            )
            + "\n"
        )
    dist.destroy_process_group(control)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
