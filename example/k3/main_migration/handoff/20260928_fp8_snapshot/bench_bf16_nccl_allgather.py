#!/usr/bin/env python3
"""Screen intra-node BF16 NCCL AllGather at K3's 64K TP8 input shape.

This is a task-local diagnostic. It does not replace a warmed PD trace.
Run it with torchrun on one exclusive eight-GPU development host.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist


def stable_tail(values):
    tail = values[-3:]
    middle = statistics.median(tail)
    return middle > 0 and all(abs(value - middle) <= middle * 0.05 for value in tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rows", type=int, default=8192)
    parser.add_argument("--width", type=int, default=7168)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--max-warmups", type=int, default=60)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.local_rows, args.width, args.samples) <= 0:
        parser.error("shape and sample count must be positive")
    if args.warmups < 10 or args.max_warmups < args.warmups:
        parser.error("at least 10 warmups and a finite maximum are required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 8:
        raise RuntimeError(f"K3 TP8 shape requires 8 ranks, got {world}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        source = torch.full(
            (args.local_rows, args.width), rank, dtype=torch.bfloat16, device=device
        )
        gathered = torch.empty(
            (args.local_rows * world, args.width),
            dtype=torch.bfloat16,
            device=device,
        )

        # Materialize communicator, allocation and kernels outside warmup.
        dist.all_gather_into_tensor(gathered, source)
        torch.cuda.synchronize(device)
        markers = (
            gathered.view(world, args.local_rows, args.width)[:, 0, 0]
            .float()
            .cpu()
            .tolist()
        )
        if markers != list(range(world)):
            raise AssertionError(f"AllGather rank markers differ: {markers}")

        def sample():
            before = torch.cuda.Event(enable_timing=True)
            after = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            before.record()
            dist.all_gather_into_tensor(gathered, source)
            after.record()
            after.synchronize()
            return before.elapsed_time(after), (time.perf_counter() - start) * 1000

        warmup_gpu = []
        warmed = False
        while len(warmup_gpu) < args.max_warmups:
            for _ in range(min(10, args.max_warmups - len(warmup_gpu))):
                warmup_gpu.append(sample()[0])
            if len(warmup_gpu) < args.warmups:
                continue
            local_stable = int(stable_tail(warmup_gpu))
            ready = torch.tensor(local_stable, dtype=torch.int32, device=device)
            dist.all_reduce(ready, op=dist.ReduceOp.MIN)
            if ready.item():
                warmed = True
                break
        if not warmed:
            raise RuntimeError("BF16 NCCL AllGather warmup did not converge")

        dist.barrier(device_ids=[local_rank])
        torch.cuda.synchronize(device)
        measured = [sample() for _ in range(args.samples)]
        all_records = [None] * world
        dist.all_gather_object(
            all_records,
            {
                "rank": rank,
                "warmup_gpu_ms": warmup_gpu,
                "gpu_ms": [row[0] for row in measured],
                "wall_ms": [row[1] for row in measured],
            },
        )
        if rank == 0:
            ordered = sorted(all_records, key=lambda item: item["rank"])
            gpu_max = [max(row["gpu_ms"][i] for row in ordered) for i in range(args.samples)]
            wall_max = [max(row["wall_ms"][i] for row in ordered) for i in range(args.samples)]
            result = {
                "purpose": "intra-node BF16 NCCL AllGather screening, not PD performance",
                "dtype": "torch.bfloat16",
                "backend": "nccl",
                "world_size": world,
                "shape_per_rank": [args.local_rows, args.width],
                "shape_gathered": [args.local_rows * world, args.width],
                "gpu_name": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": torch.cuda.nccl.version(),
                "nccl_env": {
                    key: os.environ.get(key)
                    for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")
                },
                "warmup_count": len(warmup_gpu),
                "all_rank_warmup_stable": True,
                "samples_per_rank": args.samples,
                "gpu_max_rank_ms": gpu_max,
                "gpu_max_rank_median_ms": statistics.median(gpu_max),
                "wall_max_rank_ms": wall_max,
                "wall_max_rank_median_ms": statistics.median(wall_max),
                "ranks": ordered,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({
                "gpu_max_rank_median_ms": result["gpu_max_rank_median_ms"],
                "wall_max_rank_median_ms": result["wall_max_rank_median_ms"],
                "warmup_count": result["warmup_count"],
                "output": str(args.output),
            }))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
