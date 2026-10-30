"""Same-shape NCCL ordinary vs registered history KV, no model or fake routing."""

import json
import os
import socket
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port, output):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=180))
    from rtp_llm.models_py.distributed import pynccl_cp as cp

    cp._resolve_process_group = lambda group: group
    # Compare NCCL kernels only, not the Torch multicast dispatcher.
    cp._TORCH_SYMM_GATHER = False
    group = dist.group.WORLD
    device = torch.device("cuda", rank)
    stream = cp.warmup_kv_overlap(group)
    results = []
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)
    for rows in (544, 3114, 16384, 30000, 77248):
        width = 576
        with torch.cuda.stream(stream):
            ordinary = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
            ordinary_out = torch.empty(
                (rows * 4, width), dtype=ordinary.dtype, device=device
            )
            registered = cp.registered_kv_overlap_buffers(
                rows, width, ordinary.dtype, device, group
            )
            assert registered is not None, "registered NCCL windows unavailable"
            pairs = {"ordinary": (ordinary, ordinary_out), "registered": registered}
            pattern = (
                torch.arange(rows * width, device=device)
                .reshape(rows, width)
                .remainder(113)
            )
            for x, y in pairs.values():
                x.copy_((pattern + rank).to(x.dtype))
                x[-1].zero_()
                for _ in range(5):
                    cp.all_gather_kv_overlap(x, group, output=y)
            stream.synchronize()
            for name, (x, y) in pairs.items():
                for source in range(4):
                    expected = (pattern + source).to(x.dtype)
                    expected[-1].zero_()
                    assert torch.equal(
                        y[source * rows : (source + 1) * rows], expected
                    ), (rank, rows, name)
            timings = {k: [] for k in pairs}
            for repeat in range(9):
                order = list(pairs) if repeat % 2 == 0 else list(reversed(pairs))
                for name in order:
                    x, y = pairs[name]
                    dist.barrier(group=control)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record(stream)
                    for _ in range(32):
                        cp.all_gather_kv_overlap(x, group, output=y)
                    end.record(stream)
                    end.synchronize()
                    timings[name].append(start.elapsed_time(end) * 1000 / 32)
            for name, (x, y) in pairs.items():
                dist.barrier(group=control)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    for _ in range(3):
                        cp.all_gather_kv_overlap(x, group, output=y)
                    stream.synchronize()
                prof.export_chrome_trace(str(outdir / f"{name}-{rows}-rank{rank}.json"))
            results.append(
                dict(
                    rows=rows,
                    width=width,
                    dtype=str(ordinary.dtype),
                    bitwise=True,
                    per_rank_us=timings,
                    median_us={k: statistics.median(v) for k, v in timings.items()},
                )
            )
            del ordinary, ordinary_out, pairs, pattern
    (outdir / f"metrics-rank{rank}.json").write_text(json.dumps(results, indent=2))
    (outdir / f"environment-rank{rank}.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),
                gpu=torch.cuda.get_device_name(rank),
                physical=os.environ.get("CUDA_VISIBLE_DEVICES"),
                NCCL_ALGO=os.environ.get("NCCL_ALGO"),
                NCCL_PROTO=os.environ.get("NCCL_PROTO"),
            ),
            indent=2,
        )
    )
    torch.cuda.synchronize()
    dist.barrier(group=control)
    cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    assert os.environ.get(
        "TEST_UNDECLARED_OUTPUTS_DIR"
    ), "explicit output directory required"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        worker,
        args=(port, os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]),
        nprocs=4,
        join=True,
    )
