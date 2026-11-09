"""Matched Pool IPC benchmark. Does not measure ViT, H2D, or full inference."""

import argparse
import gc
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
from memfd_transport_test import initialize_worker, video_payload

CASES = {
    "1MiB": (1, 512, 512, 4),
    "64MiB": (16, 1024, 1024, 4),
    "video80_1062x1418": (80, 1062, 1418, 3),
    "video92_2160x3840": (92, 2160, 3840, 3),
}


def memory_bytes():
    for name in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        if os.path.isfile(name):
            return int(Path(name).read_text())
    return None


def cpu_stat():
    for name in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"):
        if os.path.isfile(name):
            return {
                key: int(value)
                for key, value in (
                    line.split() for line in Path(name).read_text().splitlines()
                )
            }
    return {}


class MemorySampler:
    def __enter__(self):
        self.start = self.peak = memory_bytes()
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(0.02):
                value = memory_bytes()
                if value is not None:
                    self.peak = max(self.peak or 0, value)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def transfer_batch(pool, mode, shape, concurrency, first_id):
    jobs = [
        pool.apply_async(video_payload, (mode, shape, first_id + i))
        for i in range(concurrency)
    ]
    rows = []
    for job in jobs:
        tensor, row = job.get(120)
        obtained_ns = time.monotonic_ns()
        begin = time.monotonic_ns()
        # NumPy accumulates uint8 into uint64 without materializing an int64
        # tensor eight times larger than the payload, unlike torch.sum here.
        actual_sum = int(tensor.numpy().sum(dtype="uint64"))
        read_done_ns = time.monotonic_ns()
        assert actual_sum == row["expected_sum"], row
        assert row["first"] == 251 and tensor[(-1,) * tensor.ndim].item() == 253
        assert tensor.dtype == torch.uint8
        assert tuple(tensor.shape) == row["shape"] and tensor.stride() == row["stride"]
        row.update(
            ok=True,
            boundary_ms=(row["ready_ns"] - row["producer_ready_ns"]) / 1e6,
            sender_queue_ms=(row["export_start_ns"] - row["producer_ready_ns"]) / 1e6,
            export_ms=(row["export_end_ns"] - row["export_start_ns"]) / 1e6,
            ipc_queue_ms=(row["receiver_start_ns"] - row["export_end_ns"]) / 1e6,
            receiver_ms=(row["ready_ns"] - row["receiver_start_ns"]) / 1e6,
            ready_to_get_ms=(obtained_ns - row["ready_ns"]) / 1e6,
            full_read_ms=(read_done_ns - begin) / 1e6,
            producer_to_full_read_ms=(read_done_ns - row["producer_ready_ns"]) / 1e6,
        )
        assert row["ipc_queue_ms"] >= 0, "monotonic clock ordering invalid"
        rows.append(row)
        del tensor
    del job, jobs
    return rows


def benchmark(args):
    torch.set_num_threads(1)
    context = mp.get_context("spawn")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    worker_cpus = allowed_cpus[2:6] if len(allowed_cpus) >= 6 else None
    os.sched_setaffinity(0, allowed_cpus[:2])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        scope="synthetic CPU tensor ready -> Pool consumer CPU tensor usable",
        excludes=["download", "decode", "GPU H2D", "ViT", "LLM"],
        runtime=sys.version,
        torch=torch.__version__,
        kernel=platform.release(),
        machine=platform.machine(),
        cpus=allowed_cpus,
        worker_cpus=worker_cpus,
        torch_threads=1,
        startup_gate="all workers crossed initialization barrier",
        samples_per_repetition=args.samples,
        repetitions=3,
        shm_capacity_bytes=os.statvfs("/dev/shm").f_blocks
        * os.statvfs("/dev/shm").f_frsize,
        baseline="Pool normal torch Tensor reducer (file_descriptor), instrumented envelope",
        candidate="same Pool + copy existing entire storage to memfd + FD broker",
        workload="fresh NHWC uint8 storage, noncontiguous TCHW view; no handle reuse",
        aggregation="median complete repetition selected by boundary P50; all repeats retained",
    )
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    summaries = []
    request_id = 0
    with (output / "samples.jsonl").open("w") as raw:
        for concurrency in args.concurrency:
            ready = context.Barrier(concurrency + 1)
            with context.Pool(
                concurrency,
                initializer=initialize_worker,
                initargs=(1, worker_cpus, ready),
            ) as pool:
                ready.wait(timeout=60)
                for case in args.cases:
                    shape = CASES[case]
                    for mode in ("torch", "memfd"):
                        for _ in range(2):
                            transfer_batch(pool, mode, shape, concurrency, request_id)
                            request_id += concurrency
                    for repetition in range(3):
                        paired_request_id = request_id
                        modes = (
                            ("torch", "memfd")
                            if repetition % 2 == 0
                            else ("memfd", "torch")
                        )
                        for mode in modes:
                            request_id = paired_request_id
                            gc.collect()
                            rows = []
                            cpu_before = cpu_stat()
                            with MemorySampler() as memory:
                                started = time.monotonic()
                                for _ in range(0, args.samples, concurrency):
                                    batch = transfer_batch(
                                        pool, mode, shape, concurrency, request_id
                                    )
                                    request_id += concurrency
                                    rows.extend(batch)
                                seconds = time.monotonic() - started
                            cpu_after = cpu_stat()
                            for row in rows:
                                row.update(
                                    case=case,
                                    concurrency=concurrency,
                                    repetition=repetition,
                                )
                                raw.write(json.dumps(row) + "\n")
                            raw.flush()
                            latencies = [row["boundary_ms"] for row in rows]
                            summary = dict(
                                case=case,
                                mode=mode,
                                concurrency=concurrency,
                                repetition=repetition,
                                samples=len(rows),
                                bytes=rows[0]["logical_bytes"],
                                p50_ms=percentile(latencies, 0.5),
                                p95_ms=percentile(latencies, 0.95),
                                mean_ms=statistics.mean(latencies),
                                full_read_p50_ms=percentile(
                                    [r["full_read_ms"] for r in rows], 0.5
                                ),
                                producer_to_full_read_p50_ms=percentile(
                                    [r["producer_to_full_read_ms"] for r in rows], 0.5
                                ),
                                stage_means_ms={
                                    key: statistics.mean(r[key] for r in rows)
                                    for key in (
                                        "sender_queue_ms",
                                        "export_ms",
                                        "ipc_queue_ms",
                                        "receiver_ms",
                                    )
                                },
                                delivered_gib_per_s=sum(
                                    r["logical_bytes"] for r in rows
                                )
                                / seconds
                                / 2**30,
                                repetition_seconds=seconds,
                                memory_start_bytes=memory.start,
                                memory_peak_bytes=memory.peak,
                                cpu_stat_delta={
                                    key: value - cpu_before.get(key, 0)
                                    for key, value in cpu_after.items()
                                },
                            )
                            summaries.append(summary)
                            (output / "summary.json").write_text(
                                json.dumps(summaries, indent=2)
                            )
                            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    options = parser.parse_args()
    if any(options.samples <= 0 or options.samples % c for c in options.concurrency):
        parser.error("samples must be positive and divisible by each concurrency")
    benchmark(options)
