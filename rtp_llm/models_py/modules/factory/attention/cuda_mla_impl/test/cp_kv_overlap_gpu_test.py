"""Eight-rank NCCL ordering, allocator lifetime and concurrent-stream coverage."""

import json
import os
import socket
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, port):
    from rtp_llm.models_py.distributed import pynccl_cp

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    # Production may set a default CUDA device; the communicator ID must still
    # be created on CPU before ctypes accesses it.
    torch.set_default_device(f"cuda:{rank}")
    group = dist.group.WORLD
    producer = pynccl_cp.warmup_kv_overlap(group)
    for _ in range(100):
        assert pynccl_cp.warmup_kv_overlap(group) is producer
    compute = torch.cuda.current_stream()
    results = []
    for rows in (1, 17, 1025, 16384, 194304):
        x = torch.full((rows, 576), rank + 1, dtype=torch.bfloat16)
        row_ids = torch.arange(rows, dtype=torch.int64)
        # Encode every row exactly in BF16, so a wrong restore permutation
        # cannot pass merely because rows from one rank contain equal values.
        for column, divisor in ((1, 1), (2, 256), (3, 65536)):
            x[:, column] = (row_ids // divisor) % 256
        reference = torch.empty((8 * rows, 576), dtype=x.dtype)
        dist.all_gather_into_tensor(reference, x, group=group)
        order = torch.arange(8 * rows - 1, -1, -1, dtype=torch.int64)
        # All tensors consumed by compute are allocated there, as in production.
        restored = torch.empty_like(reference)
        probe = torch.full((17, 32), rank, dtype=torch.float32)
        probe_out = torch.empty((8 * 17, 32), dtype=probe.dtype)
        torch.cuda.synchronize()
        for iteration in range(3):
            gathered = torch.empty_like(reference)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            producer.wait_stream(compute)
            with torch.cuda.stream(producer):
                start.record()
                pynccl_cp.all_gather_kv_overlap(x, group, output=gathered)
                torch.index_select(gathered, 0, order, out=restored)
                end.record()
            # Interleave a second communicator, matching Indexer collectives.
            dist.all_gather_into_tensor(probe_out, probe, group=group)
            work_start, work_end = (
                torch.cuda.Event(enable_timing=True) for _ in range(2)
            )
            work_start.record()
            torch.cuda._sleep(5_000_000)
            work_end.record()
            compute.wait_event(end)
            del gathered
            result = restored.clone()
            # Force allocations after joining to expose premature buffer reuse.
            scratch = torch.empty_like(restored).fill_(-17)
            torch.cuda.synchronize()
            torch.testing.assert_close(result, reference[order], rtol=0, atol=0)
            torch.testing.assert_close(
                probe_out[:, 0],
                torch.arange(8, dtype=torch.float32).repeat_interleave(17),
                rtol=0,
                atol=0,
            )
            overlap = max(
                0.0,
                min(start.elapsed_time(end), start.elapsed_time(work_end))
                - max(0.0, start.elapsed_time(work_start)),
            )
            results.append(
                {
                    "rows": rows,
                    "iteration": iteration,
                    "gather_restore_ms": start.elapsed_time(end),
                    "compute_overlap_ms": overlap,
                }
            )
            del result, scratch
        del reference, restored, x
    if not any(row["compute_overlap_ms"] > 0 for row in results):
        raise AssertionError("no producer/compute overlap observed")
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
    (output / f"kv_overlap_rank{rank}.json").write_text(json.dumps(results, indent=2))
    torch.cuda.synchronize()
    dist.barrier()
    pynccl_cp.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.device_count() < 8:
        raise RuntimeError("this test requires eight allocated GPUs")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=8, join=True)
