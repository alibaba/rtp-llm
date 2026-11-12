import os
import json
import time
from pathlib import Path
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.shuffle import expert_model_parallel_shuffle_inplace_intranode


def _worker(rank, world, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    logical = 256
    redundant = 16
    local = (logical + redundant) // world
    layers = int(os.getenv("EPLB_TEST_LAYERS", "1"))
    generator = torch.Generator().manual_seed(184)
    origin = torch.cat((torch.arange(logical), torch.arange(redundant))).repeat(
        layers, 1
    )
    weights = [
        [
            torch.empty((local, *shape), dtype=dtype, device="cuda")
            for shape, dtype in (
                ((4096, 3072), torch.int8),
                ((48, 4096), torch.int32),
                ((6144, 1024), torch.int8),
                ((16, 6144), torch.int32),
            )
        ]
        for _ in range(layers)
    ]
    addresses = [[w.data_ptr() for w in row] for row in weights]
    from rtp_llm.eplb.topology import legacy_nvlink_domain

    compatible, topology = legacy_nvlink_domain(
        dist.group.WORLD, torch.device("cuda", rank)
    )
    assert compatible, topology
    if rank == 0:
        print("verified IPC topology: " + json.dumps(topology), flush=True)
    import torch.distributed._symmetric_memory as symm_mem

    mirrors = [
        symm_mem.empty(w.shape, dtype=w.dtype, device=w.device) for w in weights[0]
    ]
    handles = [symm_mem.rendezvous(w, dist.group.WORLD) for w in mirrors]
    ipc_buffers = [
        [h.get_buffer(peer, w.shape, w.dtype) for h, w in zip(handles, mirrors)]
        for peer in range(world)
    ]
    for mirror in mirrors:
        mirror.view(torch.uint8).fill_(rank)
    torch.cuda.synchronize()
    dist.barrier()
    for peer, buffers in enumerate(ipc_buffers):
        for buffer in buffers:
            assert torch.all(buffer.view(torch.uint8) == peer).item()
    dist.barrier()
    timings = []
    for layer in range(layers):
        for slot, expert in enumerate(origin[layer, rank * local : (rank + 1) * local]):
            for index, weight in enumerate(weights[layer]):
                weight[slot].view(torch.uint8).fill_(
                    (int(expert) + index + layer) % 256
                )
    # Initialize the communicator before symmetric-memory collectives.
    dist.barrier()
    signal = torch.zeros(1, device="cuda")
    for _ in range(3):
        dist.all_reduce(signal)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        signal.fill_(rank)
        dist.all_reduce(signal)
    for iteration in range(int(os.getenv("EPLB_TEST_UPDATES", "100"))):
        graph.replay()
        rows = []
        for _ in range(layers):
            row = torch.cat(
                (
                    torch.arange(logical),
                    torch.randint(logical, (redundant,), generator=generator),
                )
            )
            rows.append(row[torch.randperm(world * local, generator=generator)])
        target = torch.stack(rows)
        for layer in range(layers):
            for slot, expert in enumerate(
                origin[layer, rank * local : (rank + 1) * local]
            ):
                for index, weight in enumerate(weights[layer]):
                    weight[slot].view(torch.uint8).fill_(
                        (int(expert) + index + layer) % 256
                    )
        old_gpu, new_gpu = origin.cuda(), target.cuda()
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        expert_model_parallel_shuffle_inplace_intranode(
            ipc_buffers, old_gpu, weights, new_gpu, dist.group.WORLD
        )
        torch.cuda.synchronize()
        timings.append(
            {
                "rank": rank,
                "iteration": iteration,
                "transport": "ipc",
                "seconds": time.perf_counter() - started,
                "layers": layers,
            }
        )
        assert addresses == [[w.data_ptr() for w in row] for row in weights]
        for layer in range(layers):
            for slot, expert in enumerate(
                target[layer, rank * local : (rank + 1) * local]
            ):
                for index, weight in enumerate(weights[layer]):
                    assert torch.all(
                        weight[slot].view(torch.uint8)
                        == (int(expert) + index + layer) % 256
                    ).item(), ("ipc", rank, iteration, layer, slot, index)
        assert signal.item() == world * (world - 1) // 2
        for layer in range(layers):
            for slot, expert in enumerate(
                target[layer, rank * local : (rank + 1) * local]
            ):
                for index, weight in enumerate(weights[layer]):
                    assert torch.all(
                        weight[slot].view(torch.uint8)
                        == (int(expert) + index + layer) % 256
                    ).item(), (rank, iteration, layer, slot, index)
        origin = target
        if rank == 0:
            print(f"shuffle iteration {iteration} verified", flush=True)
    graph.reset()
    output = Path(os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir()))
    (output / f"transport_rank{rank}.json").write_text(json.dumps(timings, indent=2))
    dist.destroy_process_group()


class ShuffleLargeTest(unittest.TestCase):
    def test_glm5_weight_transport(self):
        world = int(os.getenv("EPLB_TEST_WORLD_SIZE", "4"))
        self.assertGreaterEqual(torch.cuda.device_count(), world)
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(world, f"file://{directory}/rdzv"), nprocs=world)


if __name__ == "__main__":
    unittest.main()
