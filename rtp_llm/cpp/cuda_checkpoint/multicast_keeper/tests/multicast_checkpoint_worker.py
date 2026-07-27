#!/usr/bin/env python3

import gc
import os
import signal
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
base_port = int(os.environ["MASTER_PORT"])
buffer_bytes = int(os.environ.get("RTP_MC_TEST_BUFFER_BYTES", str(64 * 1024 * 1024)))
device = torch.device(f"cuda:{local_rank}")

torch.cuda.set_device(device)

group = None
buffer = None
handle = None
baseline = None
teardown_requested = False
rebuild_requested = False
exit_requested = False


def init_generation(port: int) -> None:
    global group, buffer, handle
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    buffer = symm_mem.empty(
        buffer_bytes // torch.tensor([], dtype=torch.bfloat16).element_size(),
        dtype=torch.bfloat16,
        device=device,
    )
    handle = symm_mem.rendezvous(buffer, group=group.group_name)
    torch.cuda.synchronize(device)


def run_collectives():
    value = torch.tensor([rank + 1.0], dtype=torch.float32, device=device)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize(device)
    return value.cpu()


def clear_symmetric_memory_state() -> None:
    for cache_name in (
        "_group_name_to_workspace_tensor",
        "_backend_streams",
        "_symm_mem_pools",
    ):
        cache = getattr(symm_mem, cache_name, None)
        if isinstance(cache, dict):
            cache.clear()


def teardown_generation() -> None:
    global group, buffer, handle
    handle = None
    buffer = None
    clear_symmetric_memory_state()
    gc.collect()
    torch.cuda.synchronize(device)
    dist.destroy_process_group(group)
    group = None
    gc.collect()
    torch.cuda.synchronize(device)


def request_teardown(*_args) -> None:
    global teardown_requested
    teardown_requested = True


def request_rebuild(*_args) -> None:
    global rebuild_requested
    rebuild_requested = True


def request_exit(*_args) -> None:
    global exit_requested
    exit_requested = True


signal.signal(signal.SIGUSR1, request_teardown)
signal.signal(signal.SIGUSR2, request_rebuild)
signal.signal(signal.SIGTERM, request_exit)
signal.signal(signal.SIGINT, request_exit)

init_generation(base_port)
baseline = run_collectives()
multicast_ptr = int(getattr(handle, "multicast_ptr", 0))
print(
    f"READY rank={rank} pid={os.getpid()} multicast_ptr=0x{multicast_ptr:x} "
    f"value={baseline.tolist()}",
    flush=True,
)

while not exit_requested:
    if teardown_requested:
        teardown_requested = False
        teardown_generation()
        print(f"TORNDOWN rank={rank} pid={os.getpid()}", flush=True)
    if rebuild_requested:
        rebuild_requested = False
        init_generation(base_port + 100)
        result = run_collectives()
        rebuilt_ptr = int(getattr(handle, "multicast_ptr", 0))
        equal = torch.equal(baseline, result)
        print(
            f"RESULT rank={rank} pid={os.getpid()} equal={equal} "
            f"multicast_ptr=0x{rebuilt_ptr:x} value={result.tolist()}",
            flush=True,
        )
    time.sleep(0.05)

if group is not None:
    teardown_generation()
print(f"EXIT rank={rank} pid={os.getpid()}", flush=True)
