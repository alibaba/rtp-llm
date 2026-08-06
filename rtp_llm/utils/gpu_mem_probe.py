"""Lightweight GPU-memory milestone probe.

Logs a one-line, grep-able ``[InitMem]`` snapshot that pairs the *physical*
device residency (``torch.cuda.mem_get_info`` == cudaMemGetInfo) with torch's
caching-allocator view (reserved/allocated). The gap between them is exactly
the non-torch footprint (CUDA context + JIT cubins + NCCL comm buffers +
symmetric-memory + plugin/cuBLAS cudaMalloc workspaces) that torch does not
account for and that torch_memory_saver cannot pause.

Placed at init milestones (before/after NCCL, after engine create) this lets us
decompose the sleep residual into torch-managed (weights/KV, VMM-pausable) vs
non-torch (needs explicit free + wake recreate) buckets. Best-effort only:
never raises into the caller.
"""

import logging
import os

import torch

_MiB = 1024 * 1024
_mem_history_enabled = False


def _maybe_enable_mem_history() -> None:
    """Enable CUDA allocation backtrace recording once, if opted in via env.

    ``RTP_LLM_RECORD_MEM_HISTORY=1`` turns on ``torch.cuda.memory._record_memory_history``
    so every caching-allocator segment/block carries the Python stack that
    allocated it. The sleep-time ``[SleepReclaim]`` snapshot then attributes each
    private-MemPool segment to its real allocator (symmetric-memory vs weights-region
    vs workspace) instead of guessing from pool size. Diagnostic only (per-alloc
    stack capture has overhead); off by default.
    """
    global _mem_history_enabled
    if _mem_history_enabled or os.environ.get("RTP_LLM_RECORD_MEM_HISTORY", "0") != "1":
        return
    try:
        torch.cuda.memory._record_memory_history(max_entries=200000)
        _mem_history_enabled = True
        logging.info("[InitMem] CUDA allocation history recording ENABLED (diagnostic)")
    except Exception as e:  # best-effort; never fail the caller
        logging.warning("[InitMem] could not enable mem history (ignored): %s", e)


def log_gpu_mem(tag: str, device: object = None) -> None:
    """Log ``[InitMem][tag]`` with physical + torch-view device memory.

    physical = cudaMemGetInfo (true residency, what nvidia-smi shows).
    torch_reserved/torch_alloc = caching-allocator pool / live tensors.
    non_torch = physical_used - torch_reserved (context + JIT + NCCL + symm +
    plugin workspaces; the part TMS pause cannot reclaim).
    """
    try:
        if not torch.cuda.is_available():
            return
        _maybe_enable_mem_history()
        dev = device if device is not None else torch.cuda.current_device()
        free_b, total_b = torch.cuda.mem_get_info(dev)
        used_b = total_b - free_b
        reserved_b = torch.cuda.memory_reserved(dev)
        alloc_b = torch.cuda.memory_allocated(dev)
        non_torch_b = used_b - reserved_b
        logging.info(
            "[InitMem][%s] dev=%s phys_used=%dMiB phys_free=%dMiB "
            "torch_reserved=%dMiB torch_alloc=%dMiB non_torch=%dMiB",
            tag,
            str(dev),
            used_b // _MiB,
            free_b // _MiB,
            reserved_b // _MiB,
            alloc_b // _MiB,
            non_torch_b // _MiB,
        )
    except Exception as e:  # best-effort observability, never fail the caller
        logging.warning("[InitMem][%s] snapshot failed (ignored): %s", tag, e)
