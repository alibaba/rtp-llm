"""Opt-in GPU memory checkpoints for Kimi K3 startup diagnostics."""

import logging
import os

import torch


def log_k3_gpu_memory(stage: str) -> None:
    """Log allocator and device usage without changing allocator state.

    The device-wide figure includes CUDA allocations outside PyTorch and any
    other processes on the GPU, so it is not a per-process attribution.
    """
    if os.environ.get("K3_GPU_MEMORY_DEBUG") != "1":
        return
    try:
        if not torch.cuda.is_initialized():
            logging.info(
                "[K3_GPU_MEM] stage=%s pid=%d cuda_not_initialized", stage, os.getpid()
            )
            return

        device = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        allocated_bytes = torch.cuda.memory_allocated(device)
        reserved_bytes = torch.cuda.memory_reserved(device)
        stats = torch.cuda.memory_stats(device)
        mib = 1024 * 1024
        logging.info(
            "[K3_GPU_MEM] stage=%s pid=%d device=%d "
            "total_mib=%.1f free_mib=%.1f driver_used_mib=%.1f "
            "torch_allocated_mib=%.1f torch_reserved_mib=%.1f "
            "torch_cached_mib=%.1f inactive_split_mib=%.1f "
            "driver_used_minus_torch_reserved_mib=%.1f",
            stage,
            os.getpid(),
            device,
            total_bytes / mib,
            free_bytes / mib,
            (total_bytes - free_bytes) / mib,
            allocated_bytes / mib,
            reserved_bytes / mib,
            (reserved_bytes - allocated_bytes) / mib,
            stats.get("inactive_split_bytes.all.current", 0) / mib,
            (total_bytes - free_bytes - reserved_bytes) / mib,
        )
    except Exception:
        # Diagnostic logging must never prevent the model from starting.
        logging.exception("[K3_GPU_MEM] stage=%s snapshot_failed", stage)
