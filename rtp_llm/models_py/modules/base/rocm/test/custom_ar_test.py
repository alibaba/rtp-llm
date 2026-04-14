# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Test suite for aiter Custom AllReduce on ROCm.

import os
import sys
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore", message="barrier.*using the device under current context"
)
warnings.filterwarnings("ignore", message="Guessing device ID based on global rank")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def _setup_distributed(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )


def _cleanup_distributed():
    try:
        dist.barrier()
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Native reference
# ---------------------------------------------------------------------------
def _native_allreduce(tensor: torch.Tensor, group) -> torch.Tensor:
    result = tensor.clone()
    dist.all_reduce(result, group=group)
    return result


# ---------------------------------------------------------------------------
# Worker: custom allreduce correctness
# ---------------------------------------------------------------------------
def _worker_custom_ar(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
            aiter_custom_allreduce,
            ensure_aiter_ar_initialized,
            should_aiter_custom_ar,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_aiter_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        assert initialized, "aiter Custom AR failed to initialize"

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        allreduce_in_ref = allreduce_in.clone()

        eligible = should_aiter_custom_ar(allreduce_in)
        assert eligible, (
            f"Tensor not eligible for custom AR: "
            f"size={allreduce_in.numel() * allreduce_in.element_size()} bytes"
        )

        # Custom AR kernel
        custom_out = aiter_custom_allreduce(allreduce_in)

        # NCCL reference
        ref_out = _native_allreduce(allreduce_in_ref, dist.group.WORLD)

        # Correctness check
        diff = (custom_out - ref_out).abs().max().item()
        ref_max = ref_out.abs().max().item()
        relative_diff = diff / ref_max if ref_max > 0 else 0

        rtol, atol = 1e-2, 1e-3
        assert relative_diff < rtol or diff < atol, (
            f"[Rank {rank}] custom AR mismatch: "
            f"rel={relative_diff:.6e}, abs={diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [custom_ar] batch={batch_size} hidden={hidden_size} "
                f"dtype={dtype} rel_diff={relative_diff:.2e} \u2713"
            )

    except Exception as exc:
        print(f"[Rank {rank}] custom_ar FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: custom allreduce eligibility boundary
# ---------------------------------------------------------------------------
def _worker_custom_ar_boundary(
    rank: int,
    world_size: int,
    port: int,
):
    """Verify eligibility checks around the 64MB boundary."""
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
            ensure_aiter_ar_initialized,
            should_aiter_custom_ar,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_aiter_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        assert initialized, "aiter Custom AR failed to initialize"

        # Just below 64MB limit: should be eligible
        elements_below = (64 * 1024 * 1024 // 2) - 16  # in bf16 elements
        tensor_below = torch.randn(elements_below, dtype=torch.bfloat16, device=device)
        eligible_below = should_aiter_custom_ar(tensor_below)

        # Just above 64MB limit: should NOT be eligible
        elements_above = (64 * 1024 * 1024 // 2) + 16
        tensor_above = torch.randn(elements_above, dtype=torch.bfloat16, device=device)
        eligible_above = should_aiter_custom_ar(tensor_above)

        # Misaligned size: should NOT be eligible
        tensor_misaligned = torch.randn(7, dtype=torch.bfloat16, device=device)
        eligible_misaligned = should_aiter_custom_ar(tensor_misaligned)

        if rank == 0:
            print(f"  [boundary] below_64MB={eligible_below} (expect True)")
            print(f"  [boundary] above_64MB={eligible_above} (expect False)")
            print(f"  [boundary] misaligned={eligible_misaligned} (expect False)")

        assert eligible_below, "Tensor below 64MB should be eligible"
        assert not eligible_above, "Tensor above 64MB should NOT be eligible"
        assert not eligible_misaligned, "Misaligned tensor should NOT be eligible"

        if rank == 0:
            print("  [custom_ar_boundary] all checks passed \u2713")

    except Exception as exc:
        print(f"[Rank {rank}] custom_ar_boundary FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Port allocation helper
# ---------------------------------------------------------------------------
_next_port = 29600


def _get_free_port() -> int:
    global _next_port
    port = _next_port
    _next_port += 1
    return port


# ---------------------------------------------------------------------------
# Process launcher
# ---------------------------------------------------------------------------
def _launch_workers(worker_fn, world_size: int, timeout: int = 120, **kwargs):
    port = _get_free_port()
    processes = []
    for rank in range(world_size):
        proc = mp.Process(
            target=worker_fn,
            args=(rank, world_size, port),
            kwargs=kwargs,
            name=f"rank-{rank}",
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join(timeout=timeout)
        if proc.exitcode != 0:
            raise RuntimeError(f"Process {proc.name} exited with code {proc.exitcode}")


# ===========================================================================
# Test cases
# ===========================================================================
class TestCustomAllReduce(unittest.TestCase):
    """End-to-end correctness tests for aiter Custom AllReduce."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need >= 2 GPUs, found {gpu_count}")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # BF16 correctness
    # ------------------------------------------------------------------
    def test_custom_ar_bf16_small_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    def test_custom_ar_bf16_medium_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=64,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    def test_custom_ar_bf16_large_batch_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=256,
            hidden_size=4096,
            dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # FP16 correctness
    # ------------------------------------------------------------------
    def test_custom_ar_fp16_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=8,
            hidden_size=4096,
            dtype=torch.float16,
        )

    # ------------------------------------------------------------------
    # Different hidden sizes
    # ------------------------------------------------------------------
    def test_custom_ar_hidden2048_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=16,
            hidden_size=2048,
            dtype=torch.bfloat16,
        )

    def test_custom_ar_hidden5120_ws2(self):
        _launch_workers(
            _worker_custom_ar,
            world_size=2,
            batch_size=16,
            hidden_size=5120,
            dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Boundary / eligibility checks
    # ------------------------------------------------------------------
    def test_custom_ar_boundary_ws2(self):
        _launch_workers(
            _worker_custom_ar_boundary,
            world_size=2,
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("ENABLE_AITER_CUSTOM_AR", "1")
    unittest.main()
