# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Test suite for aiter Quick AllReduce on ROCm.

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
# Worker: quick allreduce correctness
# ---------------------------------------------------------------------------
def _worker_quick_ar(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_level: str,
):
    try:
        os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = quant_level
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
            quick_allreduce,
            should_quick_allreduce,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_quick_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        assert (
            initialized
        ), f"Quick AR failed to initialize with quant_level={quant_level}"

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        allreduce_in_ref = allreduce_in.clone()

        eligible = should_quick_allreduce(allreduce_in)
        if not eligible:
            if rank == 0:
                tensor_bytes = allreduce_in.numel() * allreduce_in.element_size()
                print(
                    f"  [quick_ar] batch={batch_size} hidden={hidden_size} "
                    f"dtype={dtype} quant={quant_level} "
                    f"size={tensor_bytes} bytes SKIPPED (not eligible)"
                )
            return

        # Quick AR kernel
        quick_out = quick_allreduce(allreduce_in)

        # NCCL reference
        ref_out = _native_allreduce(allreduce_in_ref, dist.group.WORLD)

        # Correctness check
        diff = (quick_out.float() - ref_out.float()).abs().max().item()
        ref_max = ref_out.float().abs().max().item()
        relative_diff = diff / ref_max if ref_max > 0 else 0

        # Quick AR uses quantization, so tolerances are looser
        if quant_level == "FP":
            rtol, atol = 1e-2, 1e-3
        elif quant_level == "FP8":
            rtol, atol = 5e-2, 1e-2
        elif quant_level == "INT6":
            rtol, atol = 0.1, 0.05
        else:  # INT4
            rtol, atol = 0.2, 0.1

        assert relative_diff < rtol or diff < atol, (
            f"[Rank {rank}] quick AR mismatch (quant={quant_level}): "
            f"rel={relative_diff:.6e}, abs={diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [quick_ar] batch={batch_size} hidden={hidden_size} "
                f"dtype={dtype} quant={quant_level} "
                f"rel_diff={relative_diff:.2e} \u2713"
            )

    except Exception as exc:
        print(f"[Rank {rank}] quick_ar FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: quick allreduce precision comparison across quant levels
# ---------------------------------------------------------------------------
def _worker_quick_ar_precision_comparison(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_level: str,
):
    """Measure precision loss at different quantization levels."""
    try:
        os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = quant_level
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
            quick_allreduce,
            should_quick_allreduce,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_quick_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        if not initialized:
            if rank == 0:
                print(f"  [precision] quant={quant_level} SKIPPED (init failed)")
            return

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        allreduce_in_ref = allreduce_in.clone()

        if not should_quick_allreduce(allreduce_in):
            if rank == 0:
                tensor_bytes = allreduce_in.numel() * allreduce_in.element_size()
                print(
                    f"  [precision] quant={quant_level} "
                    f"size={tensor_bytes} bytes SKIPPED (not eligible)"
                )
            return

        quick_out = quick_allreduce(allreduce_in)
        ref_out = _native_allreduce(allreduce_in_ref, dist.group.WORLD)

        max_diff = (quick_out.float() - ref_out.float()).abs().max().item()
        mean_diff = (quick_out.float() - ref_out.float()).abs().mean().item()
        ref_max = ref_out.float().abs().max().item()
        relative_max = max_diff / ref_max if ref_max > 0 else 0
        relative_mean = mean_diff / ref_max if ref_max > 0 else 0

        if rank == 0:
            print(
                f"  [precision] quant={quant_level} dtype={dtype} "
                f"max_rel={relative_max:.4e} mean_rel={relative_mean:.4e} "
                f"max_abs={max_diff:.4e}"
            )

    except Exception as exc:
        print(
            f"[Rank {rank}] precision_comparison FAILED: {exc}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: quick allreduce eligibility checks
# ---------------------------------------------------------------------------
def _worker_quick_ar_eligibility(
    rank: int,
    world_size: int,
    port: int,
    quant_level: str,
):
    """Verify eligibility checks for various tensor sizes and dtypes."""
    try:
        os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = quant_level
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
            should_quick_allreduce,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_quick_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        if not initialized:
            if rank == 0:
                print(f"  [eligibility] quant={quant_level} SKIPPED (init failed)")
            return

        # FP32 should NOT be eligible (only FP16/BF16 supported)
        tensor_fp32 = torch.randn(1024, dtype=torch.float32, device=device)
        assert not should_quick_allreduce(
            tensor_fp32
        ), "FP32 tensor should NOT be eligible for quick AR"

        # Misaligned size should NOT be eligible
        tensor_misaligned = torch.randn(7, dtype=torch.bfloat16, device=device)
        assert not should_quick_allreduce(
            tensor_misaligned
        ), "Misaligned tensor should NOT be eligible"

        # Very small tensor (below min threshold) should NOT be eligible
        tensor_tiny = torch.randn(8, dtype=torch.bfloat16, device=device)
        eligible_tiny = should_quick_allreduce(tensor_tiny)
        # Min threshold for BF16+TP2+FP is 2MB, so 16 bytes is way below
        assert not eligible_tiny, "Tiny tensor should NOT be eligible"

        if rank == 0:
            print(
                f"  [eligibility] quant={quant_level} "
                f"fp32_rejected=True misaligned_rejected=True "
                f"tiny_rejected=True \u2713"
            )

    except Exception as exc:
        print(f"[Rank {rank}] eligibility FAILED: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: BF16 vs FP16 kernel precision comparison
# ---------------------------------------------------------------------------
def _worker_quick_ar_bf16_fp16_comparison(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
):
    """Compare precision when BF16 is cast to FP16 vs native BF16."""
    try:
        os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "FP"
        os.environ["AITER_QUICK_REDUCE_CAST_BF16_TO_FP16"] = "1"
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            ensure_quick_ar_initialized,
            quick_allreduce,
            should_quick_allreduce,
        )

        device = torch.device(f"cuda:{rank}")

        initialized = ensure_quick_ar_initialized(
            group=dist.group.WORLD, device_id=rank
        )
        if not initialized:
            if rank == 0:
                print("  [bf16_fp16] SKIPPED (init failed)")
            return

        torch.manual_seed(42 + rank)
        # Use BF16 input — the kernel will internally cast to FP16
        allreduce_in = torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        )
        allreduce_in_ref = allreduce_in.clone()

        if not should_quick_allreduce(allreduce_in):
            if rank == 0:
                print("  [bf16_fp16] SKIPPED (not eligible)")
            return

        quick_out = quick_allreduce(allreduce_in)
        ref_out = _native_allreduce(allreduce_in_ref, dist.group.WORLD)

        max_diff = (quick_out.float() - ref_out.float()).abs().max().item()
        ref_max = ref_out.float().abs().max().item()
        relative_diff = max_diff / ref_max if ref_max > 0 else 0

        # BF16→FP16 cast introduces precision loss
        rtol = 1e-2
        assert relative_diff < rtol, (
            f"[Rank {rank}] BF16→FP16 quick AR precision too low: "
            f"rel={relative_diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [bf16_fp16] batch={batch_size} hidden={hidden_size} "
                f"rel_diff={relative_diff:.2e} \u2713"
            )

    except Exception as exc:
        print(
            f"[Rank {rank}] bf16_fp16_comparison FAILED: {exc}",
            file=sys.stderr,
        )
        import traceback

        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Port allocation helper
# ---------------------------------------------------------------------------
_next_port = 29700


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
class TestQuickAllReduce(unittest.TestCase):
    """End-to-end correctness tests for aiter Quick AllReduce."""

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
    # FP mode (full precision, no quantization in communication)
    # ------------------------------------------------------------------
    def test_quick_ar_fp_bf16_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="FP",
        )

    def test_quick_ar_fp_fp16_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.float16,
            quant_level="FP",
        )

    # ------------------------------------------------------------------
    # FP8 quantized communication
    # ------------------------------------------------------------------
    def test_quick_ar_fp8_bf16_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="FP8",
        )

    # ------------------------------------------------------------------
    # INT6 quantized communication
    # ------------------------------------------------------------------
    def test_quick_ar_int6_bf16_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="INT6",
        )

    # ------------------------------------------------------------------
    # INT4 quantized communication
    # ------------------------------------------------------------------
    def test_quick_ar_int4_bf16_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="INT4",
        )

    # ------------------------------------------------------------------
    # Larger batch (more tokens, simulating prefill)
    # ------------------------------------------------------------------
    def test_quick_ar_fp_large_batch_ws2(self):
        _launch_workers(
            _worker_quick_ar,
            world_size=2,
            batch_size=2048,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="FP",
        )

    # ------------------------------------------------------------------
    # Precision comparison across quantization levels
    # ------------------------------------------------------------------
    def test_precision_fp_ws2(self):
        _launch_workers(
            _worker_quick_ar_precision_comparison,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="FP",
        )

    def test_precision_fp8_ws2(self):
        _launch_workers(
            _worker_quick_ar_precision_comparison,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="FP8",
        )

    def test_precision_int6_ws2(self):
        _launch_workers(
            _worker_quick_ar_precision_comparison,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="INT6",
        )

    def test_precision_int4_ws2(self):
        _launch_workers(
            _worker_quick_ar_precision_comparison,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
            dtype=torch.bfloat16,
            quant_level="INT4",
        )

    # ------------------------------------------------------------------
    # Eligibility checks
    # ------------------------------------------------------------------
    def test_eligibility_fp_ws2(self):
        _launch_workers(
            _worker_quick_ar_eligibility,
            world_size=2,
            quant_level="FP",
        )

    # ------------------------------------------------------------------
    # BF16 → FP16 kernel cast precision
    # ------------------------------------------------------------------
    def test_bf16_fp16_cast_precision_ws2(self):
        _launch_workers(
            _worker_quick_ar_bf16_fp16_comparison,
            world_size=2,
            batch_size=512,
            hidden_size=4096,
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("AITER_QUICK_REDUCE_QUANTIZATION", "FP")
    unittest.main()
