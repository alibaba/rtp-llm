# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Test suite for TRT-LLM AllReduce Fusion kernels on ROCm.

import os
import sys
import unittest
import warnings
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore", message="barrier.*using the device under current context"
)
warnings.filterwarnings(
    "ignore", message="Guessing device ID based on global rank"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP8_DTYPE = torch.float8_e4m3fnuz
FP8_MAX_VALUE = 120.0
FP8_QUANT_TYPE_ID = 2  # e4m3fnuz


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
# Native reference implementations
# ---------------------------------------------------------------------------
def _native_allreduce(
    allreduce_in: torch.Tensor,
    group,
) -> torch.Tensor:
    """Reference allreduce using NCCL."""
    result = allreduce_in.clone()
    dist.all_reduce(result, group=group)
    return result


def _native_allreduce_residual_rmsnorm(
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    group,
    fp8_out: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference allreduce + residual add + rmsnorm."""
    allreduce_result = allreduce_in.clone()
    dist.all_reduce(allreduce_result, group=group)

    residual_out = allreduce_result + residual_in

    input_dtype = residual_out.dtype
    variance = residual_out.float().pow(2).mean(-1, keepdim=True)
    norm_out = residual_out * torch.rsqrt(variance + eps)
    norm_out = norm_out.to(input_dtype)
    norm_out = rms_weight * norm_out

    if fp8_out:
        norm_out_scale, _ = norm_out.float().abs().max(dim=-1, keepdim=True)
        norm_out_scale = norm_out_scale / FP8_MAX_VALUE
        norm_out = norm_out / norm_out_scale
        norm_out.clamp_(min=-FP8_MAX_VALUE, max=FP8_MAX_VALUE)
        norm_out = norm_out.to(FP8_DTYPE)
        return residual_out, norm_out, norm_out_scale
    else:
        scale_out = torch.empty(
            allreduce_in.shape[0], 1,
            dtype=torch.float32,
            device=allreduce_in.device,
        )
        return residual_out, norm_out, scale_out


# ---------------------------------------------------------------------------
# Worker: allreduce + residual + rmsnorm (fused vs native)
# ---------------------------------------------------------------------------
def _worker_allreduce_residual_rmsnorm(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    eps: float,
    fp8_out: bool,
    dtype: torch.dtype,
):
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.distributed.trt_allreduce import TrtllmDistEnv

        device = torch.device(f"cuda:{rank}")
        dist_env = TrtllmDistEnv(
            group=dist.group.WORLD, device_id=rank, dtype=dtype,
        )

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(
            batch_size, hidden_size, dtype=dtype, device=device
        )
        residual_in = torch.randn(
            batch_size, hidden_size, dtype=dtype, device=device
        )
        rms_weight = torch.randn(hidden_size, dtype=dtype, device=device)

        allreduce_in_ref = allreduce_in.clone()

        # Fused kernel
        residual_out, norm_out, scale_out = dist_env.allreduce_add_rms_fused(
            allreduce_in, residual_in, rms_weight, eps, fp8_out=fp8_out,
        )

        # Native reference
        ref_residual, ref_norm, ref_scale = _native_allreduce_residual_rmsnorm(
            allreduce_in_ref, residual_in, rms_weight, eps,
            dist.group.WORLD, fp8_out=fp8_out,
        )

        # --- shape / dtype checks ---
        assert residual_out.shape == (batch_size, hidden_size), (
            f"residual_out shape mismatch: {residual_out.shape}"
        )
        assert norm_out.shape == (batch_size, hidden_size), (
            f"norm_out shape mismatch: {norm_out.shape}"
        )
        if fp8_out:
            assert norm_out.dtype == FP8_DTYPE, (
                f"Expected FP8 dtype, got {norm_out.dtype}"
            )
            assert scale_out.shape == (batch_size, 1), (
                f"scale_out shape mismatch for FP8: {scale_out.shape}"
            )

        # --- NaN / Inf checks ---
        assert not torch.isnan(residual_out).any(), "residual_out contains NaN"
        assert not torch.isinf(residual_out).any(), "residual_out contains Inf"

        # --- numerical correctness ---
        residual_diff = (residual_out - ref_residual).abs().max().item()
        residual_max = ref_residual.abs().max().item()
        residual_rel = residual_diff / residual_max if residual_max > 0 else 0

        if fp8_out:
            fused_fp32 = norm_out.float() * scale_out
            ref_fp32 = ref_norm.float() * ref_scale
            norm_diff = (fused_fp32 - ref_fp32).abs().max().item()
            norm_max = ref_fp32.abs().max().item()
            rtol, atol = 0.1, 0.05
        else:
            norm_diff = (norm_out - ref_norm).abs().max().item()
            norm_max = ref_norm.abs().max().item()
            rtol, atol = 1e-2, 1e-3

        norm_rel = norm_diff / norm_max if norm_max > 0 else 0

        assert residual_rel < rtol or residual_diff < atol, (
            f"[Rank {rank}] residual mismatch: rel={residual_rel:.6e}, "
            f"abs={residual_diff:.6e}"
        )
        assert norm_rel < rtol or norm_diff < atol, (
            f"[Rank {rank}] norm mismatch: rel={norm_rel:.6e}, "
            f"abs={norm_diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [fused_rmsnorm] hidden={hidden_size} fp8={fp8_out} "
                f"residual_rel={residual_rel:.2e} norm_rel={norm_rel:.2e} ✓"
            )

    except Exception as exc:
        print(
            f"[Rank {rank}] allreduce_residual_rmsnorm FAILED: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: pure allreduce
# ---------------------------------------------------------------------------
def _worker_pure_allreduce(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.distributed.trt_allreduce import TrtllmDistEnv

        device = torch.device(f"cuda:{rank}")
        dist_env = TrtllmDistEnv(
            group=dist.group.WORLD, device_id=rank, dtype=dtype,
        )

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(
            batch_size, hidden_size, dtype=dtype, device=device
        )
        allreduce_in_ref = allreduce_in.clone()
        allreduce_out = torch.empty_like(allreduce_in)

        # Fused kernel
        dist_env.allreduce_op(allreduce_in, allreduce_out)

        # Native reference
        ref_out = _native_allreduce(allreduce_in_ref, dist.group.WORLD)

        # --- correctness ---
        diff = (allreduce_out - ref_out).abs().max().item()
        ref_max = ref_out.abs().max().item()
        rel_diff = diff / ref_max if ref_max > 0 else 0

        rtol, atol = 1e-2, 1e-3
        assert rel_diff < rtol or diff < atol, (
            f"[Rank {rank}] allreduce mismatch: rel={rel_diff:.6e}, "
            f"abs={diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [pure_allreduce] hidden={hidden_size} "
                f"rel_diff={rel_diff:.2e} ✓"
            )

    except Exception as exc:
        print(
            f"[Rank {rank}] pure_allreduce FAILED: {exc}",
            file=sys.stderr,
        )
        import traceback
        traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: fused vs native method consistency
# ---------------------------------------------------------------------------
def _worker_fused_vs_native(
    rank: int,
    world_size: int,
    port: int,
    batch_size: int,
    hidden_size: int,
    eps: float,
    fp8_out: bool,
    dtype: torch.dtype,
):
    """Compare TrtllmDistEnv.allreduce_add_rms_fused with allreduce_add_rms_native."""
    try:
        _setup_distributed(rank, world_size, port)
        from rtp_llm.models_py.distributed.trt_allreduce import TrtllmDistEnv

        device = torch.device(f"cuda:{rank}")
        dist_env = TrtllmDistEnv(
            group=dist.group.WORLD, device_id=rank, dtype=dtype,
        )

        torch.manual_seed(42 + rank)
        allreduce_in = torch.randn(
            batch_size, hidden_size, dtype=dtype, device=device
        )
        residual_in = torch.randn(
            batch_size, hidden_size, dtype=dtype, device=device
        )
        rms_weight = torch.randn(hidden_size, dtype=dtype, device=device)

        allreduce_in_native = allreduce_in.clone()

        # Fused path
        fused_residual, fused_norm, fused_scale = dist_env.allreduce_add_rms_fused(
            allreduce_in, residual_in, rms_weight, eps, fp8_out=fp8_out,
        )

        # Native path (uses the same dist_env)
        native_residual, native_norm, native_scale = dist_env.allreduce_add_rms_native(
            allreduce_in_native, residual_in, rms_weight, eps, fp8_out=fp8_out,
        )

        # --- residual check ---
        residual_diff = (fused_residual - native_residual).abs().max().item()
        residual_max = native_residual.abs().max().item()
        residual_rel = residual_diff / residual_max if residual_max > 0 else 0

        # --- norm check ---
        if fp8_out:
            fused_fp32 = fused_norm.float() * fused_scale
            native_fp32 = native_norm.float() * native_scale
            norm_diff = (fused_fp32 - native_fp32).abs().max().item()
            norm_max = native_fp32.abs().max().item()
            rtol, atol = 0.1, 0.05
        else:
            norm_diff = (fused_norm - native_norm).abs().max().item()
            norm_max = native_norm.abs().max().item()
            rtol, atol = 1e-2, 1e-3

        norm_rel = norm_diff / norm_max if norm_max > 0 else 0

        assert residual_rel < rtol or residual_diff < atol, (
            f"[Rank {rank}] fused_vs_native residual mismatch: "
            f"rel={residual_rel:.6e}, abs={residual_diff:.6e}"
        )
        assert norm_rel < rtol or norm_diff < atol, (
            f"[Rank {rank}] fused_vs_native norm mismatch: "
            f"rel={norm_rel:.6e}, abs={norm_diff:.6e}"
        )

        if rank == 0:
            print(
                f"  [fused_vs_native] hidden={hidden_size} fp8={fp8_out} "
                f"residual_rel={residual_rel:.2e} norm_rel={norm_rel:.2e} ✓"
            )

    except Exception as exc:
        print(
            f"[Rank {rank}] fused_vs_native FAILED: {exc}",
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
_next_port = 29500


def _get_free_port() -> int:
    global _next_port
    port = _next_port
    _next_port += 1
    return port


# ---------------------------------------------------------------------------
# Process launcher
# ---------------------------------------------------------------------------
def _launch_workers(worker_fn, world_size: int, timeout: int = 120, **kwargs):
    """Spawn *world_size* processes running *worker_fn* and wait."""
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
            raise RuntimeError(
                f"Process {proc.name} exited with code {proc.exitcode}"
            )


# ===========================================================================
# Test cases
# ===========================================================================
class TestTrtAllReduceFusion(unittest.TestCase):
    """End-to-end correctness tests for TRT-LLM AllReduce Fusion kernels."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            self.skipTest(f"Need ≥2 GPUs, found {gpu_count}")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Pure allreduce
    # ------------------------------------------------------------------
    def test_pure_allreduce_hidden1024_ws2(self):
        _launch_workers(
            _worker_pure_allreduce, world_size=2,
            batch_size=8, hidden_size=1024, dtype=torch.bfloat16,
        )

    def test_pure_allreduce_hidden2048_ws2(self):
        _launch_workers(
            _worker_pure_allreduce, world_size=2,
            batch_size=8, hidden_size=2048, dtype=torch.bfloat16,
        )

    def test_pure_allreduce_hidden4096_ws2(self):
        _launch_workers(
            _worker_pure_allreduce, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
        )

    def test_pure_allreduce_hidden4096_ws4(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Need ≥4 GPUs")
        _launch_workers(
            _worker_pure_allreduce, world_size=4,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Fused allreduce + residual + rmsnorm (bf16, no fp8)
    # ------------------------------------------------------------------
    def test_fused_rmsnorm_hidden1024_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=1024, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    def test_fused_rmsnorm_hidden2048_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=2048, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    def test_fused_rmsnorm_hidden4096_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    def test_fused_rmsnorm_hidden4096_ws4(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Need ≥4 GPUs")
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=4,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Fused allreduce + residual + rmsnorm with FP8 quantized output
    # ------------------------------------------------------------------
    def test_fused_rmsnorm_fp8_hidden1024_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=1024, eps=1e-6,
            fp8_out=True, dtype=torch.bfloat16,
        )

    def test_fused_rmsnorm_fp8_hidden4096_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=True, dtype=torch.bfloat16,
        )

    def test_fused_rmsnorm_fp8_hidden4096_ws4(self):
        if torch.cuda.device_count() < 4:
            self.skipTest("Need ≥4 GPUs")
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=4,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=True, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Fused vs native (TrtllmDistEnv internal consistency)
    # ------------------------------------------------------------------
    def test_fused_vs_native_hidden2048_ws2(self):
        _launch_workers(
            _worker_fused_vs_native, world_size=2,
            batch_size=8, hidden_size=2048, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    def test_fused_vs_native_fp8_hidden4096_ws2(self):
        _launch_workers(
            _worker_fused_vs_native, world_size=2,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=True, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # Larger batch sizes
    # ------------------------------------------------------------------
    def test_fused_rmsnorm_large_batch_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=64, hidden_size=4096, eps=1e-6,
            fp8_out=False, dtype=torch.bfloat16,
        )

    def test_pure_allreduce_large_batch_ws2(self):
        _launch_workers(
            _worker_pure_allreduce, world_size=2,
            batch_size=64, hidden_size=4096, dtype=torch.bfloat16,
        )

    # ------------------------------------------------------------------
    # float16 dtype
    # ------------------------------------------------------------------
    def test_fused_rmsnorm_fp16_hidden4096_ws2(self):
        _launch_workers(
            _worker_allreduce_residual_rmsnorm, world_size=2,
            batch_size=8, hidden_size=4096, eps=1e-6,
            fp8_out=False, dtype=torch.float16,
        )


if __name__ == "__main__":
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    unittest.main()
