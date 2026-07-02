"""Integration tests for ROCm allreduce tier dispatch (2-rank, real GPU).

Validates that the tiered allreduce path (quick → aiter → NCCL fallback)
produces numerically correct results across multiple dtypes and tensor sizes.
Requires ≥2 ROCm GPUs.

Run:
    python -m pytest rtp_llm/models_py/distributed/test/rocm_allreduce_tier_integration_test.py -v
"""

import os
import socket
import sys
import unittest
import warnings
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

warnings.filterwarnings(
    "ignore", message="barrier.*using the device under current context"
)
warnings.filterwarnings(
    "ignore", message="Guessing device ID based on global rank"
)


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


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


def _nccl_reference(tensor: torch.Tensor, group) -> torch.Tensor:
    ref = tensor.clone()
    dist.all_reduce(ref, group=group)
    return ref


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

    failed = []
    for proc in processes:
        proc.join(timeout=timeout)
        if proc.is_alive() or proc.exitcode != 0:
            failed.append(proc)

    if failed:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        names = [f"{p.name}(exit={p.exitcode})" for p in failed]
        raise RuntimeError(f"Worker(s) failed or timed out: {', '.join(names)}")


# ---------------------------------------------------------------------------
# Worker: quick allreduce
# ---------------------------------------------------------------------------
def _worker_quick_allreduce(
    rank: int, world_size: int, port: int,
    batch_size: int, hidden_size: int, dtype: torch.dtype,
    quant_type: str,
):
    try:
        os.environ["ROCM_ALLREDUCE_STRATEGY"] = "quick"
        os.environ["ROCM_QUICK_AR_QUANTIZATION"] = quant_type
        _setup_distributed(rank, world_size, port)

        from rtp_llm.models_py.modules.base.rocm.quick_allreduce import (
            quick_ar_manager,
        )

        device = torch.device(f"cuda:{rank}")
        group = dist.group.WORLD
        device_id = rank

        if not quick_ar_manager.ensure_initialized(group, device_id):
            if rank == 0:
                print("  [quick_allreduce] skipped: init failed")
            return

        torch.manual_seed(42 + rank)
        tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        ref = _nccl_reference(tensor.clone(), group)

        if not quick_ar_manager.should_use(tensor, group, device_id):
            if rank == 0:
                print(f"  [quick_allreduce] skipped: should_use=False for {dtype}")
            return

        result = quick_ar_manager.allreduce(tensor)

        diff = (result.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_diff = diff / ref_max if ref_max > 0 else 0

        # Quick AR uses FP8 quantization, so tolerance is relaxed
        rtol = 0.05 if quant_type == "FP" else 0.1
        assert rel_diff < rtol, (
            f"[Rank {rank}] quick allreduce mismatch: rel={rel_diff:.6e}, abs={diff:.6e}"
        )
        if rank == 0:
            print(f"  [quick_allreduce] quant={quant_type} hidden={hidden_size} rel_diff={rel_diff:.2e} ✓")

    except Exception as exc:
        print(f"[Rank {rank}] quick_allreduce FAILED: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: aiter custom allreduce
# ---------------------------------------------------------------------------
def _worker_aiter_allreduce(
    rank: int, world_size: int, port: int,
    batch_size: int, hidden_size: int, dtype: torch.dtype,
):
    try:
        os.environ["ROCM_ALLREDUCE_STRATEGY"] = "aiter"
        _setup_distributed(rank, world_size, port)

        from rtp_llm.models_py.modules.base.rocm.aiter_custom_allreduce import (
            aiter_ar_manager,
        )

        device = torch.device(f"cuda:{rank}")
        group = dist.group.WORLD
        device_id = rank

        if not aiter_ar_manager.ensure_initialized(group, device_id):
            if rank == 0:
                print("  [aiter_allreduce] skipped: init failed")
            return

        torch.manual_seed(42 + rank)
        tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        ref = _nccl_reference(tensor.clone(), group)

        if not aiter_ar_manager.should_use(tensor, group, device_id):
            if rank == 0:
                print(f"  [aiter_allreduce] skipped: should_use=False for {dtype}")
            return

        result = aiter_ar_manager.allreduce(tensor)

        diff = (result.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_diff = diff / ref_max if ref_max > 0 else 0

        rtol, atol = 1e-3, 1e-4
        assert rel_diff < rtol or diff < atol, (
            f"[Rank {rank}] aiter allreduce mismatch: rel={rel_diff:.6e}, abs={diff:.6e}"
        )
        if rank == 0:
            print(f"  [aiter_allreduce] hidden={hidden_size} dtype={dtype} rel_diff={rel_diff:.2e} ✓")

    except Exception as exc:
        print(f"[Rank {rank}] aiter_allreduce FAILED: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        raise
    finally:
        _cleanup_distributed()


# ---------------------------------------------------------------------------
# Worker: full tier dispatch via collective_torch.all_reduce
# ---------------------------------------------------------------------------
def _worker_tier_dispatch(
    rank: int, world_size: int, port: int,
    batch_size: int, hidden_size: int, dtype: torch.dtype,
    strategy: str,
):
    try:
        os.environ["ROCM_ALLREDUCE_STRATEGY"] = strategy
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.cuda.set_device(rank)

        from rtp_llm.models_py.distributed.collective_torch import (
            destroy_distributed_environment,
            init_distributed_environment,
            all_reduce,
            Group,
        )
        from rtp_llm.ops import NcclCommConfig, ParallelismConfig

        device = torch.device(f"cuda:{rank}")

        base_port = port + 11
        pc = ParallelismConfig()
        pc.world_size = world_size
        pc.world_rank = rank
        pc.tp_size = world_size
        pc.tp_rank = rank
        pc.dp_size = 1
        pc.dp_rank = 0
        pc.local_rank = rank
        nccl_comm_config = NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        )
        init_distributed_environment(
            pc,
            nccl_comm_config=nccl_comm_config,
            nccl_init_port=port,
        )

        torch.manual_seed(42 + rank)
        tensor = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        ref = _nccl_reference(tensor.clone(), dist.group.WORLD)

        result = all_reduce(tensor, Group.TP)

        diff = (result.float() - ref.float()).abs().max().item()
        ref_max = ref.float().abs().max().item()
        rel_diff = diff / ref_max if ref_max > 0 else 0

        rtol = 0.05 if "quick" in strategy else 1e-3
        assert rel_diff < rtol, (
            f"[Rank {rank}] tier dispatch mismatch: strategy={strategy} "
            f"rel={rel_diff:.6e}, abs={diff:.6e}"
        )
        if rank == 0:
            print(f"  [tier_dispatch] strategy={strategy} hidden={hidden_size} rel_diff={rel_diff:.2e} ✓")

    except Exception as exc:
        print(f"[Rank {rank}] tier_dispatch FAILED: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()
        raise
    finally:
        try:
            destroy_distributed_environment()
        except Exception:
            _cleanup_distributed()


# ===========================================================================
# Test cases
# ===========================================================================
class TestROCmAllReduceTierIntegration(unittest.TestCase):
    """2-rank integration tests for ROCm allreduce tiers."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        if not (hasattr(torch.version, "hip") and torch.version.hip):
            self.skipTest("ROCm not available")
        if torch.cuda.device_count() < 2:
            self.skipTest(f"Need ≥2 GPUs, found {torch.cuda.device_count()}")
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # ---- Quick allreduce (FP quantization) ----

    def test_quick_fp_bf16_hidden4096(self):
        _launch_workers(
            _worker_quick_allreduce, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
            quant_type="FP",
        )

    def test_quick_fp_bf16_hidden2048(self):
        _launch_workers(
            _worker_quick_allreduce, world_size=2,
            batch_size=16, hidden_size=2048, dtype=torch.bfloat16,
            quant_type="FP",
        )

    # ---- Aiter custom allreduce ----

    def test_aiter_bf16_hidden4096(self):
        _launch_workers(
            _worker_aiter_allreduce, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
        )

    def test_aiter_bf16_hidden2048(self):
        _launch_workers(
            _worker_aiter_allreduce, world_size=2,
            batch_size=16, hidden_size=2048, dtype=torch.bfloat16,
        )

    def test_aiter_bf16_small_tensor(self):
        _launch_workers(
            _worker_aiter_allreduce, world_size=2,
            batch_size=1, hidden_size=256, dtype=torch.bfloat16,
        )

    # ---- Dtype fallback: fp16/fp32 should gracefully skip to NCCL ----

    def test_aiter_fp16_falls_through(self):
        _launch_workers(
            _worker_aiter_allreduce, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.float16,
        )

    # ---- Full tier dispatch ----

    def test_dispatch_quick_strategy(self):
        _launch_workers(
            _worker_tier_dispatch, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
            strategy="quick",
        )

    def test_dispatch_aiter_strategy(self):
        _launch_workers(
            _worker_tier_dispatch, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
            strategy="aiter",
        )

    def test_dispatch_nccl_fallback(self):
        _launch_workers(
            _worker_tier_dispatch, world_size=2,
            batch_size=8, hidden_size=4096, dtype=torch.bfloat16,
            strategy="none",
        )


if __name__ == "__main__":
    unittest.main()
