"""DeepGEMM warmup tests — JIT cache correctness + multi-rank speedup."""

import logging
import multiprocessing as mp
import os
import socket
import tempfile
import time
from unittest import SkipTest, TestCase, main

import torch
import torch.nn as nn

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")


def _collect_cache_files(directory: str) -> set[str]:
    result = set()
    for root, _, files in os.walk(directory):
        for f in files:
            result.add(os.path.join(root, f))
    return result


def _count_cubin_files(directory: str) -> int:
    """Count .cubin files in directory tree — each one is a unique JIT compilation."""
    count = 0
    for _, _, files in os.walk(directory):
        count += sum(1 for f in files if f.endswith(".cubin"))
    return count


def _wrap_in_fused_moe(executor):
    """Wrap an executor in a FusedMoe so model.modules() can discover it."""
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe

    moe = FusedMoe.__new__(FusedMoe)
    nn.Module.__init__(moe)
    moe.fused_experts = executor
    return moe


def _build_model(device):
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
        CudaFp8DeepGEMMLinear,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
        DeepGemmHybridExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
        DeepGemmMaskedExecutor,
    )

    E, N, K = 4, 256, 512

    # Dense FP8 linear
    w = torch.randn(K, N, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    ws = torch.ones((K + 127) // 128, (N + 127) // 128, dtype=torch.float32, device=device)
    linear = CudaFp8DeepGEMMLinear(weight=w, weight_scales=ws)

    # MoE masked executor (wrapped in FusedMoe for discovery)
    masked = DeepGemmMaskedExecutor.__new__(DeepGemmMaskedExecutor)
    masked._E, masked._N, masked._K, masked._use_fp8, masked.top_k = E, N, K, True, 1
    masked._w1 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    masked._w2 = torch.randn(E, K, N // 2, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    masked._w1_scale = torch.ones(E, N // 128, K // 128, dtype=torch.float32, device=device)
    masked._w2_scale = torch.ones(E, K // 128, (N // 2) // 128, dtype=torch.float32, device=device)

    # MoE hybrid executor (contiguous path, wrapped in FusedMoe for discovery)
    hybrid = DeepGemmHybridExecutor.__new__(DeepGemmHybridExecutor)
    hybrid.E, hybrid.N, hybrid.K, hybrid.top_k = E, 512, 256, 1
    hybrid.w13_weight = torch.randn(E, 512, 256, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    hybrid.w2_weight = torch.randn(E, 256, 256, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    hybrid.w13_weight_scale_inv = torch.ones(E, 512 // 128, 256 // 128, dtype=torch.float32, device=device)
    hybrid.w2_weight_scale_inv = torch.ones(E, 256 // 128, 256 // 128, dtype=torch.float32, device=device)

    model = nn.Module()
    model.add_module("linear", linear)
    model.add_module("masked_moe", _wrap_in_fused_moe(masked))
    model.add_module("hybrid_moe", _wrap_in_fused_moe(hybrid))
    return model


def _find_free_port() -> int:
    """Find an available port to avoid conflicts in parallel CI runs."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _warmup_worker(rank, num_ranks, cache_dir, barrier, results, master_port):
    """Subprocess entry for warmup benchmark."""
    os.environ["DG_JIT_CACHE_DIR"] = cache_dir
    torch.cuda.set_device(rank)

    if num_ranks > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["NCCL_DEBUG"] = "WARN"
        torch.distributed.init_process_group(
            backend="nccl", world_size=num_ranks, rank=rank,
            device_id=torch.device(f"cuda:{rank}"),
        )

    from rtp_llm.models_py.kernels.cuda.deepgemm_warmup import (
        deep_gemm_warmup, reset_warmed_caches,
    )

    try:
        model = _build_model(f"cuda:{rank}")
        reset_warmed_caches()
        barrier.wait()
        t0 = time.time()
        deep_gemm_warmup(
            model, max_tokens=1024, mode="relax",
            local_rank=rank, local_world_size=num_ranks,
        )
        results[rank] = time.time() - t0
    finally:
        if num_ranks > 1:
            torch.distributed.destroy_process_group()


def _run_warmup(num_ranks: int, cache_dir: str) -> dict[int, float]:
    """Spawn num_ranks processes, run warmup, return {rank: duration}."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(num_ranks)
    results = ctx.Manager().dict()
    master_port = _find_free_port()
    procs = []
    for rank in range(num_ranks):
        p = ctx.Process(target=_warmup_worker, args=(rank, num_ranks, cache_dir, barrier, results, master_port))
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=600)
        assert p.exitcode == 0, f"Rank process failed: exit={p.exitcode}"
    return dict(results)


class DeepGemmWarmupTest(TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        try:
            import deep_gemm
            if not hasattr(deep_gemm, "fp8_gemm_nt"):
                raise SkipTest("deep_gemm package incomplete")
        except ImportError:
            raise SkipTest("deep_gemm not available")

    def test_jit_cache_correctness(self):
        """Verify: first warmup writes cache files, second warmup reuses them (<1s)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Cold: JIT compile
            results = _run_warmup(1, tmpdir)
            cold_time = results[0]
            files_after_cold = _collect_cache_files(tmpdir)
            self.assertGreater(len(files_after_cold), 0, "No JIT cache files written")

            # Warm: should reuse cache (no new files, <1s)
            results = _run_warmup(1, tmpdir)
            warm_time = results[0]
            files_after_warm = _collect_cache_files(tmpdir)

            new_files = files_after_warm - files_after_cold
            self.assertEqual(new_files, set(), f"Unexpected new files: {new_files}")
            self.assertLess(warm_time, 1.0,
                            f"Cache hit should be <1s, got {warm_time:.1f}s")

            cubin_count = _count_cubin_files(tmpdir)
            logging.info(f"Cache correctness: cold={cold_time:.1f}s, warm={warm_time:.1f}s, "
                         f"speedup={cold_time / max(warm_time, 0.01):.0f}x, "
                         f"JIT compilations={cubin_count} .cubin files")

            # Save for test_multi_rank_speedup
            DeepGemmWarmupTest._single_rank_cold = cold_time
            DeepGemmWarmupTest._single_rank_warm = warm_time

    def test_multi_rank_speedup(self):
        """Benchmark: 1/2/4 rank cold JIT compilation + cache hit."""
        max_ranks = 4
        if torch.cuda.device_count() < max_ranks:
            raise SkipTest(f"Need {max_ranks} GPUs")

        # Reuse test_jit_cache_correctness results for 1-rank
        cold_1 = getattr(DeepGemmWarmupTest, "_single_rank_cold", None)
        warm_1 = getattr(DeepGemmWarmupTest, "_single_rank_warm", None)
        if cold_1 is None:
            raise SkipTest("test_jit_cache_correctness must run first")

        timings: dict[int, float] = {1: cold_1}

        for n_ranks in [2, 4]:
            with tempfile.TemporaryDirectory() as tmpdir:
                results = _run_warmup(n_ranks, tmpdir)
                timings[n_ranks] = max(results.values())

        # Print summary
        logging.info("=" * 50)
        logging.info("Multi-rank warmup benchmark (max_tokens=512)")
        logging.info("-" * 50)
        for n_ranks in [1, 2, 4]:
            t = timings[n_ranks]
            logging.info(f"  {n_ranks} rank(s) cold: {t:.1f}s  ({cold_1 / max(t, 0.01):.2f}x)")
        logging.info(f"  1 rank   warm: {warm_1:.1f}s  ({cold_1 / max(warm_1, 0.01):.0f}x)")
        logging.info("=" * 50)


if __name__ == "__main__":
    main()
