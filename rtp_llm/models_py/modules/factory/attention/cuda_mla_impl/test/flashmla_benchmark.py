"""
Benchmark test for flash_mla_with_kvcache and flash_mla_sparse_fwd.
Measures performance across different configurations.
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


# Check if CUDA version >= 12.9 for flash_mla support
def check_cuda_version():
    """Check if CUDA version >= 12.9"""
    try:
        if torch.version.cuda:
            major, minor = map(int, torch.version.cuda.split(".")[:2])
            return (major, minor) >= (12, 9)
        return False
    except (AttributeError, ValueError):
        return False


if not check_cuda_version():
    print(
        f"Error: flash_mla requires CUDA >= 12.9, current: {torch.version.cuda if torch.version.cuda else 'N/A'}"
    )
    sys.exit(1)

from flash_mla import flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata
from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.ops.compute_ops import rtp_llm_ops


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark test."""

    # Shape parameters
    num_tokens: int
    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 512
    top_k: int = 2048
    page_size: int = 64
    total_cache_len: int = 20480
    batch_size: int = 1

    # Benchmark parameters
    warmup_iters: int = 10
    test_iters: int = 100
    softmax_extra_scale: float = 1.0
    seed: int = 42
    use_fp8: bool = False

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __str__(self) -> str:
        return (
            f"BenchmarkConfig(\n"
            f"  num_tokens={self.num_tokens}, "
            f"batch_size={self.batch_size}, "
            f"num_heads={self.num_heads}\n"
            f"  kv_lora_rank={self.kv_lora_rank}, "
            f"qk_head_dim={self.qk_head_dim}, "
            f"top_k={self.top_k}\n"
            f"  total_cache_len={self.total_cache_len}, "
            f"page_size={self.page_size}\n"
            f"  warmup={self.warmup_iters}, "
            f"test={self.test_iters}, "
            f"fp8={self.use_fp8}\n"
            f")"
        )


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    config: BenchmarkConfig
    kernel_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    throughput_tokens_per_sec: float
    memory_allocated_mb: float
    memory_reserved_mb: float

    def __str__(self) -> str:
        return (
            f"{self.kernel_name:30s} | "
            f"Avg: {self.avg_time_ms:7.3f}ms | "
            f"Min: {self.min_time_ms:7.3f}ms | "
            f"Max: {self.max_time_ms:7.3f}ms | "
            f"Std: {self.std_time_ms:7.3f}ms | "
            f"Throughput: {self.throughput_tokens_per_sec:10.1f} tokens/s | "
            f"Mem: {self.memory_allocated_mb:7.1f}MB"
        )


def generate_block_table(
    batch_size: int, total_cache_len: int, page_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate block table for benchmark.

    Returns:
        Tuple of (block_table_host, block_table_device)
    """
    num_blocks_per_seq = math.ceil(total_cache_len / page_size)

    # Create block table: [batch_size, num_blocks_per_seq]
    block_table_host = torch.zeros(
        [batch_size, num_blocks_per_seq], dtype=torch.int32, device=torch.device("cpu")
    )

    bias = 0
    for i in range(batch_size):
        block_table_host[i, :] = torch.arange(
            bias,
            bias + num_blocks_per_seq,
            dtype=torch.int32,
            device=torch.device("cpu"),
        )
        bias += num_blocks_per_seq

    block_table_device = block_table_host.to(device)
    return block_table_host, block_table_device


def benchmark_flash_mla_sparse_fwd(
    config: BenchmarkConfig, device: torch.device
) -> BenchmarkResult:
    """
    Benchmark flash_mla_sparse_fwd kernel.

    KV is produced from FP8 paged cache by cp_gather_and_upconvert_fp8_kv_cache,
    then passed to flash_mla_sparse_fwd. Reported time includes both
    cp_gather_and_upconvert_fp8_kv_cache and flash_mla_sparse_fwd.
    """
    set_seed(config.seed)

    # MLA head_dim for gathered KV (512 + 64 = 576)
    head_dim_kv = 576
    page_size = config.page_size
    total_cache_len = config.total_cache_len
    num_blocks = math.ceil(total_cache_len / page_size)

    # FP8 paged KV cache: [num_blocks, block_size, 656]
    fp8_kv_cache = torch.zeros(
        [num_blocks, page_size, FP8_SPARSE_BYTES_PER_TOKEN],
        dtype=torch.uint8,
        device=device,
    )
    fp8_kv_cache.random_(0, 255)

    # Gather params: single batch, one sequence of length total_cache_len
    block_table = torch.arange(
        0, num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(
        0
    )  # [1, num_blocks]
    seq_lens = torch.tensor([total_cache_len], dtype=torch.int32, device=device)
    workspace_starts = torch.tensor([0], dtype=torch.int32, device=device)

    # BF16 workspace: [total_cache_len, 512] and [total_cache_len, 64]
    compressed_kv = torch.empty(
        [total_cache_len, 512], dtype=torch.bfloat16, device=device
    )
    k_pe = torch.empty([total_cache_len, 64], dtype=torch.bfloat16, device=device)

    # Q: [num_tokens, num_heads, qk_head_dim]
    q = torch.randn(
        [config.num_tokens, config.num_heads, config.qk_head_dim],
        dtype=torch.bfloat16,
        device=device,
    )

    # Global indices: [num_tokens, 1, top_k]
    # For benchmark, use random indices
    global_indices = torch.randint(
        0,
        config.total_cache_len,
        [config.num_tokens, 1, config.top_k],
        dtype=torch.int32,
        device=device,
    )

    scale = (config.qk_head_dim**-0.5) * config.softmax_extra_scale
    # Warmup: gather + flash_mla_sparse_fwd
    for _ in range(config.warmup_iters):
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
            fp8_kv_cache,
            compressed_kv,
            k_pe,
            block_table,
            seq_lens,
            workspace_starts,
            batch_size=1,
        )
        # KV for flash_mla_sparse_fwd: [total_cache_len, 1, 576]
        kv_workspace = torch.cat([compressed_kv, k_pe], dim=-1)
        kv = kv_workspace.unsqueeze(1)
        _, _, _ = flash_mla_sparse_fwd(
            q, kv, global_indices, scale, d_v=config.kv_lora_rank
        )
    torch.cuda.synchronize()

    # Benchmark: time = cp_gather_and_upconvert_fp8_kv_cache + flash_mla_sparse_fwd
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    times = []
    for _ in range(config.test_iters):
        start = time.perf_counter()
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
            fp8_kv_cache,
            compressed_kv,
            k_pe,
            block_table,
            seq_lens,
            workspace_starts,
            batch_size=1,
        )
        # KV for flash_mla_sparse_fwd: [total_cache_len, 1, 576]
        kv_workspace = torch.cat([compressed_kv, k_pe], dim=-1)
        kv = kv_workspace.unsqueeze(1)
        out, _, _ = flash_mla_sparse_fwd(
            q, kv, global_indices, scale, d_v=config.kv_lora_rank
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    end_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Calculate statistics
    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()
    throughput = (config.num_tokens * config.test_iters) / (sum(times) / 1000)

    return BenchmarkResult(
        config=config,
        kernel_name="flash_mla_sparse_fwd+gather_fp8",
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=end_mem - start_mem,
        memory_reserved_mb=peak_mem,
    )


def benchmark_cp_gather_and_upconvert_fp8_kv_cache(
    config: BenchmarkConfig, device: torch.device
) -> BenchmarkResult:
    """
    Benchmark cp_gather_and_upconvert_fp8_kv_cache only (FP8 paged cache -> BF16 workspace).
    """
    set_seed(config.seed)

    page_size = config.page_size
    total_cache_len = config.total_cache_len
    num_blocks = math.ceil(total_cache_len / page_size)

    fp8_kv_cache = torch.zeros(
        [num_blocks, page_size, FP8_SPARSE_BYTES_PER_TOKEN],
        dtype=torch.uint8,
        device=device,
    )
    fp8_kv_cache.random_(0, 255)

    block_table = torch.arange(
        0, num_blocks, dtype=torch.int32, device=device
    ).unsqueeze(0)
    seq_lens = torch.tensor([total_cache_len], dtype=torch.int32, device=device)
    workspace_starts = torch.tensor([0], dtype=torch.int32, device=device)
    compressed_kv = torch.empty(
        [total_cache_len, 512], dtype=torch.bfloat16, device=device
    )
    k_pe = torch.empty([total_cache_len, 64], dtype=torch.bfloat16, device=device)

    for _ in range(config.warmup_iters):
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
            fp8_kv_cache,
            compressed_kv,
            k_pe,
            block_table,
            seq_lens,
            workspace_starts,
            batch_size=1,
        )
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024

    times = []
    for _ in range(config.test_iters):
        start = time.perf_counter()
        rtp_llm_ops.cp_gather_and_upconvert_fp8_kv_cache(
            fp8_kv_cache,
            compressed_kv,
            k_pe,
            block_table,
            seq_lens,
            workspace_starts,
            batch_size=1,
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    end_mem = torch.cuda.memory_allocated() / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()
    # Throughput: cache tokens gathered per second
    throughput = (total_cache_len * config.test_iters) / (sum(times) / 1000)

    return BenchmarkResult(
        config=config,
        kernel_name="cp_gather_and_upconvert_fp8_kv_cache",
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=end_mem - start_mem,
        memory_reserved_mb=peak_mem,
    )


# DeepSeek V3 FP8 sparse KV cache layout: 656 bytes per token
# (512 fp8 NoPE + 16 scale float32 + 128 bf16 RoPE). flash_mla_with_kvcache with
# indices (sparse) requires is_fp8_kvcache=True and this shape.
FP8_SPARSE_BYTES_PER_TOKEN = 656


def benchmark_flash_mla_with_kvcache(
    config: BenchmarkConfig, device: torch.device, is_decode: bool = True
) -> BenchmarkResult:
    """
    Benchmark flash_mla_with_kvcache kernel.

    This kernel is used in the SparseMlaFp8Op class for FP8 quantized attention.
    When using sparse attention (indices not None), FlashMLA requires
    is_fp8_kvcache=True and k_cache shape (num_blocks, page_block_size, h_kv, bytes_per_token)
    with bytes_per_token=656 (DeepSeek V3 layout).
    """
    set_seed(config.seed)

    # Sparse path (indices) requires FP8; use_fp8 is forced True for this benchmark path.
    is_fp8 = True

    # Generate input tensors
    # Q: [1, num_tokens, num_heads, qk_head_dim] (batched format)
    q = torch.randn(
        [1, config.num_tokens, config.num_heads, config.qk_head_dim],
        dtype=torch.bfloat16,
        device=device,
    )

    # KV cache for sparse+FP8: (num_blocks, page_block_size, h_kv, bytes_per_token)
    # Last dim must be 656 for DeepSeek V3 FP8 sparse layout.
    num_blocks = (
        math.ceil(config.total_cache_len / config.page_size) * config.batch_size
    )
    kv_cache = torch.zeros(
        [num_blocks, config.page_size, 1, FP8_SPARSE_BYTES_PER_TOKEN],
        dtype=torch.uint8,
        device=device,
    )
    # Fill with random bytes so memory is touched (benchmark is timing-only)
    kv_cache.random_(0, 255)

    # Global indices: [1, num_tokens, top_k] (batched format)
    global_indices = torch.randint(
        0,
        config.total_cache_len,
        [1, config.num_tokens, config.top_k],
        dtype=torch.int32,
        device=device,
    )

    # Generate block table
    block_table_host, block_table_device = generate_block_table(
        config.batch_size,
        config.total_cache_len // config.batch_size,
        config.page_size,
        device,
    )

    # Get metadata (sparse requires is_fp8_kvcache=True)
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=config.num_tokens * config.num_heads,
        topk=config.top_k,
        num_heads_q=config.num_heads,
        num_heads_k=1,
        is_fp8_kvcache=is_fp8,
    )

    scale = (config.qk_head_dim**-0.5) * config.softmax_extra_scale

    # Warmup
    for _ in range(config.warmup_iters):
        tile_scheduler_metadata.tile_scheduler_metadata = None
        tile_scheduler_metadata.num_splits = None
        _, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache,
            block_table=block_table_device,
            head_dim_v=config.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            is_fp8_kvcache=is_fp8,
            indices=global_indices,
            softmax_scale=scale,
        )
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    times = []
    for _ in range(config.test_iters):
        tile_scheduler_metadata.tile_scheduler_metadata = None
        tile_scheduler_metadata.num_splits = None

        start = time.perf_counter()
        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_cache,
            block_table=block_table_device,
            head_dim_v=config.kv_lora_rank,
            cache_seqlens=None,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            is_fp8_kvcache=is_fp8,
            indices=global_indices,
            softmax_scale=scale,
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    end_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Calculate statistics
    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()
    throughput = (config.num_tokens * config.test_iters) / (sum(times) / 1000)

    kernel_name = "flash_mla_with_kvcache_fp8_sparse"

    return BenchmarkResult(
        config=config,
        kernel_name=kernel_name,
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=end_mem - start_mem,
        memory_reserved_mb=peak_mem,
    )


def benchmark_batch_prefill_with_ragged_kvcache_wrapper(
    config: BenchmarkConfig, device: torch.device
) -> BenchmarkResult:
    """
    Benchmark BatchPrefillWithRaggedKVCacheWrapper.

    This wrapper is used for prefill stage with standard attention (not sparse).
    Note: FlashInfer requires head_dim_vo to be 64, 128, or 256 on SM90.
    """
    set_seed(config.seed)

    # FlashInfer constraint: head_dim_vo must be 64, 128, or 256
    # Use 128 as a reasonable value for V dimension
    head_dim_vo = 128

    # Create workspace buffer
    workspace_buffer = torch.empty(
        512 * 1024 * 1024,
        dtype=torch.int8,
        device=device,
    )

    # Create wrapper
    prefill_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer,
        "NHD",
        backend="auto",
        use_cuda_graph=False,
    )

    # Generate input tensors
    # Q: [num_tokens, num_heads, qk_head_dim]
    qk_head_dim = 128 + 64
    q = torch.randn(
        [config.num_tokens, config.num_heads, qk_head_dim],
        dtype=torch.bfloat16,
        device=device,
    )

    # K: [total_cache_len, num_heads, qk_head_dim]
    # V: [total_cache_len, num_heads, head_dim_vo] (limited to 64/128/256)
    k = torch.randn(
        [config.num_tokens, config.num_heads, qk_head_dim],
        dtype=torch.bfloat16,
        device=device,
    )
    v = torch.randn(
        [config.num_tokens, config.num_heads, head_dim_vo],
        dtype=torch.bfloat16,
        device=device,
    )

    # Create qo_indptr and kv_indptr for ragged tensor
    # Single batch: qo_indptr = [0, num_tokens]
    # kv_indptr = [0, total_cache_len]
    qo_indptr = torch.tensor([0, config.num_tokens], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, config.num_tokens], dtype=torch.int32, device=device)

    scale = (qk_head_dim**-0.5) * config.softmax_extra_scale

    # Plan with head_dim_vo = 128
    prefill_wrapper.plan(
        qo_indptr,
        kv_indptr,
        config.num_heads,
        config.num_heads,
        qk_head_dim,
        head_dim_vo,
        sm_scale=scale,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )

    # Warmup
    for _ in range(config.warmup_iters):
        _ = prefill_wrapper.run(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB

    times = []
    for _ in range(config.test_iters):
        start = time.perf_counter()
        out = prefill_wrapper.run(q, k, v)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    end_mem = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Calculate statistics
    times_tensor = torch.tensor(times)
    avg_time = times_tensor.mean().item()
    min_time = times_tensor.min().item()
    max_time = times_tensor.max().item()
    std_time = times_tensor.std().item()
    throughput = (config.num_tokens * config.test_iters) / (sum(times) / 1000)

    return BenchmarkResult(
        config=config,
        kernel_name=f"BatchPrefillWithRaggedKVCacheWrapper(vo={head_dim_vo})",
        avg_time_ms=avg_time,
        min_time_ms=min_time,
        max_time_ms=max_time,
        std_time_ms=std_time,
        throughput_tokens_per_sec=throughput,
        memory_allocated_mb=end_mem - start_mem,
        memory_reserved_mb=peak_mem,
    )


def print_results_table(results: List[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 140)
    print("BENCHMARK RESULTS")
    print("=" * 140)

    # Group by kernel
    kernel_results: Dict[str, List[BenchmarkResult]] = {}
    for result in results:
        if result.kernel_name not in kernel_results:
            kernel_results[result.kernel_name] = []
        kernel_results[result.kernel_name].append(result)

    for kernel_name, kernel_res in kernel_results.items():
        print(f"\n{kernel_name}:")
        print("-" * 140)
        for result in kernel_res:
            cfg = result.config
            print(
                f"  T={cfg.num_tokens:5d} | H={cfg.num_heads:3d} | "
                f"D={cfg.qk_head_dim:3d} | K={cfg.top_k:3d} | "
                f"Cache={cfg.total_cache_len:5d} | {result}"
            )

    print("=" * 140)


def print_sparse_fwd_vs_with_kvcache_comparison(results: List[BenchmarkResult]) -> None:
    """
    Compare flash_mla_sparse_fwd+gather_fp8 vs flash_mla_with_kvcache_fp8_sparse.
    Baseline = flash_mla_with_kvcache_fp8_sparse.
    faster% = (baseline_avg_ms - sparse_fwd_gather_avg_ms) / baseline_avg_ms * 100
    (positive = sparse_fwd+gather faster than baseline, negative = slower).
    """
    key_fwd = "flash_mla_sparse_fwd+gather_fp8"
    key_kvcache = "flash_mla_with_kvcache_fp8_sparse"

    by_kernel: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.kernel_name not in by_kernel:
            by_kernel[r.kernel_name] = []
        by_kernel[r.kernel_name].append(r)

    if key_fwd not in by_kernel or key_kvcache not in by_kernel:
        return

    # Index by (num_tokens, num_heads)
    def config_key(cfg: BenchmarkConfig) -> tuple:
        return (cfg.num_tokens, cfg.num_heads)

    fwd_by_key = {config_key(r.config): r for r in by_kernel[key_fwd]}
    kvcache_by_key = {config_key(r.config): r for r in by_kernel[key_kvcache]}

    common_keys = sorted(set(fwd_by_key) & set(kvcache_by_key))
    if not common_keys:
        return

    print("\n" + "=" * 120)
    print(
        "COMPARISON: flash_mla_sparse_fwd+gather_fp8 vs flash_mla_with_kvcache_fp8_sparse (baseline)"
    )
    print(
        "  faster% = (baseline - sparse_fwd+gather) / baseline * 100  (positive = sparse_fwd+gather 更快)"
    )
    print("=" * 120)
    print(
        f"  {'num_tokens':>10} | {'num_heads':>8} | "
        f"{'sparse_fwd+gather Avg(ms)':>24} | "
        f"{'with_kvcache Avg(ms)':>22} | "
        f"{'faster %':>10}"
    )
    print("-" * 120)
    for num_tokens, num_heads in common_keys:
        r_fwd = fwd_by_key[(num_tokens, num_heads)]
        r_kvcache = kvcache_by_key[(num_tokens, num_heads)]
        baseline_ms = r_kvcache.avg_time_ms
        other_ms = r_fwd.avg_time_ms
        faster_pct = (
            (baseline_ms - other_ms) / baseline_ms * 100.0 if baseline_ms > 0 else 0.0
        )
        print(
            f"  {num_tokens:10d} | {num_heads:8d} | "
            f"{r_fwd.avg_time_ms:24.3f} | "
            f"{r_kvcache.avg_time_ms:22.3f} | "
            f"{faster_pct:+9.2f}%"
        )
    print("=" * 120)


def run_benchmark_suite():
    """Run benchmark with two loops: num_heads in (64, 128), num_tokens 1 to 20480."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # num_tokens: 1 to 20480 (powers of 2 + 20480 to keep run count reasonable)
    num_tokens_list = sorted(set([2**i for i in range(0, 15)] + [10240, 20480]))
    num_heads_list = [64, 128]

    results = []

    # flash_mla_sparse_fwd
    print("\n" + "=" * 80)
    print("Benchmarking flash_mla_sparse_fwd")
    print("=" * 80)
    total_sparse = len(num_heads_list) * len(num_tokens_list)
    idx = 0
    for num_heads in num_heads_list:
        for num_tokens in num_tokens_list:
            idx += 1
            config = BenchmarkConfig(
                num_tokens=num_tokens,
                num_heads=num_heads,
                kv_lora_rank=512,
                top_k=2048,
            )
            print(
                f"\n[{idx}/{total_sparse}] num_heads={num_heads}, num_tokens={num_tokens}"
            )
            try:
                result = benchmark_flash_mla_sparse_fwd(config, device)
                results.append(result)
                print(f"  {result}")
            except Exception as e:
                print(f"  ERROR: {e}")
            try:
                result_gather = benchmark_cp_gather_and_upconvert_fp8_kv_cache(
                    config, device
                )
                results.append(result_gather)
                print(f"  {result_gather}")
            except Exception as e:
                print(f"  ERROR (gather): {e}")

    # flash_mla_with_kvcache
    print("\n" + "=" * 80)
    print("Benchmarking flash_mla_with_kvcache")
    print("=" * 80)
    idx = 0
    for num_heads in num_heads_list:
        for num_tokens in num_tokens_list:
            idx += 1
            config = BenchmarkConfig(
                num_tokens=num_tokens,
                num_heads=num_heads,
                kv_lora_rank=512,
                top_k=2048,
                use_fp8=True,
            )
            print(
                f"\n[{idx}/{total_sparse}] num_heads={num_heads}, num_tokens={num_tokens}"
            )
            try:
                result = benchmark_flash_mla_with_kvcache(
                    config, device, is_decode=True
                )
                results.append(result)
                print(f"  {result}")
            except Exception as e:
                print(f"  ERROR: {e}")

    # BatchPrefillWithRaggedKVCacheWrapper
    print("\n" + "=" * 80)
    print("Benchmarking BatchPrefillWithRaggedKVCacheWrapper")
    print("=" * 80)
    idx = 0
    for num_heads in num_heads_list:
        for num_tokens in num_tokens_list:
            idx += 1
            config = BenchmarkConfig(
                num_tokens=num_tokens,
                num_heads=num_heads,
                kv_lora_rank=512,
                top_k=2048,
            )
            print(
                f"\n[{idx}/{total_sparse}] num_heads={num_heads}, num_tokens={num_tokens}"
            )
            try:
                result = benchmark_batch_prefill_with_ragged_kvcache_wrapper(
                    config, device
                )
                results.append(result)
                print(f"  {result}")
            except Exception as e:
                print(f"  ERROR: {e}")

    print_results_table(results)
    print_sparse_fwd_vs_with_kvcache_comparison(results)
    return results


def run_custom_benchmark(args):
    """Run benchmark with custom parameters from command line."""
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    config = BenchmarkConfig(
        num_tokens=args.num_tokens,
        num_heads=args.num_heads,
        kv_lora_rank=args.kv_lora_rank,
        qk_rope_head_dim=args.qk_rope_head_dim,
        qk_nope_head_dim=args.qk_nope_head_dim,
        top_k=args.top_k,
        page_size=args.page_size,
        total_cache_len=args.total_cache_len,
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        test_iters=args.iters,
        softmax_extra_scale=args.softmax_scale,
        seed=args.seed,
        use_fp8=args.use_fp8,
    )

    print("\n" + "=" * 80)
    print("Running Custom Benchmark")
    print("=" * 80)
    print(f"\nConfiguration:\n{config}")

    results = []

    if args.kernel in ["sparse_fwd", "all"]:
        print("\nBenchmarking cp_gather_and_upconvert_fp8_kv_cache only")
        try:
            result_gather = benchmark_cp_gather_and_upconvert_fp8_kv_cache(
                config, device
            )
            results.append(result_gather)
            print(f"Result: {result_gather}")
        except Exception as e:
            print(f"ERROR: {e}")

        print("\n" + "-" * 80)
        print("Benchmarking flash_mla_sparse_fwd (+ gather)")
        print("-" * 80)
        try:
            result = benchmark_flash_mla_sparse_fwd(config, device)
            results.append(result)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"ERROR: {e}")

    if args.kernel in ["with_kvcache", "all"]:
        print("\n" + "-" * 80)
        print("Benchmarking flash_mla_with_kvcache")
        print("-" * 80)
        try:
            result = benchmark_flash_mla_with_kvcache(
                config, device, is_decode=args.is_decode
            )
            results.append(result)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"ERROR: {e}")

    if args.kernel in ["ragged_prefill", "all"]:
        print("\n" + "-" * 80)
        print("Benchmarking BatchPrefillWithRaggedKVCacheWrapper")
        print("-" * 80)
        try:
            result = benchmark_batch_prefill_with_ragged_kvcache_wrapper(config, device)
            results.append(result)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"ERROR: {e}")

    if len(results) > 1:
        print_results_table(results)
        print_sparse_fwd_vs_with_kvcache_comparison(results)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark flash_mla_with_kvcache and flash_mla_sparse_fwd kernels"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["suite", "custom"],
        default="suite",
        help="Run benchmark suite or custom configuration",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["sparse_fwd", "with_kvcache", "ragged_prefill", "all"],
        default="all",
        help="Which kernel to benchmark (default: all)",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument(
        "--num-tokens", type=int, default=128, help="Number of query tokens"
    )
    parser.add_argument(
        "--num-heads", type=int, default=64, help="Number of attention heads"
    )
    parser.add_argument("--kv-lora-rank", type=int, default=512, help="KV lora rank")
    parser.add_argument(
        "--qk-rope-head-dim", type=int, default=64, help="QK rope head dimension"
    )
    parser.add_argument(
        "--qk-nope-head-dim", type=int, default=448, help="QK nope head dimension"
    )
    parser.add_argument(
        "--top-k", type=int, default=128, help="Top K for sparse attention"
    )
    parser.add_argument(
        "--page-size", type=int, default=64, help="Page size for KV cache"
    )
    parser.add_argument(
        "--total-cache-len", type=int, default=2048, help="Total cache length"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of test iterations"
    )
    parser.add_argument(
        "--softmax-scale", type=float, default=1.0, help="Softmax extra scale"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-fp8", action="store_true", help="Use FP8 quantization")
    parser.add_argument(
        "--is-decode", action="store_true", help="Use decode stage parameters"
    )

    args = parser.parse_args()

    if args.mode == "suite":
        run_benchmark_suite()
    else:
        run_custom_benchmark(args)


if __name__ == "__main__":
    main()
