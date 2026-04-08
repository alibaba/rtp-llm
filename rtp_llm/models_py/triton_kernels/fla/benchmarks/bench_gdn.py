# -*- coding: utf-8 -*-
"""
GDN (Gated Delta Network) kernel-level benchmark for Qwen3.5 hybrid attention.

Benchmarks three scenarios:
  1. Prefill: FLA Triton chunk_gated_delta_rule vs FlashInfer (SM90+)
  2. Chunked Prefill: varying chunk sizes with initial_state
  3. Decode: FLA split → FLA fused → FlashInfer FP32 → FlashInfer BF16

Usage:
    # Full benchmark on H20/H800 (SM90+)
    python -m rtp_llm.models_py.triton_kernels.fla.benchmarks.bench_gdn

    # Skip FlashInfer (A10 / SM86)
    python -m rtp_llm.models_py.triton_kernels.fla.benchmarks.bench_gdn --skip-flashinfer

    # Specific scenario
    python -m rtp_llm.models_py.triton_kernels.fla.benchmarks.bench_gdn --scenario decode

    # Output CSV
    python -m rtp_llm.models_py.triton_kernels.fla.benchmarks.bench_gdn --output results.csv
"""

import argparse
import csv
import io
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_gdn")

# ---------------------------------------------------------------------------
# Qwen3.5 default config
# ---------------------------------------------------------------------------
QWEN35_NUM_K_HEADS = 16
QWEN35_NUM_V_HEADS = 32
QWEN35_HEAD_K_DIM = 128
QWEN35_HEAD_V_DIM = 128
QWEN35_CONV_KERNEL = 4
QWEN35_CHUNK_SIZE = 64

WARMUP_ITERS = 10
BENCH_ITERS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    scenario: str
    backend: str
    batch_size: int
    seq_len: int
    extra: str  # e.g. "chunk=256" or "state=bf16"
    latency_us: float
    throughput_tok_per_s: float = 0.0


def _do_bench(fn: Callable, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> float:
    """Returns median latency in microseconds."""
    # warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    times.sort()
    # median
    n = len(times)
    return times[n // 2]


def _check_flashinfer_gdn() -> bool:
    """Check if FlashInfer GDN kernels are available."""
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def _make_qkv(batch: int, seq: int, device: str = "cuda", dtype=torch.bfloat16):
    """Create random Q, K, V tensors in Qwen3.5 config."""
    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    q = torch.randn(batch, seq, HK, K, device=device, dtype=dtype)
    k = torch.randn(batch, seq, HK, K, device=device, dtype=dtype)
    v = torch.randn(batch, seq, HV, V, device=device, dtype=dtype)
    return q, k, v


def _make_gates(batch: int, seq: int, device: str = "cuda", dtype=torch.bfloat16):
    """Create gate tensors (g, beta) for prefill, or raw (A_log, a, dt_bias, b) for decode."""
    HV = QWEN35_NUM_V_HEADS
    g = torch.randn(batch, seq, HV, device=device, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(batch, seq, HV, device=device, dtype=dtype)).to(torch.float32)
    return g, beta


def _make_raw_gates(batch: int, seq: int, device: str = "cuda", dtype=torch.bfloat16):
    """Raw gate parameters as stored in the model (A_log, a, dt_bias, b)."""
    HV = QWEN35_NUM_V_HEADS
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.1
    a = torch.randn(batch, seq, HV, device=device, dtype=dtype)
    dt_bias = torch.randn(HV, device=device, dtype=dtype)
    b = torch.randn(batch, seq, HV, device=device, dtype=dtype)
    return A_log, a, dt_bias, b


def _make_state(batch: int, device: str = "cuda", dtype=torch.float32):
    """Create SSM state [B, HV, V, K] (K-last layout)."""
    HV, V, K = QWEN35_NUM_V_HEADS, QWEN35_HEAD_V_DIM, QWEN35_HEAD_K_DIM
    return torch.randn(batch, HV, V, K, device=device, dtype=dtype) * 0.01


def _make_block_map(batch: int, seq_len: int, seq_size_per_block: int = 64, device: str = "cuda"):
    """Create a simple block_map for benchmark (no sparse allocation)."""
    num_blocks_per_seq = (seq_len + seq_size_per_block - 1) // seq_size_per_block + 1
    total_blocks = batch * num_blocks_per_seq + 1  # +1 for padding
    block_map = torch.zeros(batch, num_blocks_per_seq, device=device, dtype=torch.int32)
    for i in range(batch):
        for j in range(num_blocks_per_seq):
            block_map[i, j] = i * num_blocks_per_seq + j + 1  # block 0 reserved
    return block_map, total_blocks


# ---------------------------------------------------------------------------
# Prefill Benchmarks
# ---------------------------------------------------------------------------
def bench_prefill_fla_triton(batch: int, seq_len: int) -> float:
    """Benchmark FLA Triton chunk_gated_delta_rule (prefill)."""
    from rtp_llm.models_py.triton_kernels.fla import chunk_gated_delta_rule

    q, k, v = _make_qkv(batch, seq_len)
    g, beta = _make_gates(batch, seq_len)
    initial_state = _make_state(batch)

    # Build cu_seqlens for single batch (varlen format)
    cu_seqlens = torch.arange(0, (batch + 1) * seq_len, seq_len, device="cuda", dtype=torch.long)

    # Flatten batch into varlen
    q_flat = q.reshape(1, batch * seq_len, q.shape[2], q.shape[3])
    k_flat = k.reshape(1, batch * seq_len, k.shape[2], k.shape[3])
    v_flat = v.reshape(1, batch * seq_len, v.shape[2], v.shape[3])
    g_flat = g.reshape(1, batch * seq_len, g.shape[2])
    beta_flat = beta.reshape(1, batch * seq_len, beta.shape[2])

    def run():
        chunk_gated_delta_rule(
            q_flat, k_flat, v_flat, g_flat, beta_flat,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    return _do_bench(run)


def bench_prefill_flashinfer(batch: int, seq_len: int) -> float:
    """Benchmark FlashInfer chunk_gated_delta_rule (prefill, SM90+)."""
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as fi_chunk_gdr

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    total_tokens = batch * seq_len

    q = torch.randn(total_tokens, HK, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_tokens, HK, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total_tokens, HV, V, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(total_tokens, HV, device="cuda", dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(total_tokens, HV, device="cuda", dtype=torch.float32))
    initial_state = torch.randn(batch, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    cu_seqlens = torch.arange(0, (batch + 1) * seq_len, seq_len, device="cuda", dtype=torch.int64)

    def run():
        fi_chunk_gdr(
            q, k, v,
            g=g, beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

    return _do_bench(run)


# ---------------------------------------------------------------------------
# Chunked Prefill Benchmarks
# ---------------------------------------------------------------------------
def bench_chunked_prefill_fla(batch: int, total_seq_len: int, chunk_size: int, with_state: bool) -> float:
    """Benchmark FLA Triton chunked prefill with varying chunk sizes."""
    from rtp_llm.models_py.triton_kernels.fla import chunk_gated_delta_rule

    num_chunks = (total_seq_len + chunk_size - 1) // chunk_size
    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM

    # Pre-generate all chunks
    chunks_q = [torch.randn(1, batch * min(chunk_size, total_seq_len - i * chunk_size), HK, K, device="cuda", dtype=torch.bfloat16) for i in range(num_chunks)]
    chunks_k = [torch.randn_like(chunks_q[i]) for i in range(num_chunks)]
    chunks_v = [torch.randn(1, chunks_q[i].shape[1], HV, V, device="cuda", dtype=torch.bfloat16) for i in range(num_chunks)]
    chunks_g = [torch.randn(1, chunks_q[i].shape[1], HV, device="cuda", dtype=torch.float32) * 0.1 for i in range(num_chunks)]
    chunks_beta = [torch.sigmoid(torch.randn(1, chunks_q[i].shape[1], HV, device="cuda", dtype=torch.float32)) for i in range(num_chunks)]

    state = _make_state(batch) if with_state else None

    def run():
        s = state.clone() if state is not None else None
        for i in range(num_chunks):
            actual_len = chunks_q[i].shape[1] // batch
            cu = torch.arange(0, (batch + 1) * actual_len, actual_len, device="cuda", dtype=torch.long)
            _, _, s = chunk_gated_delta_rule(
                chunks_q[i], chunks_k[i], chunks_v[i], chunks_g[i], chunks_beta[i],
                initial_state=s,
                output_final_state=True,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
            )

    return _do_bench(run)


# ---------------------------------------------------------------------------
# Decode Benchmarks
# ---------------------------------------------------------------------------
def bench_decode_fla_split(batch: int, seq_len: int = 1024) -> float:
    """Benchmark FLA Triton decode: fused_gdn_gating + fused_recurrent (2 kernels)."""
    from rtp_llm.models_py.triton_kernels.fla import fused_recurrent_gated_delta_rule
    from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    SPB = 64

    q, k, v = _make_qkv(batch, 1)
    A_log, a_raw, dt_bias, b_raw = _make_raw_gates(batch, 1)
    a_2d = a_raw.squeeze(1)  # [B, HV]
    b_2d = b_raw.squeeze(1)

    block_map, total_blocks = _make_block_map(batch, seq_len, SPB)
    ssm_pool = torch.randn(total_blocks, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    seq_lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    def run():
        g, beta = fused_gdn_gating(A_log, a_2d, b_2d, dt_bias)
        g = g.view(batch, 1, HV)
        beta = beta.view(batch, 1, HV)
        fused_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            scale=None,
            initial_state=ssm_pool,
            inplace_final_state=True,
            block_map=block_map,
            seq_size_per_block=SPB,
            sequence_lengths=seq_lengths,
            use_qk_l2norm_in_kernel=True,
        )

    return _do_bench(run)


def bench_decode_flashinfer_fp32(batch: int, seq_len: int = 1024) -> float:
    """Benchmark FlashInfer decode with FP32 state."""
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    SPB = 64

    q = torch.randn(batch, 1, HK, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, HK, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log, a, dt_bias, b = _make_raw_gates(batch, 1)

    block_map, total_blocks = _make_block_map(batch, seq_len, SPB)
    ssm_pool = torch.randn(total_blocks, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    seq_lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    # Compute read indices
    read_offsets = (seq_lengths - 1) // SPB
    indices = block_map[torch.arange(batch, device="cuda"), read_offsets].to(torch.int64)

    def run():
        gated_delta_rule_decode_pretranspose(
            q, k, v,
            state=None,
            A_log=A_log, a=a, dt_bias=dt_bias, b=b,
            initial_state=ssm_pool,
            initial_state_indices=indices,
            use_qk_l2norm=True,
        )

    return _do_bench(run)


def bench_decode_flashinfer_bf16(batch: int, seq_len: int = 1024) -> float:
    """Benchmark FlashInfer decode with BF16 state (K=V=128 fast path)."""
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    SPB = 64

    q = torch.randn(batch, 1, HK, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, HK, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, 1, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log, a, dt_bias, b = _make_raw_gates(batch, 1)

    block_map, total_blocks = _make_block_map(batch, seq_len, SPB)
    ssm_pool = torch.randn(total_blocks, HV, V, K, device="cuda", dtype=torch.bfloat16) * 0.01
    seq_lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    read_offsets = (seq_lengths - 1) // SPB
    indices = block_map[torch.arange(batch, device="cuda"), read_offsets].to(torch.int64)

    def run():
        gated_delta_rule_decode_pretranspose(
            q, k, v,
            state=None,
            A_log=A_log, a=a, dt_bias=dt_bias, b=b,
            initial_state=ssm_pool,
            initial_state_indices=indices,
            use_qk_l2norm=True,
        )

    return _do_bench(run)


def bench_decode_flashinfer_mtp(batch: int, num_tokens: int = 4, seq_len: int = 1024) -> float:
    """Benchmark FlashInfer MTP decode (multi-token, for speculative decoding)."""
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    SPB = 64

    q = torch.randn(batch, num_tokens, HK, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, num_tokens, HK, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, num_tokens, HV, V, device="cuda", dtype=torch.bfloat16)
    A_log, a, dt_bias, b = _make_raw_gates(batch, num_tokens)

    block_map, total_blocks = _make_block_map(batch, seq_len, SPB)
    ssm_pool = torch.randn(total_blocks, HV, V, K, device="cuda", dtype=torch.float32) * 0.01
    seq_lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    read_offsets = (seq_lengths - 1) // SPB
    indices = block_map[torch.arange(batch, device="cuda"), read_offsets].to(torch.int64)

    def run():
        gated_delta_rule_mtp(
            q, k, v,
            initial_state=ssm_pool,
            initial_state_indices=indices,
            A_log=A_log, a=a, dt_bias=dt_bias, b=b,
            use_qk_l2norm=True,
        )

    return _do_bench(run)


# ---------------------------------------------------------------------------
# Conv1d Benchmark (baseline)
# ---------------------------------------------------------------------------
def bench_conv1d_update(batch: int) -> float:
    """Benchmark causal_conv1d_update (no alternative, baseline only)."""
    from rtp_llm.models_py.triton_kernels.causal_conv1d.causal_conv1d import causal_conv1d_update

    HK, HV, K, V = QWEN35_NUM_K_HEADS, QWEN35_NUM_V_HEADS, QWEN35_HEAD_K_DIM, QWEN35_HEAD_V_DIM
    qkv_size = HK * K * 2 + HV * V
    conv_kernel = QWEN35_CONV_KERNEL
    SPB = 64
    seq_len = 1024

    x = torch.randn(batch, qkv_size, 1, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(qkv_size, conv_kernel, device="cuda", dtype=torch.bfloat16)

    block_map, total_blocks = _make_block_map(batch, seq_len, SPB)
    conv_state = torch.randn(total_blocks, qkv_size, conv_kernel - 1, device="cuda", dtype=torch.bfloat16) * 0.01
    seq_lengths = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)

    def run():
        causal_conv1d_update(
            x, conv_state, weight,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=block_map,
            seq_size_per_block=SPB,
            sequence_lengths=seq_lengths,
        )

    return _do_bench(run)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_benchmarks(
    scenarios: List[str],
    skip_flashinfer: bool = False,
    output_file: Optional[str] = None,
):
    results: List[BenchResult] = []
    fi_available = not skip_flashinfer and _check_flashinfer_gdn()

    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability()
    log.info(f"GPU: {device_name} (SM{major}{minor})")
    log.info(f"FlashInfer GDN: {'available' if fi_available else 'not available'}")
    log.info(f"Scenarios: {scenarios}")
    log.info("=" * 80)

    # ---- Prefill ----
    if "prefill" in scenarios or "all" in scenarios:
        log.info("=== PREFILL BENCHMARK ===")
        for batch in [1, 4, 8]:
            for seq_len in [128, 512, 1024, 2048, 4096]:
                total_tokens = batch * seq_len
                # FLA Triton
                lat = bench_prefill_fla_triton(batch, seq_len)
                tput = total_tokens / (lat / 1e6)
                results.append(BenchResult("prefill", "fla_triton", batch, seq_len, "", lat, tput))
                log.info(f"  FLA Triton  B={batch} L={seq_len}: {lat:.0f} us ({tput:.0f} tok/s)")

                # FlashInfer
                if fi_available:
                    lat = bench_prefill_flashinfer(batch, seq_len)
                    tput = total_tokens / (lat / 1e6)
                    results.append(BenchResult("prefill", "flashinfer", batch, seq_len, "", lat, tput))
                    log.info(f"  FlashInfer  B={batch} L={seq_len}: {lat:.0f} us ({tput:.0f} tok/s)")

    # ---- Chunked Prefill ----
    if "chunked_prefill" in scenarios or "all" in scenarios:
        log.info("=== CHUNKED PREFILL BENCHMARK ===")
        total_seq = 4096
        for batch in [1, 4]:
            for chunk_size in [128, 256, 512, 1024, 2048, 4096]:
                for with_state in [False, True]:
                    state_tag = "with_state" if with_state else "no_state"
                    lat = bench_chunked_prefill_fla(batch, total_seq, chunk_size, with_state)
                    tput = (batch * total_seq) / (lat / 1e6)
                    results.append(BenchResult("chunked_prefill", "fla_triton", batch, total_seq,
                                               f"chunk={chunk_size},{state_tag}", lat, tput))
                    log.info(f"  FLA B={batch} total={total_seq} chunk={chunk_size} {state_tag}: {lat:.0f} us ({tput:.0f} tok/s)")

    # ---- Decode ----
    if "decode" in scenarios or "all" in scenarios:
        log.info("=== DECODE BENCHMARK ===")
        for batch in [1, 4, 8, 16, 32, 64, 128, 256]:
            for seq_len in [256, 1024, 4096]:
                # FLA split (current)
                lat = bench_decode_fla_split(batch, seq_len)
                tput = batch / (lat / 1e6)
                results.append(BenchResult("decode", "fla_split", batch, seq_len, "", lat, tput))
                log.info(f"  FLA split   B={batch} L={seq_len}: {lat:.0f} us ({tput:.0f} tok/s)")

                # FlashInfer FP32
                if fi_available:
                    lat = bench_decode_flashinfer_fp32(batch, seq_len)
                    tput = batch / (lat / 1e6)
                    results.append(BenchResult("decode", "fi_fp32", batch, seq_len, "", lat, tput))
                    log.info(f"  FI FP32     B={batch} L={seq_len}: {lat:.0f} us ({tput:.0f} tok/s)")

                    # FlashInfer BF16
                    lat = bench_decode_flashinfer_bf16(batch, seq_len)
                    tput = batch / (lat / 1e6)
                    results.append(BenchResult("decode", "fi_bf16", batch, seq_len, "", lat, tput))
                    log.info(f"  FI BF16     B={batch} L={seq_len}: {lat:.0f} us ({tput:.0f} tok/s)")

        # MTP decode (speculative)
        if fi_available:
            log.info("=== MTP DECODE BENCHMARK ===")
            for batch in [1, 4, 8, 16, 32]:
                for T in [2, 4]:
                    lat = bench_decode_flashinfer_mtp(batch, T, 1024)
                    tput = (batch * T) / (lat / 1e6)
                    results.append(BenchResult("decode_mtp", "fi_mtp", batch, 1024, f"T={T}", lat, tput))
                    log.info(f"  FI MTP      B={batch} T={T}: {lat:.0f} us ({tput:.0f} tok/s)")

    # ---- Conv1d ----
    if "conv1d" in scenarios or "all" in scenarios:
        log.info("=== CONV1D UPDATE BENCHMARK ===")
        for batch in [1, 4, 8, 16, 32, 64, 128, 256]:
            lat = bench_conv1d_update(batch)
            results.append(BenchResult("conv1d", "triton", batch, 1, "", lat, batch / (lat / 1e6)))
            log.info(f"  Conv1d      B={batch}: {lat:.0f} us")

    # ---- Output ----
    log.info("=" * 80)
    log.info(f"Total results: {len(results)}")

    if output_file:
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scenario", "backend", "batch_size", "seq_len", "extra", "latency_us", "throughput_tok_per_s"])
            for r in results:
                writer.writerow([r.scenario, r.backend, r.batch_size, r.seq_len, r.extra, f"{r.latency_us:.1f}", f"{r.throughput_tok_per_s:.0f}"])
        log.info(f"Results written to {output_file}")

    # Print summary table
    _print_summary(results)
    return results


def _print_summary(results: List[BenchResult]):
    """Print a readable summary table."""
    if not results:
        return

    print("\n" + "=" * 100)
    print(f"{'Scenario':<20} {'Backend':<15} {'Batch':<8} {'SeqLen':<8} {'Extra':<25} {'Latency(us)':<15} {'Throughput(tok/s)':<18}")
    print("-" * 100)
    for r in results:
        print(f"{r.scenario:<20} {r.backend:<15} {r.batch_size:<8} {r.seq_len:<8} {r.extra:<25} {r.latency_us:<15.1f} {r.throughput_tok_per_s:<18.0f}")
    print("=" * 100)

    # Print speedup summary for decode
    decode_results = [r for r in results if r.scenario == "decode"]
    if decode_results:
        print("\n--- Decode Speedup Summary ---")
        fla_by_key = {(r.batch_size, r.seq_len): r.latency_us for r in decode_results if r.backend == "fla_split"}
        for backend in ["fi_fp32", "fi_bf16"]:
            fi_results = [(r.batch_size, r.seq_len, r.latency_us) for r in decode_results if r.backend == backend]
            if fi_results:
                speedups = []
                for b, s, lat in fi_results:
                    fla_lat = fla_by_key.get((b, s))
                    if fla_lat:
                        speedups.append(fla_lat / lat)
                if speedups:
                    avg_speedup = sum(speedups) / len(speedups)
                    print(f"  {backend} vs fla_split: avg {avg_speedup:.2f}x speedup (range {min(speedups):.2f}x - {max(speedups):.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="GDN kernel benchmark for Qwen3.5")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "prefill", "chunked_prefill", "decode", "conv1d"],
                        help="Which scenario to benchmark")
    parser.add_argument("--skip-flashinfer", action="store_true",
                        help="Skip FlashInfer benchmarks (for SM8x GPUs)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario != "all" else ["all"]

    with torch.no_grad():
        run_benchmarks(scenarios, args.skip_flashinfer, args.output)


if __name__ == "__main__":
    main()
