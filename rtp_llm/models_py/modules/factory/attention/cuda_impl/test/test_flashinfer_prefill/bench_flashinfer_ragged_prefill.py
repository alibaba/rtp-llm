# -*- coding: utf-8 -*-
"""
Benchmark FlashInfer Ragged KV Cache Prefill Performance

This benchmark compares the performance of different FlashInfer backends for ragged KV cache:
1. fa2 (FlashAttention 2)
2. fa3 (FlashAttention 3, SM90+)
3. fa2 with custom mask

The benchmark tests with padded Q, K, V layout (for CUDA graph mode) and ragged Q, K, V layout.
"""
import argparse
import logging
import statistics
from typing import List, Tuple

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench_utils import (
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time_with_cuda_event,
    set_seed,
)

logging.basicConfig(level=print, format="%(message)s")


def create_custom_mask_for_padded_qkv(
    seq_lens: List[int], max_seq_len: int, device: torch.device
) -> torch.Tensor:
    """
    Create custom mask for padded Q, K, V scenario: [max_seq_len, max_seq_len] per batch.

    Args:
        seq_lens: List of real sequence lengths
        max_seq_len: Maximum sequence length (padded)
        device: Device

    Returns:
        custom_mask: Flattened custom mask
    """
    batch_size = len(seq_lens)
    mask_list: List[torch.Tensor] = []

    for i in range(batch_size):
        real_len = seq_lens[i]
        # Create causal mask: [max_seq_len, max_seq_len]
        mask = torch.zeros(max_seq_len, max_seq_len, dtype=torch.bool, device=device)
        for j in range(real_len):
            mask[j, : j + 1] = True
        mask_list.append(mask.flatten())

    return torch.cat(mask_list, dim=0)


def benchmark_ragged_prefill_padded(
    batch_size: int,
    seq_lens: List[int],
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    backend: str,
    use_custom_mask: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 10,
    repeat_iters: int = 100,
    q_padded: torch.Tensor = None,  # For shared input
    k_padded: torch.Tensor = None,  # For shared input
    v_padded: torch.Tensor = None,  # For shared input
) -> Tuple[float, float, torch.Tensor]:
    """
    Benchmark ragged prefill with PADDED Q, K, V (CUDA graph mode).

    Returns:
        (mean_time_ms, tflops_per_sec, output_ragged)
        Note: output_ragged is always in ragged format for accuracy comparison
    """
    max_seq_len = max(seq_lens)
    total_padded_tokens = batch_size * max_seq_len

    # Create padded Q, K, V if not provided
    if q_padded is None:
        set_seed(42)
        q_padded = torch.randn(
            total_padded_tokens, num_heads, head_dim, dtype=dtype, device=device
        )
    if k_padded is None:
        k_padded = torch.randn(
            total_padded_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )
    if v_padded is None:
        v_padded = torch.randn(
            total_padded_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )

    # Create qo_indptr and kv_indptr (both padded)
    qo_indptr = torch.arange(
        0, (batch_size + 1) * max_seq_len, max_seq_len, dtype=torch.int32, device=device
    )
    kv_indptr = qo_indptr.clone()

    # Allocate output
    output = torch.empty_like(q_padded)

    # Create wrapper
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer=workspace_buffer, kv_layout="NHD", backend=backend
    )

    # Prepare custom mask if needed
    custom_mask = None
    if use_custom_mask:
        custom_mask = create_custom_mask_for_padded_qkv(seq_lens, max_seq_len, device)
        # Plan with custom mask (causal=False when using custom mask)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_heads,
            kv_heads,
            head_dim,
            head_dim,
            causal=False,
            custom_mask=custom_mask,
            q_data_type=dtype,
        )
    else:
        # Plan with causal attention
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_heads,
            kv_heads,
            head_dim,
            head_dim,
            causal=True,
            q_data_type=dtype,
        )

    # Define forward function
    output = None  # Will be set in first forward call

    def forward_fn():
        nonlocal output
        output = wrapper.run(q_padded, k_padded, v_padded)

    # Warmup
    for _ in range(warmup_iters):
        forward_fn()
    torch.cuda.synchronize()

    # Benchmark
    measured_times = bench_gpu_time_with_cuda_event(
        forward_fn,
        dry_run_iters=10,
        repeat_iters=repeat_iters,
        l2_flush=True,
    )

    mean_time = statistics.mean(measured_times)

    # Calculate TFLOPs/s (use real sequence lengths for accurate FLOP count)
    actual_seq_lens_q = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    tflops_per_sec = attention_tflops_per_sec_with_actual_seq_lens(
        actual_seq_lens_q,
        actual_seq_lens_kv,
        head_dim,
        head_dim,
        num_heads,
        causal=True,
        ms=mean_time,
    )

    # Convert output to ragged format for accuracy comparison
    # Output is padded, extract ragged part
    output_ragged_list: List[torch.Tensor] = []
    for i in range(batch_size):
        start_idx = i * max_seq_len
        real_len = seq_lens[i]
        output_ragged_list.append(output[start_idx : start_idx + real_len])
    output_ragged = torch.cat(output_ragged_list, dim=0)

    return mean_time, tflops_per_sec, output_ragged


def benchmark_ragged_prefill_ragged(
    batch_size: int,
    seq_lens: List[int],
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    backend: str,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 10,
    repeat_iters: int = 100,
    q_ragged: torch.Tensor = None,  # For shared input
    k_ragged: torch.Tensor = None,  # For shared input
    v_ragged: torch.Tensor = None,  # For shared input
) -> Tuple[float, float, torch.Tensor]:
    """
    Benchmark ragged prefill with RAGGED Q, K, V (no padding, normal mode).

    Returns:
        (mean_time_ms, tflops_per_sec, output_ragged)
    """
    total_tokens = sum(seq_lens)

    # Create ragged Q, K, V if not provided
    if q_ragged is None:
        set_seed(42)
        q_ragged = torch.randn(
            total_tokens, num_heads, head_dim, dtype=dtype, device=device
        )
    if k_ragged is None:
        k_ragged = torch.randn(
            total_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )
    if v_ragged is None:
        v_ragged = torch.randn(
            total_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )

    # Create qo_indptr and kv_indptr
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    qo_indptr = cu_seqlens
    kv_indptr = cu_seqlens

    # Create wrapper
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer=workspace_buffer, kv_layout="NHD", backend=backend
    )

    # Plan with causal attention (no custom mask for ragged case)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_heads,
        kv_heads,
        head_dim,
        head_dim,
        causal=True,
        q_data_type=dtype,
    )

    # Define forward function
    output = None  # Will be set in first forward call

    def forward_fn():
        nonlocal output
        output = wrapper.run(q_ragged, k_ragged, v_ragged)

    # Warmup
    for _ in range(warmup_iters):
        forward_fn()
    torch.cuda.synchronize()

    # Benchmark
    measured_times = bench_gpu_time_with_cuda_event(
        forward_fn,
        dry_run_iters=10,
        repeat_iters=repeat_iters,
        l2_flush=True,
    )

    mean_time = statistics.mean(measured_times)

    # Calculate TFLOPs/s
    actual_seq_lens_q = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    actual_seq_lens_kv = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    tflops_per_sec = attention_tflops_per_sec_with_actual_seq_lens(
        actual_seq_lens_q,
        actual_seq_lens_kv,
        head_dim,
        head_dim,
        num_heads,
        causal=True,
        ms=mean_time,
    )

    # Output is already ragged, return directly
    return mean_time, tflops_per_sec, output.clone()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlashInfer Ragged Prefill Performance"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Sequence lengths for each request",
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of query heads"
    )
    parser.add_argument(
        "--kv-heads", type=int, default=8, help="Number of KV heads (GQA)"
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension")
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--repeat-iters", type=int, default=100, help="Repeat iterations"
    )
    parser.add_argument(
        "--skip-ragged",
        action="store_true",
        help="Skip ragged (non-padded) benchmark",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logging.error("CUDA is not available")
        return

    device = torch.device("cuda")
    dtype = torch.float16

    # Ensure seq_lens matches batch_size
    if len(args.seq_lens) != args.batch_size:
        logging.warning(
            f"seq_lens length ({len(args.seq_lens)}) does not match batch_size ({args.batch_size}), "
            f"adjusting seq_lens..."
        )
        if len(args.seq_lens) < args.batch_size:
            args.seq_lens = args.seq_lens * (args.batch_size // len(args.seq_lens) + 1)
        args.seq_lens = args.seq_lens[: args.batch_size]

    print("=" * 80)
    print("FlashInfer Ragged KV Cache Prefill Benchmark")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Max sequence length: {max(args.seq_lens)}")
    print(f"Total tokens: {sum(args.seq_lens)}")
    print(f"Num heads: {args.num_heads}")
    print(f"KV heads: {args.kv_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("=" * 80)

    # Get GPU info
    gpu_name = torch.cuda.get_device_name()
    compute_capability = torch.cuda.get_device_capability()
    print(f"GPU: {gpu_name}")
    print(f"Compute Capability: SM{compute_capability[0]}{compute_capability[1]}")
    print("=" * 80)

    # Determine available backends
    backends_to_test = ["fa2"]
    if compute_capability[0] >= 9:  # SM90+
        backends_to_test.append("fa3")

    results = []

    # Prepare shared input data for accuracy comparison
    print("\nPreparing shared input data for accuracy comparison...")
    set_seed(42)

    # Create ragged Q, K, V for reference
    total_tokens = sum(args.seq_lens)
    q_ragged_shared = torch.randn(
        total_tokens, args.num_heads, args.head_dim, dtype=dtype, device=device
    )
    k_ragged_shared = torch.randn(
        total_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device
    )
    v_ragged_shared = torch.randn(
        total_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device
    )

    # Create padded Q, K, V from ragged (for padded mode benchmarks)
    max_seq_len = max(args.seq_lens)
    q_padded_shared = torch.zeros(
        args.batch_size * max_seq_len,
        args.num_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    k_padded_shared = torch.zeros(
        args.batch_size * max_seq_len,
        args.kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )
    v_padded_shared = torch.zeros(
        args.batch_size * max_seq_len,
        args.kv_heads,
        args.head_dim,
        dtype=dtype,
        device=device,
    )

    # Fill padded tensors with ragged data
    token_offset = 0
    for i in range(args.batch_size):
        seq_len = args.seq_lens[i]
        padded_start = i * max_seq_len
        q_padded_shared[padded_start : padded_start + seq_len] = q_ragged_shared[
            token_offset : token_offset + seq_len
        ]
        k_padded_shared[padded_start : padded_start + seq_len] = k_ragged_shared[
            token_offset : token_offset + seq_len
        ]
        v_padded_shared[padded_start : padded_start + seq_len] = v_ragged_shared[
            token_offset : token_offset + seq_len
        ]
        token_offset += seq_len

    # Compute reference output (ground truth)
    print("Computing reference output (ground truth)...")
    cu_seqlens = torch.zeros(args.batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(args.seq_lens, dtype=torch.int32, device=device), dim=0
    )

    reference_output_ragged = compute_flashinfer_prefill_reference(
        q_ragged_shared, k_ragged_shared, v_ragged_shared, cu_seqlens, causal=True
    )
    print("")

    # Store outputs for accuracy comparison (only padded mode needs validation)
    accuracy_outputs_padded = {}

    # ========== Padded QKV Benchmarks (CUDA Graph Mode) ==========
    print("\n" + "=" * 80)
    print("Padded Q, K, V Benchmarks (CUDA Graph Mode)")
    print("=" * 80)

    # Benchmark fa2 (padded)
    if "fa2" in backends_to_test:
        print("Benchmarking: BatchPrefillRaggedWrapper + fa2 (causal, padded QKV)")
        mean_time, tflops, output = benchmark_ragged_prefill_padded(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            backend="fa2",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            k_padded=k_padded_shared,
            v_padded=v_padded_shared,
        )
        results.append(("fa2 (causal, padded)", mean_time, tflops))
        accuracy_outputs_padded["fa2 (causal)"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa3 (padded)
    if "fa3" in backends_to_test:
        print("Benchmarking: BatchPrefillRaggedWrapper + fa3 (causal, padded QKV)")
        mean_time, tflops, output = benchmark_ragged_prefill_padded(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            backend="fa3",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            k_padded=k_padded_shared,
            v_padded=v_padded_shared,
        )
        results.append(("fa3 (causal, padded)", mean_time, tflops))
        accuracy_outputs_padded["fa3 (causal)"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa2 with custom mask (padded)
    print("Benchmarking: BatchPrefillRaggedWrapper + fa2 + custom mask (padded QKV)")
    try:
        mean_time, tflops, output = benchmark_ragged_prefill_padded(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            backend="fa2",
            use_custom_mask=True,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            k_padded=k_padded_shared,
            v_padded=v_padded_shared,
        )
        results.append(("fa2 + custom mask (padded)", mean_time, tflops))
        accuracy_outputs_padded["fa2 + custom mask"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")
    except RuntimeError as e:
        if "operation not supported" in str(e):
            logging.warning(f"  Custom mask not supported on this GPU: {e}")
            print("")
        else:
            raise

    # ========== Ragged QKV Benchmarks (Normal Mode) ==========
    if not args.skip_ragged:
        print("\n" + "=" * 80)
        print("Ragged Q, K, V Benchmarks (Normal Mode, No Padding)")
        print("=" * 80)

        # Benchmark fa2 (ragged)
        if "fa2" in backends_to_test:
            print("Benchmarking: BatchPrefillRaggedWrapper + fa2 (causal, ragged QKV)")
            mean_time, tflops, _ = benchmark_ragged_prefill_ragged(
                args.batch_size,
                args.seq_lens,
                args.num_heads,
                args.kv_heads,
                args.head_dim,
                backend="fa2",
                device=device,
                dtype=dtype,
                warmup_iters=args.warmup_iters,
                repeat_iters=args.repeat_iters,
                q_ragged=q_ragged_shared,
                k_ragged=k_ragged_shared,
                v_ragged=v_ragged_shared,
            )
            results.append(("fa2 (causal, ragged)", mean_time, tflops))
            print(f"  Mean time: {mean_time:.3f} ms")
            print(f"  TFLOPs/s: {tflops:.2f}")
            print("")

        # Benchmark fa3 (ragged)
        if "fa3" in backends_to_test:
            print("Benchmarking: BatchPrefillRaggedWrapper + fa3 (causal, ragged QKV)")
            mean_time, tflops, _ = benchmark_ragged_prefill_ragged(
                args.batch_size,
                args.seq_lens,
                args.num_heads,
                args.kv_heads,
                args.head_dim,
                backend="fa3",
                device=device,
                dtype=dtype,
                warmup_iters=args.warmup_iters,
                repeat_iters=args.repeat_iters,
                q_ragged=q_ragged_shared,
                k_ragged=k_ragged_shared,
                v_ragged=v_ragged_shared,
            )
            results.append(("fa3 (causal, ragged)", mean_time, tflops))
            print(f"  Mean time: {mean_time:.3f} ms")
            print(f"  TFLOPs/s: {tflops:.2f}")
            print("")

    # Accuracy Comparison (only for padded mode)
    if accuracy_outputs_padded:
        print("\n" + "=" * 80)
        print("Accuracy Comparison (vs FlashInfer Reference Implementation)")
        print("=" * 80)
        print("Reference: compute_flashinfer_prefill_reference (ground truth)")
        print("Note: Padded outputs are sliced to real lengths before comparison")
        print("")

        for config_name, output_ragged in accuracy_outputs_padded.items():
            max_diff = (output_ragged - reference_output_ragged).abs().max().item()
            mean_diff = (output_ragged - reference_output_ragged).abs().mean().item()

            # Find where max diff occurs
            abs_diff = (output_ragged - reference_output_ragged).abs()
            max_diff_idx = abs_diff.argmax()
            max_diff_ref_val = reference_output_ragged.flatten()[max_diff_idx].item()
            max_diff_out_val = output_ragged.flatten()[max_diff_idx].item()

            # Compute relative error for non-zero values
            mask = reference_output_ragged.abs() > 1e-6
            if mask.any():
                rel_diff = (
                    (output_ragged - reference_output_ragged).abs()
                    / (reference_output_ragged.abs() + 1e-8)
                )[mask]
                max_rel_diff = rel_diff.max().item()
                mean_rel_diff = rel_diff.mean().item()
            else:
                max_rel_diff = 0.0
                mean_rel_diff = 0.0

            # Adjust status based on absolute error for FP16
            status = "✅ PASS" if max_diff < 5e-3 else "⚠️  CHECK"
            print(f"{config_name}: {status}")
            print(
                f"  Max absolute diff:   {max_diff:.2e} (ref={max_diff_ref_val:.4f}, out={max_diff_out_val:.4f})"
            )
            print(f"  Mean absolute diff:  {mean_diff:.2e}")
            print(f"  Max relative diff:   {max_rel_diff:.2e}")
            print(f"  Mean relative diff:  {mean_rel_diff:.2e}")
            print("")

        print("=" * 80)

    # Performance Summary
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Time (ms)':<15} {'TFLOPs/s':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results[0][1] if results else 1.0
    for config, mean_time, tflops in results:
        speedup = baseline_time / mean_time
        print(f"{config:<35} {mean_time:<15.3f} {tflops:<15.2f} {speedup:<10.2f}x")

    print("=" * 80)


if __name__ == "__main__":
    main()
