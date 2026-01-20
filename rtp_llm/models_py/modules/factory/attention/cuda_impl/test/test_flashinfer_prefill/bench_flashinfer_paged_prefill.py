# -*- coding: utf-8 -*-
"""
Benchmark FlashInfer Paged KV Cache Prefill Performance

This benchmark compares the performance of different FlashInfer backends for paged KV cache:
1. fa2 (FlashAttention 2)
2. fa3 (FlashAttention 3, SM90+)
3. fa2 with custom mask

The benchmark tests with padded Q and paged KV cache layout, which is typical for CUDA graph mode.
"""
import argparse
import logging
import statistics
from typing import List, Tuple

import torch
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.attention_ref import (
    compute_flashinfer_prefill_reference,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.test.bench_utils import (
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time_with_cuda_event,
    set_seed,
)

logging.basicConfig(level=print, format="%(message)s")


def create_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lens: List[int],
    page_size: int,
    kv_heads: int,
    head_dim: int,
    device: torch.device,
    num_layers: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert ragged K, V to paged KV cache format.

    Args:
        k: [total_tokens, kv_heads, head_dim]
        v: [total_tokens, kv_heads, head_dim]
        seq_lens: List of sequence lengths
        page_size: Page size for KV cache
        kv_heads: Number of KV heads
        head_dim: Head dimension
        device: Device
        num_layers: Number of layers (default 1)

    Returns:
        paged_kv_cache: [num_layers, num_pages, 2, page_size, kv_heads, head_dim]
        paged_kv_indptr: [batch_size + 1]
        paged_kv_indices: [total_pages]
        paged_kv_last_page_len: [batch_size]
    """
    # Calculate total pages needed
    total_pages = sum((seq_len + page_size - 1) // page_size for seq_len in seq_lens)

    # Allocate paged KV cache with num_layers dimension
    paged_kv_cache = torch.zeros(
        num_layers,
        total_pages,
        2,
        page_size,
        kv_heads,
        head_dim,
        dtype=k.dtype,
        device=device,
    )

    paged_kv_indptr = [0]
    paged_kv_indices = []
    paged_kv_last_page_len = []

    page_idx = 0
    token_offset = 0

    for seq_len in seq_lens:
        num_pages = (seq_len + page_size - 1) // page_size
        last_page_len = seq_len % page_size if seq_len % page_size != 0 else page_size

        # Fill pages with K, V data (for layer 0 only in this benchmark)
        for i in range(num_pages):
            start_token = i * page_size
            end_token = min(start_token + page_size, seq_len)
            num_tokens_in_page = end_token - start_token

            # Copy K, V to page (layer 0)
            paged_kv_cache[0, page_idx, 0, :num_tokens_in_page] = k[
                token_offset + start_token : token_offset + end_token
            ]
            paged_kv_cache[0, page_idx, 1, :num_tokens_in_page] = v[
                token_offset + start_token : token_offset + end_token
            ]

            paged_kv_indices.append(page_idx)
            page_idx += 1

        paged_kv_indptr.append(paged_kv_indptr[-1] + num_pages)
        paged_kv_last_page_len.append(last_page_len)
        token_offset += seq_len

    return (
        paged_kv_cache,
        torch.tensor(paged_kv_indptr, dtype=torch.int32, device=device),
        torch.tensor(paged_kv_indices, dtype=torch.int32, device=device),
        torch.tensor(paged_kv_last_page_len, dtype=torch.int32, device=device),
    )


def create_custom_mask_for_padded_q(
    seq_lens: List[int], max_seq_len: int, device: torch.device
) -> torch.Tensor:
    """
    Create custom mask for padded Q scenario: [max_seq_len, real_len] per batch.

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
        # Create causal mask: [max_seq_len, real_len]
        mask = torch.zeros(max_seq_len, real_len, dtype=torch.bool, device=device)
        for j in range(real_len):
            mask[j, : j + 1] = True
        mask_list.append(mask.flatten())

    return torch.cat(mask_list, dim=0)


def benchmark_paged_prefill(
    batch_size: int,
    seq_lens: List[int],
    num_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    backend: str,
    use_custom_mask: bool,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    warmup_iters: int = 10,
    repeat_iters: int = 100,
    q_ragged: torch.Tensor = None,  # For non-custom-mask configs
    q_padded: torch.Tensor = None,  # For custom-mask config
    paged_kv_cache: torch.Tensor = None,
    paged_kv_indptr: torch.Tensor = None,
    paged_kv_indices: torch.Tensor = None,
    paged_kv_last_page_len: torch.Tensor = None,
) -> Tuple[float, float, torch.Tensor]:
    """
    Benchmark paged prefill with specified configuration.

    Returns:
        (mean_time_ms, tflops_per_sec, output_ragged)
        Note: output_ragged is always in ragged format for accuracy comparison
    """
    set_seed(42)

    max_seq_len = max(seq_lens)
    total_real_tokens = sum(seq_lens)
    total_padded_tokens = batch_size * max_seq_len

    # Determine if using padded Q or ragged Q based on custom_mask
    if use_custom_mask:
        # For custom_mask, Q must be padded
        if q_padded is None:
            q_padded = torch.randn(
                total_padded_tokens, num_heads, head_dim, dtype=dtype, device=device
            )
        q_to_use = q_padded
        # qo_indptr for padded Q
        qo_indptr = torch.arange(
            0,
            (batch_size + 1) * max_seq_len,
            max_seq_len,
            dtype=torch.int32,
            device=device,
        )
    else:
        # For non-custom_mask (causal), use ragged Q
        if q_ragged is None:
            # If no Q provided, create ragged Q
            q_ragged = torch.randn(
                total_real_tokens, num_heads, head_dim, dtype=dtype, device=device
            )
        q_to_use = q_ragged
        # qo_indptr for ragged Q (cumulative sequence lengths)
        qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
        )

    if paged_kv_cache is None:
        # Create ragged K, V
        k_ragged = torch.randn(
            total_real_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )
        v_ragged = torch.randn(
            total_real_tokens, kv_heads, head_dim, dtype=dtype, device=device
        )

        # Convert to paged KV cache
        paged_kv_cache, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len = (
            create_paged_kv_cache(
                k_ragged, v_ragged, seq_lens, page_size, kv_heads, head_dim, device
            )
        )

    # Create wrapper
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        float_workspace_buffer=workspace_buffer, kv_layout="NHD", backend=backend
    )

    # Prepare custom mask if needed
    custom_mask = None
    if use_custom_mask:
        custom_mask = create_custom_mask_for_padded_q(seq_lens, max_seq_len, device)
        # Plan with custom mask (causal=False when using custom mask)
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=False,
            custom_mask=custom_mask,
            q_data_type=dtype,
        )
    else:
        # Plan with causal attention
        wrapper.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_heads,
            kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=dtype,
        )

    # For paged KV cache, we need to pass kv_cache for layer 0 (without layer dimension)
    # paged_kv_cache shape: [num_layers, num_pages, 2, page_size, kv_heads, head_dim]
    # wrapper.run expects: [num_pages, 2, page_size, kv_heads, head_dim]
    kv_cache_layer0 = paged_kv_cache[0]  # Extract layer 0

    # Define forward function
    output = None  # Will be set in first forward call

    def forward_fn():
        nonlocal output
        output = wrapper.run(q_to_use, kv_cache_layer0)

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

    # Convert output to ragged format for accuracy comparison
    if use_custom_mask:
        # Output is padded, extract ragged part
        output_ragged_list: List[torch.Tensor] = []
        for i in range(batch_size):
            start_idx = i * max_seq_len
            real_len = seq_lens[i]
            output_ragged_list.append(output[start_idx : start_idx + real_len])
        output_ragged = torch.cat(output_ragged_list, dim=0)
    else:
        # Output is already ragged
        output_ragged = output.clone()

    return mean_time, tflops_per_sec, output_ragged


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlashInfer Paged Prefill Performance"
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
        "--page-size", type=int, default=16, help="Page size for KV cache"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--repeat-iters", type=int, default=100, help="Repeat iterations"
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
    print("FlashInfer Paged KV Cache Prefill Benchmark")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Max sequence length: {max(args.seq_lens)}")
    print(f"Num heads: {args.num_heads}")
    print(f"KV heads: {args.kv_heads}")
    print(f"Head dim: {args.head_dim}")
    print(f"Page size: {args.page_size}")
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

    # Prepare shared input data for all configurations (for accuracy comparison)
    set_seed(42)
    max_seq_len = max(args.seq_lens)
    total_real_tokens = sum(args.seq_lens)
    total_padded_tokens = args.batch_size * max_seq_len

    print("Preparing shared input data for accuracy comparison...")
    # Create ragged Q (for non-custom-mask configs)
    q_ragged_shared = torch.randn(
        total_real_tokens, args.num_heads, args.head_dim, dtype=dtype, device=device
    )

    # Create padded Q (for custom-mask config) from ragged Q
    q_padded_shared = torch.zeros(
        total_padded_tokens, args.num_heads, args.head_dim, dtype=dtype, device=device
    )
    offset = 0
    for i in range(args.batch_size):
        start_idx = i * max_seq_len
        real_len = args.seq_lens[i]
        q_padded_shared[start_idx : start_idx + real_len] = q_ragged_shared[
            offset : offset + real_len
        ]
        offset += real_len

    k_ragged_shared = torch.randn(
        total_real_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device
    )
    v_ragged_shared = torch.randn(
        total_real_tokens, args.kv_heads, args.head_dim, dtype=dtype, device=device
    )

    (
        paged_kv_cache_shared,
        paged_kv_indptr_shared,
        paged_kv_indices_shared,
        paged_kv_last_page_len_shared,
    ) = create_paged_kv_cache(
        k_ragged_shared,
        v_ragged_shared,
        args.seq_lens,
        args.page_size,
        args.kv_heads,
        args.head_dim,
        device,
    )

    # Compute reference output using FlashInfer's reference implementation
    print("Computing reference output (ground truth)...")
    cu_seqlens = torch.zeros(args.batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(args.seq_lens, dtype=torch.int32, device=device), dim=0
    )

    # Reference output is in ragged format (no padding)
    reference_output_ragged = compute_flashinfer_prefill_reference(
        q_ragged_shared, k_ragged_shared, v_ragged_shared, cu_seqlens, causal=True
    )

    print("")

    results = []
    outputs = {}  # Store outputs for accuracy comparison (padded Q configs)

    # Benchmark fa2 (ragged Q)
    if "fa2" in backends_to_test:
        print("Benchmarking: BatchPrefillPagedWrapper + fa2 (causal, ragged Q)")
        mean_time, tflops, _ = benchmark_paged_prefill(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            backend="fa2",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_ragged=q_ragged_shared,
            paged_kv_cache=paged_kv_cache_shared,
            paged_kv_indptr=paged_kv_indptr_shared,
            paged_kv_indices=paged_kv_indices_shared,
            paged_kv_last_page_len=paged_kv_last_page_len_shared,
        )
        results.append(("fa2 (causal, ragged Q)", mean_time, tflops))
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa3 (ragged Q)
    if "fa3" in backends_to_test:
        print("Benchmarking: BatchPrefillPagedWrapper + fa3 (causal, ragged Q)")
        mean_time, tflops, _ = benchmark_paged_prefill(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            backend="fa3",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_ragged=q_ragged_shared,
            paged_kv_cache=paged_kv_cache_shared,
            paged_kv_indptr=paged_kv_indptr_shared,
            paged_kv_indices=paged_kv_indices_shared,
            paged_kv_last_page_len=paged_kv_last_page_len_shared,
        )
        results.append(("fa3 (causal, ragged Q)", mean_time, tflops))
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa2 (padded Q, causal, no custom mask)
    if "fa2" in backends_to_test:
        print("Benchmarking: BatchPrefillPagedWrapper + fa2 (causal, padded Q)")
        mean_time, tflops, output = benchmark_paged_prefill(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            backend="fa2",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            paged_kv_cache=paged_kv_cache_shared,
            paged_kv_indptr=paged_kv_indptr_shared,
            paged_kv_indices=paged_kv_indices_shared,
            paged_kv_last_page_len=paged_kv_last_page_len_shared,
        )
        results.append(("fa2 (causal, padded Q)", mean_time, tflops))
        outputs["fa2 (causal, padded Q)"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa3 (padded Q, causal, no custom mask)
    if "fa3" in backends_to_test:
        print("Benchmarking: BatchPrefillPagedWrapper + fa3 (causal, padded Q)")
        mean_time, tflops, output = benchmark_paged_prefill(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            backend="fa3",
            use_custom_mask=False,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            paged_kv_cache=paged_kv_cache_shared,
            paged_kv_indptr=paged_kv_indptr_shared,
            paged_kv_indices=paged_kv_indices_shared,
            paged_kv_last_page_len=paged_kv_last_page_len_shared,
        )
        results.append(("fa3 (causal, padded Q)", mean_time, tflops))
        outputs["fa3 (causal, padded Q)"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")

    # Benchmark fa2 with custom mask (padded Q)
    print("Benchmarking: BatchPrefillPagedWrapper + fa2 (padded Q + custom mask)")
    try:
        mean_time, tflops, output = benchmark_paged_prefill(
            args.batch_size,
            args.seq_lens,
            args.num_heads,
            args.kv_heads,
            args.head_dim,
            args.page_size,
            backend="fa2",
            use_custom_mask=True,
            device=device,
            dtype=dtype,
            warmup_iters=args.warmup_iters,
            repeat_iters=args.repeat_iters,
            q_padded=q_padded_shared,
            paged_kv_cache=paged_kv_cache_shared,
            paged_kv_indptr=paged_kv_indptr_shared,
            paged_kv_indices=paged_kv_indices_shared,
            paged_kv_last_page_len=paged_kv_last_page_len_shared,
        )
        results.append(("fa2 (padded Q + custom mask)", mean_time, tflops))
        outputs["fa2 (padded Q + custom mask)"] = output
        print(f"  Mean time: {mean_time:.3f} ms")
        print(f"  TFLOPs/s: {tflops:.2f}")
        print("")
    except RuntimeError as e:
        if "operation not supported" in str(e):
            logging.warning(f"  Custom mask not supported on this GPU: {e}")
            print("")
        else:
            raise

    # Accuracy comparison (for all padded Q configs)
    if outputs:
        print("=" * 80)
        print("Accuracy Comparison (vs FlashInfer Reference Implementation)")
        print("=" * 80)
        print("Reference: compute_flashinfer_prefill_reference (ground truth)")
        print(
            "Note: Verifying padded Q configurations (padded outputs sliced to real lengths)"
        )
        print("")

        for config_name, output_ragged in outputs.items():
            # Compare ragged outputs (only valid tokens, no padding)
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

            # Use absolute error as pass criteria for FP16
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
        print("")

    # Performance Summary
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Time (ms)':<15} {'TFLOPs/s':<15} {'Speedup':<10}")
    print("-" * 80)

    baseline_time = results[0][1] if results else 1.0
    for config, mean_time, tflops in results:
        speedup = baseline_time / mean_time
        print(f"{config:<30} {mean_time:<15.3f} {tflops:<15.2f} {speedup:<10.2f}x")

    print("=" * 80)


if __name__ == "__main__":
    main()
