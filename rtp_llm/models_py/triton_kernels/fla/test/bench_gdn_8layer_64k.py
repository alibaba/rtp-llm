"""
Benchmark: 8-layer GDN prefill @ 64K tokens (TP2 Qwen3.5-35B config).
Measures full gating + chunk_gated_delta_rule pipeline per layer,
run 8 times sequentially to simulate 8 linear attention layers.
Also produces a Chrome trace JSON for timeline analysis.
"""

import json
import os
import time

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating

DEVICE = "cuda"
NUM_LAYERS = 8
SEQ_LEN = 65536
NUM_K_HEADS = 8
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
DTYPE = torch.bfloat16
WARMUP = 5
REPEAT = 20


def make_layer_inputs():
    qkv_dim = NUM_K_HEADS * HEAD_K_DIM * 2 + NUM_V_HEADS * HEAD_V_DIM
    mixed_qkv = torch.randn(SEQ_LEN, qkv_dim, dtype=DTYPE, device=DEVICE)
    A_log = torch.randn(NUM_V_HEADS, dtype=DTYPE, device=DEVICE)
    a = torch.randn(SEQ_LEN, NUM_V_HEADS, dtype=DTYPE, device=DEVICE)
    b = torch.randn(SEQ_LEN, NUM_V_HEADS, dtype=DTYPE, device=DEVICE)
    dt_bias = torch.randn(NUM_V_HEADS, dtype=DTYPE, device=DEVICE)
    cu_seqlens = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=DEVICE)
    initial_state = torch.randn(
        1,
        NUM_V_HEADS,
        HEAD_V_DIM,
        HEAD_K_DIM,
        dtype=torch.float32,
        device=DEVICE,
    )
    return mixed_qkv, A_log, a, b, dt_bias, cu_seqlens, initial_state


def run_one_layer(mixed_qkv, A_log, a, b, dt_bias, cu_seqlens, initial_state):
    g, beta = fused_gdn_gating(A_log, a, b, dt_bias)

    query, key, value = torch.split(
        mixed_qkv,
        [NUM_K_HEADS * HEAD_K_DIM, NUM_K_HEADS * HEAD_K_DIM, NUM_V_HEADS * HEAD_V_DIM],
        dim=-1,
    )
    query = query.view(1, query.shape[0], NUM_K_HEADS, HEAD_K_DIM)
    key = key.view(1, key.shape[0], NUM_K_HEADS, HEAD_K_DIM)
    value = value.view(1, value.shape[0], NUM_V_HEADS, HEAD_V_DIM)

    attn_out, h, final_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True,
    )
    return attn_out


def run_8_layers(all_inputs):
    for i in range(NUM_LAYERS):
        run_one_layer(*all_inputs[i])


def main():
    print(f"=== GDN 8-layer 64K Prefill Benchmark ===")
    print(
        f"seq_len={SEQ_LEN}, layers={NUM_LAYERS}, h_k={NUM_K_HEADS}, h_v={NUM_V_HEADS}, d={HEAD_K_DIM}"
    )
    print(f"dtype={DTYPE}, warmup={WARMUP}, repeat={REPEAT}")

    torch.manual_seed(42)
    all_inputs = [make_layer_inputs() for _ in range(NUM_LAYERS)]

    # Warmup
    print("Warming up...")
    for _ in range(WARMUP):
        run_8_layers(all_inputs)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    print("Benchmarking...")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(REPEAT)]

    for i in range(REPEAT):
        start_events[i].record()
        run_8_layers(all_inputs)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    trim = len(times) // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    avg_ms = sum(trimmed) / len(trimmed)
    per_layer = avg_ms / NUM_LAYERS

    print(f"\n--- Results (8 layers total) ---")
    print(
        f"  Total:     avg={avg_ms:.3f}ms  min={min(times):.3f}ms  max={max(times):.3f}ms"
    )
    print(f"  Per-layer: avg={per_layer:.3f}ms")
    print(f"  p50={times[len(times)//2]:.3f}ms  p90={times[int(len(times)*0.9)]:.3f}ms")

    # Profile with torch profiler and export Chrome trace
    trace_path = "/home/wangyin.yx/workspace/RTP-LLM/gdn_8layer_64k_trace.json"
    print(f"\nRecording profiler trace -> {trace_path}")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        run_8_layers(all_inputs)
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    print(f"Trace saved: {trace_path}")

    # Print kernel summary
    print("\n--- Top 30 CUDA Kernels ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
