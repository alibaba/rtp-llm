"""Benchmark: old 3-kernel pipeline vs new fused kkt+solve + exp2.

Production shapes from MI355X timeline (397B TP2, 64K prefill):
  B=1, T=65536, Hg=8 (k-heads/TP), H=32 (v-heads/TP), DK=DV=128, BT=64
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/rtp-llm")

# ── shapes ──────────────────────────────────────────────────────
B = 1
T = 65536  # 64K tokens → 1024 chunks
Hg = 8  # k/q heads per TP
H = 32  # v heads per TP
DK = 128
DV = 128
BT = 64
SCALE = DK**-0.5

RCP_LN2 = 1.0 / 0.6931471805599453

device = "cuda"
dtype = torch.bfloat16

print(f"Shape: B={B} T={T} Hg={Hg} H={H} DK={DK} DV={DV}")
print(f"Chunks: NT={T // BT}")
print()

# ── inputs ──────────────────────────────────────────────────────
torch.manual_seed(42)
q = torch.randn(B, T, Hg, DK, device=device, dtype=dtype)
k = F.normalize(torch.randn(B, T, Hg, DK, device=device, dtype=dtype), p=2, dim=-1)
v = torch.randn(B, T, H, DV, device=device, dtype=dtype)
g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()

from rtp_llm.models_py.triton_kernels.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_o import chunk_fwd_o

# ── old pipeline (3-kernel: kkt → solve → w_u) ─────────────────
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd


def old_pipeline(q, k, v, g, beta):
    g_cum = chunk_local_cumsum(g, chunk_size=BT)
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cum, output_dtype=torch.float32
    )
    A = solve_tril(A=A, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, g_cumsum=g_cum, cu_seqlens=None)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cum, output_final_state=False
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cum, scale=SCALE)
    return o, g_cum


# ── new pipeline (fused kkt+solve + exp2) ───────────────────────
from rtp_llm.models_py.triton_kernels.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)


def new_pipeline(q, k, v, g, beta):
    g_cum = chunk_local_cumsum(g, chunk_size=BT, scale=RCP_LN2)
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_cum, beta=beta, use_exp2=True
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g_cum, output_final_state=False, use_exp2=True
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g_cum, scale=SCALE, use_exp2=True)
    return o, g_cum


# ── precision check ─────────────────────────────────────────────
print("=== Precision Check ===")
torch.cuda.synchronize()
o_old, g_old = old_pipeline(q, k, v, g, beta)
torch.cuda.synchronize()
o_new, g_new = new_pipeline(q, k, v, g, beta)
torch.cuda.synchronize()

o_diff = (o_old.float() - o_new.float()).abs()
o_rel = o_diff / (o_old.float().abs() + 1e-8)
print(f"  O max_abs_diff: {o_diff.max().item():.6e}")
print(f"  O mean_abs_diff: {o_diff.mean().item():.6e}")
print(f"  O max_rel_err:  {o_rel.max().item():.6e}")
print(f"  O mean_rel_err: {o_rel.mean().item():.6e}")

g_diff = (g_old.float() - g_new.float()).abs()
g_rel = g_diff / (g_old.float().abs() + 1e-8)
print(f"  g_cum max_rel_err: {g_rel.max().item():.6e}  (exp vs exp2 base change)")
print()

PASS = o_rel.mean().item() < 1e-2
print(f"  Precision: {'PASS' if PASS else 'FAIL'} (mean_rel < 1e-2)")
print()

# ── benchmark ───────────────────────────────────────────────────
WARMUP = 5
REPEAT = 20


def bench(fn, name):
    for _ in range(WARMUP):
        fn(q, k, v, g, beta)
    torch.cuda.synchronize()

    times = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(q, k, v, g, beta)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us

    avg = sum(times) / len(times)
    mn = min(times)
    print(f"  {name:25s}  avg={avg:10.0f} us  min={mn:10.0f} us")
    return avg, mn


print("=== Performance Benchmark ===")
avg_old, min_old = bench(old_pipeline, "old (3-kernel)")
avg_new, min_new = bench(new_pipeline, "new (fused+exp2)")
print()
speedup_avg = avg_old / avg_new
speedup_min = min_old / min_new
print(f"  Speedup (avg): {speedup_avg:.3f}x")
print(f"  Speedup (min): {speedup_min:.3f}x")
print(f"  Savings (avg): {avg_old - avg_new:.0f} us/call")

# ── per-stage breakdown ─────────────────────────────────────────
print()
print("=== Per-Stage Breakdown ===")


def bench_stage(fn, name, reps=REPEAT):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    avg = sum(times) / len(times)
    print(f"  {name:35s}  avg={avg:8.0f} us")
    return avg


g_cum_old = chunk_local_cumsum(g, chunk_size=BT)
g_cum_new = chunk_local_cumsum(g, chunk_size=BT, scale=RCP_LN2)

# Old preprocessing stages
bench_stage(lambda: chunk_local_cumsum(g, chunk_size=BT), "OLD cumsum")
bench_stage(
    lambda: chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cum_old, output_dtype=torch.float32
    ),
    "OLD kkt",
)

A_old = chunk_scaled_dot_kkt_fwd(
    k=k, beta=beta, g_cumsum=g_cum_old, output_dtype=torch.float32
)
bench_stage(lambda: solve_tril(A=A_old, output_dtype=k.dtype), "OLD solve_tril")

A_solved = solve_tril(A=A_old, output_dtype=k.dtype)
bench_stage(
    lambda: recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A_solved, g_cumsum=g_cum_old, cu_seqlens=None
    ),
    "OLD recompute_w_u",
)

print()

# New preprocessing stages
bench_stage(
    lambda: chunk_local_cumsum(g, chunk_size=BT, scale=RCP_LN2), "NEW cumsum (scaled)"
)
bench_stage(
    lambda: chunk_gated_delta_rule_fwd_intra(
        k=k, v=v, g=g_cum_new, beta=beta, use_exp2=True
    ),
    "NEW fused_kkt_solve + w_u",
)

print()
print("Done.")
