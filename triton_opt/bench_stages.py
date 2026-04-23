"""Isolate per-stage timing: fused_kkt_solve vs recompute_w_u vs exp2 overhead."""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/rtp-llm")

B, T, Hg, H, DK, DV, BT = 1, 65536, 8, 32, 128, 128, 64
RCP_LN2 = 1.0 / 0.6931471805599453

torch.manual_seed(42)
k = F.normalize(
    torch.randn(B, T, Hg, DK, device="cuda", dtype=torch.bfloat16), p=2, dim=-1
)
v = torch.randn(B, T, H, DV, device="cuda", dtype=torch.bfloat16)
g = F.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.bfloat16))
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid()

import triton

from rtp_llm.models_py.triton_kernels.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_kkt_solve_kernel,
)
from rtp_llm.models_py.triton_kernels.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.cumsum import chunk_local_cumsum
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.solve_tril import solve_tril
from rtp_llm.models_py.triton_kernels.fla.wy_fast import recompute_w_u_fwd

g_old = chunk_local_cumsum(g, chunk_size=BT)
g_new = chunk_local_cumsum(g, chunk_size=BT, scale=RCP_LN2)

WARMUP, REPS = 5, 20


def bench(fn, name):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    avg = sum(times) / len(times)
    print(f"  {name:40s} {avg:8.0f} us")
    return avg


print("=== OLD Pipeline Stages ===")
bench(
    lambda: chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_old, output_dtype=torch.float32
    ),
    "kkt",
)
A_old = chunk_scaled_dot_kkt_fwd(
    k=k, beta=beta, g_cumsum=g_old, output_dtype=torch.float32
)
bench(lambda: solve_tril(A=A_old, output_dtype=k.dtype), "solve_tril")
A_solved = solve_tril(A=A_old, output_dtype=k.dtype)
bench(
    lambda: recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A_solved, g_cumsum=g_old, cu_seqlens=None, use_exp2=False
    ),
    "recompute_w_u (exp)",
)
bench(
    lambda: recompute_w_u_fwd(
        k=k, v=v, beta=beta, A=A_solved, g_cumsum=g_new, cu_seqlens=None, use_exp2=True
    ),
    "recompute_w_u (exp2)",
)

print()
print("=== NEW Fused kkt+solve (alone, no w_u) ===")
NT = triton.cdiv(T, BT)
A_new = torch.zeros(B, T, H, BT, device="cuda", dtype=k.dtype)


def run_fused_only():
    A_new.zero_()
    chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
        k=k,
        g=g_new,
        beta=beta,
        A=A_new,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        Hg=Hg,
        K=DK,
        BT=BT,
        BC=16,
        BK=64,
        USE_EXP2=True,
        num_warps=1,
        num_stages=1,
    )


bench(run_fused_only, "fused_kkt_solve (BK=64, warps=1)")

# Try different configs
for bk in [32, 64]:
    for nw in [1, 2, 4, 8]:

        def run_cfg(bk=bk, nw=nw):
            A_new.zero_()
            chunk_gated_delta_rule_fwd_kkt_solve_kernel[(NT, B * H)](
                k=k,
                g=g_new,
                beta=beta,
                A=A_new,
                cu_seqlens=None,
                chunk_indices=None,
                T=T,
                H=H,
                Hg=Hg,
                K=DK,
                BT=BT,
                BC=16,
                BK=bk,
                USE_EXP2=True,
                num_warps=nw,
                num_stages=1,
            )

        bench(run_cfg, f"fused (BK={bk}, warps={nw}, stages=1)")

print()
print("=== Comparison ===")
kkt_t = bench(
    lambda: chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_old, output_dtype=torch.float32
    ),
    "kkt",
)
solve_t = bench(lambda: solve_tril(A=A_old, output_dtype=k.dtype), "solve_tril")
fused_t = bench(run_fused_only, "fused_kkt_solve")
print(
    f"\n  kkt+solve = {kkt_t + solve_t:.0f} us  vs  fused = {fused_t:.0f} us  ({fused_t/(kkt_t+solve_t):.2f}x)"
)
