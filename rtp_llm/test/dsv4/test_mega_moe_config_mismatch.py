"""Comprehensive MegaMoE cross-rank config mismatch tests.

With num_experts=16, topk=4, num_ranks=2:
  expected = T * 2 * 4 / 16 = T * 0.5
  Thresholds: 8.5 (T=17/18), 16.5 (T=33/34), 32.5 (T=65/66), 64.5, 96.5

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node=2 \
        github-opensource/rtp_llm/test/dsv4/test_mega_moe_config_mismatch.py
"""

import os
import signal
import sys
import tempfile
import traceback

DEEP_GEMM_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../bazel-bin/rtp_llm/test/server_test.runfiles/"
    "pip_gpu_cuda13_torch_deep_gemm/site-packages",
)
sys.path.insert(0, os.path.abspath(DEEP_GEMM_PATH))
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
)
os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

import torch
import torch.distributed as dist
import deep_gemm

NUM_EXPERTS = 16
NUM_TOPK = 4
HIDDEN = 4096
INTER = 2048
FP4_BLOCK = 32
TIMEOUT = 60


def get_expected_block_m(T, num_ranks=2, num_experts=16, num_topk=4):
    expected = float(T) * num_ranks * num_topk / num_experts
    if expected <= 8.5:   return 16
    if expected <= 16.5:  return 32
    if expected <= 32.5:  return 64
    if expected <= 64.5:  return 96
    if expected <= 96.5:  return 128
    return 192


def make_random_fp4_weights(e_local, device):
    torch.manual_seed(123)
    l1_w = torch.randint(-128, 127, (e_local, 2 * INTER, HIDDEN // 2), dtype=torch.int8, device=device)
    l1_s = torch.ones((e_local, 2 * INTER, HIDDEN // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device)
    l2_w = torch.randint(-128, 127, (e_local, HIDDEN, INTER // 2), dtype=torch.int8, device=device)
    l2_s = torch.ones((e_local, HIDDEN, INTER // FP4_BLOCK), dtype=torch.float8_e8m0fnu, device=device)
    l1_s_int = deep_gemm.transform_sf_into_required_layout(l1_s.float(), 2 * INTER, HIDDEN, (1, FP4_BLOCK), e_local)
    l2_s_int = deep_gemm.transform_sf_into_required_layout(l2_s.float(), HIDDEN, INTER, (1, FP4_BLOCK), e_local)
    return deep_gemm.transform_weights_for_mega_moe((l1_w, l1_s_int), (l2_w, l2_s_int))


def make_inputs(T, rank, device):
    torch.manual_seed(42 + rank * 1000 + T)
    x_bf16 = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=device)
    x_fp8, x_sf = deep_gemm.per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    e_local = NUM_EXPERTS // 2
    local_start = rank * e_local
    topk_idx = (local_start + torch.randint(0, e_local, (T, NUM_TOPK), device=device)).to(torch.int64)
    topk_weights = torch.rand(T, NUM_TOPK, dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return x_fp8, x_sf, topk_idx, topk_weights


def run_mega_moe(buf, l1, l2, T, rank, device, timeout_sec=TIMEOUT):
    x_fp8, x_sf, idx, w = make_inputs(T, rank, device)

    buf.buffer.zero_()
    dist.barrier()
    torch.cuda.synchronize()

    buf.x[:T].copy_(x_fp8)
    buf.x_sf[:T].copy_(x_sf)
    buf.topk_idx[:T].copy_(idx)
    buf.topk_weights[:T].copy_(w)
    dist.barrier()

    y = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=device)

    old_handler = signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError("kernel deadlock")))
    signal.alarm(timeout_sec)
    try:
        deep_gemm.fp8_fp4_mega_moe(y, l1, l2, buf, recipe=(1, 1, FP4_BLOCK), activation="swiglu", fast_math=True)
        dist.barrier()
        torch.cuda.synchronize()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return y.clone()


def run_test_case(name, T_r0, T_r1, buf, l1, l2, rank, device):
    bm0 = get_expected_block_m(T_r0)
    bm1 = get_expected_block_m(T_r1)
    T_local = T_r0 if rank == 0 else T_r1
    tag = "SAME" if bm0 == bm1 else f"DIFF({bm0} vs {bm1})"

    if rank == 0:
        print(f"\n  [{name}] rank0 T={T_r0}(bm={bm0}), rank1 T={T_r1}(bm={bm1})  [{tag}]", flush=True)

    try:
        y = run_mega_moe(buf, l1, l2, T_local, rank, device)
        norm = y.float().norm().item()
        has_nan = torch.isnan(y).any().item()
        has_inf = torch.isinf(y).any().item()
        status = "OK"
        if has_nan:
            status = "NaN!"
        elif has_inf:
            status = "Inf!"
        if rank == 0:
            print(f"    rank0: norm={norm:.2f} {status}", flush=True)
        dist.barrier()
        return status
    except TimeoutError:
        if rank == 0:
            print(f"    *** DEADLOCK (timeout {TIMEOUT}s) ***", flush=True)
        return "DEADLOCK"
    except Exception as e:
        if rank == 0:
            print(f"    *** EXCEPTION: {e} ***", flush=True)
        return "CRASH"


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    e_local = NUM_EXPERTS // 2
    l1, l2 = make_random_fp4_weights(e_local, device)

    T_max = 200
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group=dist.group.WORLD, num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=T_max, num_topk=NUM_TOPK,
        hidden=HIDDEN, intermediate_hidden=INTER,
    )

    if rank == 0:
        print(f"num_experts={NUM_EXPERTS}, topk={NUM_TOPK}, hidden={HIDDEN}, inter={INTER}")
        print(f"SymmBuffer num_max_tokens_per_rank={buf.num_max_tokens_per_rank}")
        print(f"Thresholds (expected=T*0.5): 8.5→T=17/18, 16.5→T=33/34, 32.5→T=65/66")
        print(f"\n{'='*70}")

    test_cases = [
        # --- Same config ---
        ("same_small",     2,   2),
        ("same_medium",   50,  50),
        ("same_large",   180, 180),

        # --- Cross 8.5 threshold (block_m 16→32) ---
        ("cross_8.5",     17,  18),
        ("cross_8.5_wide", 10, 25),

        # --- Cross 16.5 threshold (block_m 32→64) ---
        ("cross_16.5",    33,  34),
        ("cross_16.5_wide", 20, 50),

        # --- Cross 32.5 threshold (block_m 64→96) ---
        ("cross_32.5",    65,  66),

        # --- Cross multiple thresholds ---
        ("multi_16v64",   17,  34),    # block_m=16 vs 64
        ("multi_16v96",   17,  66),    # block_m=16 vs 96
        ("multi_16v192",  17, 200),    # block_m=16 vs 192

        # --- Extreme asymmetry ---
        ("asym_1v100",     1, 100),
        ("asym_1v200",     1, 200),
        ("asym_200v1",   200,   1),

        # --- T=0 (empty rank) ---
        ("empty_rank0",    0,  18),
        ("empty_rank1",   18,   0),

        # --- Sequential pattern (simulating decode batches) ---
        ("decode_like",   16,  18),
        ("decode_like2",  15,  20),
    ]

    results = {}
    for name, T_r0, T_r1 in test_cases:
        T_local = T_r0 if rank == 0 else T_r1
        if T_local > buf.num_max_tokens_per_rank:
            if rank == 0:
                print(f"\n  [{name}] SKIP: T={T_local} > buf capacity {buf.num_max_tokens_per_rank}")
            results[name] = "SKIP"
            continue
        status = run_test_case(name, T_r0, T_r1, buf, l1, l2, rank, device)
        results[name] = status

    if rank == 0:
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        for name, T_r0, T_r1 in test_cases:
            bm0 = get_expected_block_m(T_r0) if T_r0 > 0 else "N/A"
            bm1 = get_expected_block_m(T_r1) if T_r1 > 0 else "N/A"
            status = results.get(name, "?")
            flag = "" if status == "OK" or status == "SKIP" else " <<<<<"
            print(f"  {name:20s}  T=({T_r0:3d},{T_r1:3d})  bm=({str(bm0):>3s},{str(bm1):>3s})  {status}{flag}")
        print(f"{'='*70}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
