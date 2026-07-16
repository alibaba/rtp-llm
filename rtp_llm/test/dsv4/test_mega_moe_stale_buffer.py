"""Reproduce MegaMoE stale SymmBuffer bug.

When chunked MoE calls fp8_fp4_mega_moe twice on the same SymmBuffer, the
second call inherits stale routing data (topk_idx, topk_weights, x, x_sf)
from the first call at positions [T_small : T_large].  If the kernel reads
beyond T into stale slots, the output is silently corrupted.

Usage:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc-per-node=2 \
        github-opensource/rtp_llm/test/dsv4/test_mega_moe_stale_buffer.py
"""

import os
import sys
import tempfile

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

NUM_EXPERTS = 256
NUM_TOPK = 6
HIDDEN = 4096
INTER = 2048
FP4_BLOCK = 32
T_LARGE = 16
T_SMALL = 2


def make_random_fp4_weights(e_local: int, device: torch.device):
    """Create random FP4 weights in the format MegaMoE expects."""
    l1_w = torch.randint(
        -128, 127, (e_local, 2 * INTER, HIDDEN // 2), dtype=torch.int8, device=device
    )
    l1_s_raw = torch.ones(
        (e_local, 2 * INTER, HIDDEN // FP4_BLOCK),
        dtype=torch.float8_e8m0fnu,
        device=device,
    )
    l2_w = torch.randint(
        -128, 127, (e_local, HIDDEN, INTER // 2), dtype=torch.int8, device=device
    )
    l2_s_raw = torch.ones(
        (e_local, HIDDEN, INTER // FP4_BLOCK),
        dtype=torch.float8_e8m0fnu,
        device=device,
    )

    l1_s_int = deep_gemm.transform_sf_into_required_layout(
        l1_s_raw.float(), 2 * INTER, HIDDEN, (1, FP4_BLOCK), e_local
    )
    l2_s_int = deep_gemm.transform_sf_into_required_layout(
        l2_s_raw.float(), HIDDEN, INTER, (1, FP4_BLOCK), e_local
    )

    (l1_w, l1_sf), (l2_w, l2_sf) = deep_gemm.transform_weights_for_mega_moe(
        (l1_w, l1_s_int), (l2_w, l2_s_int)
    )
    return (l1_w, l1_sf), (l2_w, l2_sf)


def make_inputs(T: int, rank: int, device: torch.device):
    """Create random MoE inputs: bf16 hidden states + routing."""
    torch.manual_seed(42 + rank * 1000 + T)
    x_bf16 = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=device)
    x_fp8, x_sf = deep_gemm.per_token_cast_to_fp8(
        x_bf16, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )

    e_local = NUM_EXPERTS // 2
    local_start = rank * e_local
    topk_idx = (
        local_start
        + torch.randint(0, e_local, (T, NUM_TOPK), device=device)
    ).to(torch.int64)
    topk_weights = torch.rand(T, NUM_TOPK, dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return x_fp8, x_sf, topk_idx, topk_weights


def pack_into_buf(buf, x_fp8, x_sf, topk_idx, topk_weights, T: int):
    buf.x[:T].copy_(x_fp8[:T])
    buf.x_sf[:T].copy_(x_sf[:T])
    buf.topk_idx[:T].copy_(topk_idx[:T])
    buf.topk_weights[:T].copy_(topk_weights[:T])


def call_mega_moe(buf, l1, l2, T: int, device: torch.device) -> torch.Tensor:
    y = torch.empty(T, HIDDEN, dtype=torch.bfloat16, device=device)
    deep_gemm.fp8_fp4_mega_moe(
        y, l1, l2, buf,
        recipe=(1, 1, FP4_BLOCK),
        activation="swiglu",
        fast_math=True,
    )
    return y.clone()


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"Need exactly 2 ranks, got {world_size}"

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    e_local = NUM_EXPERTS // world_size

    if rank == 0:
        print(f"[rank {rank}] Creating random FP4 weights for {e_local} experts...")
    l1, l2 = make_random_fp4_weights(e_local, device)

    if rank == 0:
        print(f"[rank {rank}] Allocating SymmBuffer (num_max_tokens_per_rank={T_LARGE})...")
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group=dist.group.WORLD,
        num_experts=NUM_EXPERTS,
        num_max_tokens_per_rank=T_LARGE,
        num_topk=NUM_TOPK,
        hidden=HIDDEN,
        intermediate_hidden=INTER,
        use_fp8_dispatch=True,
        activation="swiglu",
    )
    if rank == 0:
        print(
            f"[rank {rank}] SymmBuffer allocated: num_max_tokens_per_rank="
            f"{buf.num_max_tokens_per_rank}, buf.x.shape={tuple(buf.x.shape)}"
        )

    x_large_fp8, x_large_sf, idx_large, w_large = make_inputs(T_LARGE, rank, device)
    x_small_fp8, x_small_sf, idx_small, w_small = make_inputs(T_SMALL, rank, device)

    # ---- Test A: clean buffer → call with T_SMALL ----
    if rank == 0:
        print(f"\n[Test A] Clean buffer, T={T_SMALL}")
    buf.buffer.zero_()
    dist.barrier()
    torch.cuda.synchronize()

    pack_into_buf(buf, x_small_fp8, x_small_sf, idx_small, w_small, T_SMALL)
    dist.barrier()
    y_clean = call_mega_moe(buf, l1, l2, T_SMALL, device)
    dist.barrier()
    torch.cuda.synchronize()

    # ---- Test B: T_LARGE then T_SMALL (stale) ----
    if rank == 0:
        print(f"\n[Test B] Contaminate with T={T_LARGE}, then T={T_SMALL}")
    buf.buffer.zero_()
    dist.barrier()
    torch.cuda.synchronize()

    pack_into_buf(buf, x_large_fp8, x_large_sf, idx_large, w_large, T_LARGE)
    dist.barrier()
    y_warmup = call_mega_moe(buf, l1, l2, T_LARGE, device)
    dist.barrier()
    torch.cuda.synchronize()

    pack_into_buf(buf, x_small_fp8, x_small_sf, idx_small, w_small, T_SMALL)
    dist.barrier()
    y_stale = call_mega_moe(buf, l1, l2, T_SMALL, device)
    dist.barrier()
    torch.cuda.synchronize()

    # ---- Compare A vs B ----
    diff_ab = (y_clean.float() - y_stale.float()).abs()
    max_diff_ab = diff_ab.max().item()

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[A vs B] max abs diff: {max_diff_ab:.6f}")
        print(f"  y_clean norm: {y_clean.float().norm().item():.4f}")
        print(f"  y_stale norm: {y_stale.float().norm().item():.4f}")
        if max_diff_ab > 0:
            print(f"  *** BUG: stale INPUT buffer contaminates output ***")
        else:
            print(f"  OK: stale input buffer does NOT affect output.")

    # ---- Test C: garbage in intermediate activation buffers ----
    if rank == 0:
        print(f"\n[Test C] Garbage in l1_acts/l2_acts, then T={T_SMALL}")

    buf.buffer.zero_()
    dist.barrier()
    torch.cuda.synchronize()

    pack_into_buf(buf, x_small_fp8, x_small_sf, idx_small, w_small, T_SMALL)
    dist.barrier()
    y_clean2 = call_mega_moe(buf, l1, l2, T_SMALL, device)
    dist.barrier()
    torch.cuda.synchronize()

    # Now trash the intermediate buffers while keeping input intact
    buf.l1_acts.fill_(127)
    buf.l1_acts_sf.fill_(0x7F7F7F7F)
    buf.l2_acts.fill_(127)
    buf.l2_acts_sf.fill_(0x7F7F7F7F)
    dist.barrier()
    torch.cuda.synchronize()

    # Re-pack same inputs (input buffers are still correct)
    pack_into_buf(buf, x_small_fp8, x_small_sf, idx_small, w_small, T_SMALL)
    dist.barrier()
    y_garbage_acts = call_mega_moe(buf, l1, l2, T_SMALL, device)
    dist.barrier()
    torch.cuda.synchronize()

    diff_c = (y_clean2.float() - y_garbage_acts.float()).abs()
    max_diff_c = diff_c.max().item()
    if rank == 0:
        print(f"[A vs C] max abs diff: {max_diff_c:.6f}")
        print(f"  y_clean2       norm: {y_clean2.float().norm().item():.4f}")
        print(f"  y_garbage_acts norm: {y_garbage_acts.float().norm().item():.4f}")
        if max_diff_c > 0:
            print(f"  *** BUG: stale INTERMEDIATE buffers contaminate output ***")
        else:
            print(f"  OK: intermediate buffers do NOT affect output.")

    # ---- Test D: repeated calls — check determinism ----
    if rank == 0:
        print(f"\n[Test D] Repeat T={T_SMALL} call 5 times, check determinism")
    results = []
    for i in range(5):
        buf.buffer.zero_()
        dist.barrier()
        torch.cuda.synchronize()
        pack_into_buf(buf, x_small_fp8, x_small_sf, idx_small, w_small, T_SMALL)
        dist.barrier()
        y_i = call_mega_moe(buf, l1, l2, T_SMALL, device)
        dist.barrier()
        torch.cuda.synchronize()
        results.append(y_i)

    max_diff_d = 0.0
    for i in range(1, len(results)):
        d = (results[0].float() - results[i].float()).abs().max().item()
        max_diff_d = max(max_diff_d, d)
    if rank == 0:
        print(f"[D] max abs diff across 5 runs: {max_diff_d:.6f}")
        if max_diff_d > 0:
            print(f"  *** BUG: kernel is non-deterministic ***")
        else:
            print(f"  OK: kernel is deterministic across repeated calls.")

    any_bug = max_diff_ab > 0 or max_diff_c > 0 or max_diff_d > 0
    if rank == 0:
        print(f"\n{'='*60}")
        if any_bug:
            print("RESULT: At least one stale buffer scenario causes corruption.")
        else:
            print("RESULT: No stale buffer contamination detected.")
        print(f"{'='*60}")

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        sys.exit(1 if any_bug else 0)


if __name__ == "__main__":
    main()
