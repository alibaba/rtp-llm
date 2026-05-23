"""Precision + performance test for fused RoPE + paged KV cache write (MLA).

Uses the production wrapper `fused_qk_rope_cat_cache_mla` directly, with
inputs matching the real GLM-5 / DeepSeek-V3.2 production environment:

  GLM-5 config.json:
    kv_lora_rank      = 512
    qk_nope_head_dim  = 192
    qk_rope_head_dim  = 64
    num_attention_heads = 64
    → q shape: [T, 64, 256]   (256 = 192 + 64)
    → compressed_kv: [T, 512]
    → k_pe: [T, 64]

  BF16 ("auto"):
    kv_cache: torch.bfloat16, [num_blocks, 64, 576]   (576 = 512 + 64)

  FP8 ("fp8_ds_mla"):
    kv_cache: torch.uint8, [num_blocks, 64, 656]
    656 = 512B fp8_nope + 16B fp32_scales + 128B bf16_rope

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m rtp_llm.models_py.triton_kernels.sparse_mla.test.test_fused_qk_rope_cat_cache_mla
"""

import statistics
import sys

import flashinfer.rope as fi_rope
import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla import (
    fused_qk_rope_cat_cache_mla,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops

# ---- GLM-5 production constants ----
KV_LORA = 512  # kv_lora_rank
NOPE_HEAD_DIM = 192  # qk_nope_head_dim
ROPE = 64  # qk_rope_head_dim
HALF_ROPE = 32
QK_HEAD_DIM = NOPE_HEAD_DIM + ROPE  # 256: q last dim
KV_CACHE_DIM = KV_LORA + ROPE  # 576: bf16 kv_cache last dim
FP8_BYTES = 656  # per-slot bytes for fp8_ds_mla
H = 64  # num_attention_heads
BLOCK_SIZE = 64  # SEQ_SIZE_PER_BLOCK
device = "cuda"
SCALE_1 = torch.tensor(1.0, dtype=torch.float32, device=device)


# ==========================================================================
# Baseline: flashinfer rope + rtp_llm concat_and_cache_mla
# ==========================================================================
def baseline_rope(q, kpe, pos, csc):
    fi_rope._apply_rope_pos_ids_cos_sin_cache(
        q=q[..., NOPE_HEAD_DIM:],
        k=kpe.unsqueeze(1),
        q_rope=q[..., NOPE_HEAD_DIM:],
        k_rope=kpe.unsqueeze(1),
        cos_sin_cache=csc,
        pos_ids=pos,
        interleave=False,
    )


def baseline_bf16(q, ck, kpe, kv_cache, slot, pos, csc):
    baseline_rope(q, kpe, pos, csc)
    rtp_llm_ops.concat_and_cache_mla(ck, kpe, kv_cache, slot, "auto", SCALE_1)


def baseline_fp8(q, ck, kpe, kv_cache, slot, pos, csc):
    baseline_rope(q, kpe, pos, csc)
    rtp_llm_ops.concat_and_cache_mla(ck, kpe, kv_cache, slot, "fp8_ds_mla", SCALE_1)


# ==========================================================================
# Fused: production wrapper
# ==========================================================================
def fused_bf16(q, ck, kpe, kv_cache, slot, pos, csc):
    fused_qk_rope_cat_cache_mla(
        q,
        ck,
        kpe,
        kv_cache,
        slot,
        pos,
        csc,
        kv_lora_rank=KV_LORA,
        rope_head_dim=ROPE,
        is_neox_style=True,
        kv_cache_type="auto",
    )


def fused_fp8(q, ck, kpe, kv_cache, slot, pos, csc):
    fused_qk_rope_cat_cache_mla(
        q,
        ck,
        kpe,
        kv_cache,
        slot,
        pos,
        csc,
        kv_lora_rank=KV_LORA,
        rope_head_dim=ROPE,
        is_neox_style=True,
        kv_cache_type="fp8_ds_mla",
    )


# ==========================================================================
# Input generators — exact GLM-5 production shapes and dtypes
# ==========================================================================
def make_cos_sin_cache():
    inv = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE, 2, device=device, dtype=torch.float32) / ROPE)
    )
    positions = torch.arange(16384.0, device=device)
    return torch.cat(
        [torch.outer(positions, inv).cos(), torch.outer(positions, inv).sin()], dim=-1
    ).to(torch.float32)


def make_inputs_bf16(T, num_heads=H, num_pages=8, seed=0):
    """Production BF16: q=[T,H,256], ck=[T,512], kpe=[T,64], kvc=bf16[N,64,576]."""
    torch.manual_seed(seed)
    q = torch.randn(
        T, num_heads, QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    ).contiguous()
    ck = torch.randn(T, KV_LORA, device=device, dtype=torch.bfloat16)
    kpe = torch.randn(T, ROPE, device=device, dtype=torch.bfloat16)
    kvc = torch.full(
        (num_pages, BLOCK_SIZE, KV_CACHE_DIM), 7.5, device=device, dtype=torch.bfloat16
    )
    total_slots = num_pages * BLOCK_SIZE
    slot = torch.randperm(total_slots, device=device)[:T].to(torch.int64)
    pos = torch.randint(0, 16383, (T,), device=device, dtype=torch.int32)
    return q, ck, kpe, kvc, slot, pos


def make_inputs_fp8(T, num_heads=H, num_pages=8, seed=0):
    """Production FP8: q=[T,H,256], ck=[T,512], kpe=[T,64], kvc=uint8[N,64,656]."""
    torch.manual_seed(seed)
    q = torch.randn(
        T, num_heads, QK_HEAD_DIM, device=device, dtype=torch.bfloat16
    ).contiguous()
    ck = torch.randn(T, KV_LORA, device=device, dtype=torch.bfloat16)
    kpe = torch.randn(T, ROPE, device=device, dtype=torch.bfloat16)
    kvc = torch.full(
        (num_pages, BLOCK_SIZE, FP8_BYTES), 0xAA, device=device, dtype=torch.uint8
    )
    total_slots = num_pages * BLOCK_SIZE
    slot = torch.randperm(total_slots, device=device)[:T].to(torch.int64)
    pos = torch.randint(0, 16383, (T,), device=device, dtype=torch.int32)
    return q, ck, kpe, kvc, slot, pos


# ==========================================================================
# Precision check
# ==========================================================================
csc = make_cos_sin_cache()


def check_precision(name, make_fn, base_fn, fused_fn, cache_cmp_bytes=False):
    print(f"\n{'='*60}")
    print(f"Precision: {name}")
    print(f"{'='*60}")
    all_ok = True
    for seed in [0, 42, 12345]:
        for T in [1, 4, 32]:
            q1, ck1, kpe1, kvc1, slot, pos = make_fn(T, seed=seed)
            q2, ck2, kpe2, kvc2 = q1.clone(), ck1.clone(), kpe1.clone(), kvc1.clone()

            base_fn(q1, ck1, kpe1, kvc1, slot, pos, csc)
            fused_fn(q2, ck2, kpe2, kvc2, slot, pos, csc)
            torch.cuda.synchronize()

            q_ok = torch.equal(q1, q2)
            kpe_ok = torch.equal(kpe1, kpe2)
            kvc_ok = torch.equal(kvc1, kvc2)
            ok = q_ok and kpe_ok and kvc_ok
            if not ok:
                all_ok = False
            extra = ""
            if not kvc_ok:
                if cache_cmp_bytes:
                    d = (kvc1.int() - kvc2.int()).abs()
                else:
                    d = (kvc1.float() - kvc2.float()).abs()
                extra = f"  kvc_diff={int((d>0).sum())}/{d.numel()}"
            if not q_ok:
                qd = (q1.float() - q2.float()).abs()
                extra += f"  q_diff={int((qd>0).sum())}/{qd.numel()}"
            print(
                f"  seed={seed:>5} T={T:>2} H={H:>3}  "
                f"{'OK' if ok else 'FAIL'}  "
                f"q={'OK' if q_ok else 'X'} kpe={'OK' if kpe_ok else 'X'} "
                f"kvc={'OK' if kvc_ok else 'X'}{extra}"
            )
    status = "BIT-EXACT" if all_ok else "FAILED"
    print(f"\n  >> {name}: {status}")
    return all_ok


bf16_ok = check_precision(
    "BF16 (auto) GLM-5 dims",
    make_inputs_bf16,
    baseline_bf16,
    fused_bf16,
    cache_cmp_bytes=False,
)

fp8_ok = check_precision(
    "FP8 (fp8_ds_mla) GLM-5 dims",
    make_inputs_fp8,
    baseline_fp8,
    fused_fp8,
    cache_cmp_bytes=True,
)

if not (bf16_ok and fp8_ok):
    print("\nPRECISION FAILED — skipping perf")
    sys.exit(1)


# ==========================================================================
# Performance: CUDA Graph + N=64 amortized
# ==========================================================================
def cudagraph_time(N, body_fn):
    def whole():
        for _ in range(N):
            body_fn()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            whole()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        whole()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
    torch.cuda.synchronize()
    for i in range(200):
        starts[i].record()
        g.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return (
        statistics.median([s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)])
        / N
    )


def perf_sweep(name, make_fn, base_fn, fused_fn):
    print(f"\n{'='*60}")
    print(f"Performance: {name}  (H={H}, CUDA Graph N=64)")
    print(f"{'='*60}")
    print(f"{'T':>5} | {'baseline us':>12} | {'fused us':>12} | {'speedup':>8}")
    print("-" * 55)
    for T in [1, 2, 4, 8, 16, 24, 32]:
        q, ck, kpe, kvc, slot, pos = make_fn(T)
        kvc_f = kvc.clone()

        def cb(q=q, ck=ck, kpe=kpe, kvc=kvc, s=slot, p=pos, c=csc):
            base_fn(q, ck, kpe, kvc, s, p, c)

        def cf(q=q, ck=ck, kpe=kpe, kvc=kvc_f, s=slot, p=pos, c=csc):
            fused_fn(q, ck, kpe, kvc_f, s, p, c)

        tb = cudagraph_time(64, cb)
        tf = cudagraph_time(64, cf)
        print(f"{T:>5d} | {tb:>12.2f} | {tf:>12.2f} | {tb/tf:>7.2f}x")


perf_sweep("BF16 (auto)", make_inputs_bf16, baseline_bf16, fused_bf16)
perf_sweep("FP8 (fp8_ds_mla)", make_inputs_fp8, baseline_fp8, fused_fp8)

print("\nDone.")
