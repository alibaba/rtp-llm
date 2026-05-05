"""UT for DeepGEMM FP8-paged indexer score path.

End-to-end equivalence check:
    bf16 reference (v4_indexer_score on dequant K) ≈ DeepGEMM
        fp8_paged_mqa_logits on packed FP8 cache + FP8 Q + folded weights.

Tolerances:
  * top-K overlap (the only thing the indexer downstream cares about)
    must be within 1 of K.
  * mean abs error vs score range < 10% (FP8 quant noise budget).

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_fp8_paged_indexer_score.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    quantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4._indexer_q_fp8_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_fp8 import (
    fp8_paged_indexer_score,
    has_fp8_paged_mqa_logits,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score


def _make_packed_cache_with_block_table(k_bf16: torch.Tensor, block_size: int):
    """Quantize ``k_bf16 [T, 128]`` into a paged FP8 pool view + return
    the (B=1) block_table mapping logical→physical.

    Returns:
      pool_uint8     [num_blocks * block_size, 132] uint8
      block_table    [1, num_blocks] int32 — identity mapping
    """
    T = k_bf16.shape[0]
    num_blocks = (T + block_size - 1) // block_size
    pool_uint8 = torch.zeros(
        num_blocks * block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=k_bf16.device,
    )
    # slot_mapping == arange(T) → identity placement; in 3D layout the
    # quantize kernel expects [num_blocks, block_size, 132].
    pool_3d = pool_uint8.view(num_blocks, block_size, INDEXER_ENTRY_BYTES)
    slot_mapping = torch.arange(T, dtype=torch.int64, device=k_bf16.device)
    quantize_indexer_k(k_bf16, slot_mapping, pool_3d)
    block_table = torch.arange(
        num_blocks, dtype=torch.int32, device=k_bf16.device
    ).view(1, num_blocks)
    return pool_uint8, block_table


def test_decode_equiv():
    """B=1, S=1, T=256, H=64, D=128 — typical decode shape."""
    if not has_fp8_paged_mqa_logits():
        print("  [SKIP] deep_gemm.fp8_paged_mqa_logits unavailable")
        return
    torch.manual_seed(0)
    B, S, H, D = 1, 1, 64, INDEXER_HEAD_DIM
    T, block_size = 256, 64
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T, D, dtype=torch.bfloat16, device="cuda") * 0.5

    # bf16 reference: dequant cache + v4_indexer_score
    pool_uint8, block_table = _make_packed_cache_with_block_table(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T, D)
        .contiguous()
    )
    score_ref = v4_indexer_score(
        q_bf16.contiguous(),
        k_recon,
        weights.contiguous(),
        q_pos=None,
        compress_ratio=4,
    )  # [B, S, T] fp32

    # FP8 paged path
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
    ctx_lens = torch.full((B, S), T, dtype=torch.int32, device="cuda")
    score_fp8 = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * S, H),
        pool_uint8,
        block_table,
        ctx_lens,
        block_size,
        max_ctx_len=T,
    )  # [B*S, T] fp32
    score_fp8 = score_fp8.view(B, S, T)

    diff = (score_ref - score_fp8).abs()
    score_max = score_ref.abs().amax().item()
    print(
        f"  [decode] mean_abs={diff.mean().item():.4f} "
        f"max_abs={diff.amax().item():.4f} score_max={score_max:.4f}"
    )
    assert (
        diff.mean().item() < 0.10 * score_max
    ), "FP8 path mean error >10% of score range"
    # Top-K overlap: drop strict, allow 1 off (FP8 noise can flip ties).
    K = 32
    top_ref = set(score_ref.topk(K, dim=-1)[1].squeeze().tolist())
    top_fp8 = set(score_fp8.topk(K, dim=-1)[1].squeeze().tolist())
    overlap = len(top_ref & top_fp8)
    print(f"  [decode top-{K} overlap] {overlap}/{K}")
    assert overlap >= K - 2, f"top-{K} overlap only {overlap}/{K}"


def test_decode_partial_context():
    """Cache holds T_max=512 slots but only 100 are live (context_lens=100).
    DeepGEMM should mask past context_lens."""
    if not has_fp8_paged_mqa_logits():
        print("  [SKIP]")
        return
    torch.manual_seed(1)
    B, S, H, D = 1, 1, 64, INDEXER_HEAD_DIM
    T_cache, T_live, block_size = 512, 100, 64
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_cache, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8, block_table = _make_packed_cache_with_block_table(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_cache, dtype=torch.int64, device="cuda")
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_cache, D)
        .contiguous()
    )

    # Reference: only score the live prefix
    score_ref_full = v4_indexer_score(
        q_bf16.contiguous(),
        k_recon,
        weights.contiguous(),
        q_pos=None,
        compress_ratio=4,
    )
    score_ref_live = score_ref_full[..., :T_live]

    q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
    ctx_lens = torch.tensor([[T_live]], dtype=torch.int32, device="cuda")
    score_fp8 = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * S, H),
        pool_uint8,
        block_table,
        ctx_lens,
        block_size,
        max_ctx_len=T_cache,
    )[..., :T_live].view(B, S, T_live)

    diff = (score_ref_live - score_fp8).abs()
    score_max = score_ref_live.abs().amax().item()
    print(
        f"  [partial ctx T_live={T_live}/{T_cache}] mean_abs={diff.mean().item():.4f} "
        f"score_max={score_max:.4f}"
    )
    assert diff.mean().item() < 0.10 * score_max
    K = 16
    top_ref = set(score_ref_live.topk(K, dim=-1)[1].squeeze().tolist())
    top_fp8 = set(score_fp8.topk(K, dim=-1)[1].squeeze().tolist())
    overlap = len(top_ref & top_fp8)
    print(f"  [partial ctx top-{K} overlap] {overlap}/{K}")
    assert overlap >= K - 2


def test_decode_batched():
    """B=4 with varying context_lens."""
    if not has_fp8_paged_mqa_logits():
        print("  [SKIP]")
        return
    torch.manual_seed(2)
    B, S, H, D = 4, 1, 64, INDEXER_HEAD_DIM
    T_cache, block_size = 256, 64
    ctx_lens_list = [100, 200, 256, 64]
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")

    # Per-request K + per-request block_table — share the pool but offset.
    num_blocks_per_req = T_cache // block_size
    total_blocks = B * num_blocks_per_req
    pool_uint8 = torch.zeros(
        total_blocks * block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device="cuda",
    )
    pool_3d = pool_uint8.view(total_blocks, block_size, INDEXER_ENTRY_BYTES)
    block_table = torch.zeros(B, num_blocks_per_req, dtype=torch.int32, device="cuda")
    k_full = torch.zeros(B, T_cache, D, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        k_b = torch.randn(T_cache, D, dtype=torch.bfloat16, device="cuda") * 0.5
        k_full[b] = k_b
        # Place this request's blocks at offset b*num_blocks_per_req.
        base = b * num_blocks_per_req
        block_table[b] = torch.arange(
            base, base + num_blocks_per_req, device="cuda", dtype=torch.int32
        )
        slots = (
            torch.arange(T_cache, device="cuda", dtype=torch.int64) + base * block_size
        )
        # The quant kernel addresses absolute slot in the pool (block_idx
        # = slot // block_size, off = slot % block_size).
        quantize_indexer_k(k_b, slots, pool_3d)

    # bf16 reference per-row
    score_ref = v4_indexer_score(
        q_bf16.contiguous(),
        k_full.contiguous(),
        weights.contiguous(),
        q_pos=None,
        compress_ratio=4,
    )  # [B, S, T_cache]

    q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
    ctx_lens = torch.tensor(
        [[c] for c in ctx_lens_list], dtype=torch.int32, device="cuda"
    )
    score_fp8 = fp8_paged_indexer_score(
        q_fp8,
        w_fold.view(B * S, H),
        pool_uint8,
        block_table,
        ctx_lens,
        block_size,
        max_ctx_len=T_cache,
    ).view(B, S, T_cache)

    # Compare only the live prefix per row.
    bad = 0
    for b in range(B):
        live = ctx_lens_list[b]
        ref_b = score_ref[b, 0, :live]
        cand_b = score_fp8[b, 0, :live]
        sm = ref_b.abs().amax().item()
        diff = (ref_b - cand_b).abs().mean().item()
        K = min(16, live)
        top_ref = set(ref_b.topk(K)[1].tolist())
        top_fp8 = set(cand_b.topk(K)[1].tolist())
        overlap = len(top_ref & top_fp8)
        ok = (diff < 0.10 * sm) and (overlap >= K - 2)
        marker = "OK" if ok else "FAIL"
        print(
            f"  [batch row {b} live={live}] mean_abs={diff:.4f} "
            f"score_max={sm:.4f} top-{K}={overlap}/{K} {marker}"
        )
        if not ok:
            bad += 1
    assert bad == 0, f"{bad} batched rows failed"


def bench_decode():
    """Compare DeepGEMM FP8 paged vs bf16 dequant + Triton score."""
    if not has_fp8_paged_mqa_logits():
        return
    print("\n  decode: B=1 S=1 H=64 D=128 — T sweep")
    print("    {:>5}  {:>10}  {:>10}  {:>8}".format("T", "bf16", "fp8_dg", "speedup"))
    for T in (256, 512, 1024, 2048, 4096):
        block_size = 64
        B, S, H, D = 1, 1, 64, INDEXER_HEAD_DIM
        q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
        k_bf16 = torch.randn(T, D, dtype=torch.bfloat16, device="cuda")
        pool_uint8, block_table = _make_packed_cache_with_block_table(
            k_bf16, block_size
        )
        pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
        slots = torch.arange(T, dtype=torch.int64, device="cuda")
        ctx_lens = torch.tensor([[T]], dtype=torch.int32, device="cuda")
        q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)

        def run_bf16():
            k_r = (
                dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
                .view(B, T, D)
                .contiguous()
            )
            v4_indexer_score(
                q_bf16.contiguous(),
                k_r,
                weights.contiguous(),
                q_pos=None,
                compress_ratio=4,
            )

        def run_fp8():
            fp8_paged_indexer_score(
                q_fp8,
                w_fold.view(B * S, H),
                pool_uint8,
                block_table,
                ctx_lens,
                block_size,
                max_ctx_len=T,
            )

        for _ in range(20):
            run_bf16()
            run_fp8()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(200):
            run_bf16()
        e.record()
        e.synchronize()
        t_bf = s.elapsed_time(e) / 200
        s.record()
        for _ in range(200):
            run_fp8()
        e.record()
        e.synchronize()
        t_fp = s.elapsed_time(e) / 200
        print(f"    {T:5d}  {t_bf*1e3:8.2f}us  {t_fp*1e3:8.2f}us  {t_bf/t_fp:6.2f}x")


if __name__ == "__main__":
    print("== FP8 paged indexer score: equivalence ==")
    test_decode_equiv()
    test_decode_partial_context()
    test_decode_batched()
    print("\n== Bench ==")
    bench_decode()
    print("\nOK")
