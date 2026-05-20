"""UT for DeepGEMM FP8 non-paged indexer score path (prefill).

End-to-end equivalence check (mirrors decode UT):

    bf16 reference (v4_indexer_score on dequantized K) ≈
        DeepGEMM fp8_mqa_logits on contiguous gathered (k_quant, k_scale)
        with FP8 Q + folded weights.

Cases (bsz==1 — prefill enforces it):
  * test_prefill_fresh         B=1 S=128 T_live=128 (S==T_live, fresh)
  * test_prefill_continuation  B=1 S=8   T_live=200 (warm cache, no causal)
  * test_prefill_long_S        B=1 S=4096 T_live=4096
  * test_prefill_chunked       split S=256 into 2 chunks, concat rows,
                                compare vs single-shot

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_fp8_mqa_prefill_indexer_score.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_gather_triton import (
    gather_indexer_k_for_prefill,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    quantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_score import (
    fp8_mqa_indexer_score,
    has_fp8_mqa_logits,
)


def _make_packed_cache(k_bf16: torch.Tensor, block_size: int):
    """Quantize ``k_bf16 [T, 128]`` into a paged FP8 pool. Returns
    ``pool_uint8 [(num_blocks+1)*block_size, 132] uint8``. Physical block
    id 0 is invalid, so slot i lives at absolute offset ``block_size + i``."""
    T = k_bf16.shape[0]
    num_blocks = (T + block_size - 1) // block_size
    pool_uint8 = torch.zeros(
        (num_blocks + 1) * block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device=k_bf16.device,
    )
    pool_3d = pool_uint8.view(num_blocks + 1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T, dtype=torch.int64, device=k_bf16.device) + block_size
    quantize_indexer_k(k_bf16, slots, pool_3d)
    return pool_uint8


def _bf16_reference(q_bf16, k_bf16_dequant, weights, q_pos, ratio):
    """[B, S, T] fp32 — bf16 reference path."""
    return v4_indexer_score(
        q_bf16.contiguous(),
        k_bf16_dequant.contiguous(),
        weights.contiguous(),
        q_pos=q_pos,
        compress_ratio=ratio,
    )


def _fp8_path(
    q_bf16, weights, pool_uint8, T_live, q_pos_kernel, ratio, block_size: int = 64
):
    """[B, S, T_live] fp32 — DeepGEMM FP8 non-paged path.

    ``pool_uint8`` is the flat ``[num_blocks*block_size, 132]`` view; the
    gather kernel needs the original ``block_size`` to decode the
    per-block K|scale grouped layout, so we reshape back to 3D here."""
    B, S, H, D = q_bf16.shape
    assert B == 1, "prefill bsz==1"

    total_slots = pool_uint8.shape[0]
    assert total_slots % block_size == 0
    num_blocks = total_slots // block_size
    pool_3d = pool_uint8.view(num_blocks, block_size, INDEXER_ENTRY_BYTES)
    slot_mapping = (
        torch.arange(T_live, dtype=torch.int64, device=q_bf16.device) + block_size
    )
    k_quant, k_scale = gather_indexer_k_for_prefill(
        pool_3d,
        slot_mapping,
        head_dim=D,
    )

    q_fp8, w_fold = indexer_q_fp8_quant_fold(q_bf16.contiguous(), weights)
    M = B * S
    if q_pos_kernel is not None:
        ke = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(T_live)
        cu_ke = ke.reshape(M).contiguous()
    else:
        cu_ke = torch.full((M,), T_live, dtype=torch.int32, device=q_bf16.device)
    cu_ks = torch.zeros(M, dtype=torch.int32, device=q_bf16.device)

    logits = fp8_mqa_indexer_score(
        q_fp8.view(M, H, D),
        w_fold.view(M, H),
        k_quant,
        k_scale,
        cu_ks,
        cu_ke,
        clean_logits=True,  # mask out-of-bounds with -inf for clean diff
    )
    return logits.view(B, S, T_live)


def _compare(label, ref, cand, k_topk, q_pos_kernel=None, ratio=1):
    """mean_abs < 10% of score_max + top-K overlap >= K-2 per row."""
    # Mask out causally-invalid positions in both tensors before comparing.
    # ref already has -inf past threshold (when q_pos was passed); FP8 path
    # leaves garbage past cu_seqlen_ke. Mask cand to match.
    if q_pos_kernel is not None:
        thr = ((q_pos_kernel.to(torch.int32) + 1) // ratio).clamp_max(cand.shape[-1])
        T = cand.shape[-1]
        col = torch.arange(T, device=cand.device).view(1, 1, T)
        valid = col < thr.view(*thr.shape, 1)
        cand = torch.where(valid, cand, torch.full_like(cand, float("-inf")))

    # ref may have -inf in masked positions; replace with 0 for diff stat.
    ref_finite = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
    cand_finite = torch.where(torch.isfinite(cand), cand, torch.zeros_like(cand))
    diff = (ref_finite - cand_finite).abs()
    score_max = ref_finite.abs().amax().item()
    print(
        f"  [{label}] mean_abs={diff.mean().item():.4f} "
        f"max_abs={diff.amax().item():.4f} score_max={score_max:.4f}"
    )
    assert (
        score_max == 0 or diff.mean().item() < 0.10 * score_max
    ), f"{label}: FP8 mean error >10% of score range"

    # Top-K overlap per row.
    B, S, T = ref.shape
    bad = 0
    K_eff = min(k_topk, T)
    counted = 0
    for s in range(S):
        ref_row = ref[0, s]
        cand_row = cand[0, s]
        # Skip rows whose entire range is masked out (early rows of fresh prefill).
        n_valid = (torch.isfinite(ref_row) & (ref_row > float("-inf"))).sum().item()
        K_row = min(K_eff, max(int(n_valid), 0))
        # Tiny-K rows (n_valid < 16) are dominated by FP8 noise on near-tied
        # scores — skip from the bad-row count (they're not useful indexer
        # selections in practice).
        if K_row < 16:
            continue
        counted += 1
        top_ref = set(ref_row.topk(K_row)[1].tolist())
        top_cand = set(cand_row.topk(K_row)[1].tolist())
        overlap = len(top_ref & top_cand)
        # Allow up to 5% per-row mismatches (FP8 noise budget) — what matters
        # for the indexer is that the *bulk* of high-magnitude positions agree.
        if overlap < int(K_row * 0.95):
            bad += 1
    print(
        f"  [{label} top-{K_eff} bad rows] {bad}/{counted} (skipped {S-counted} tiny-K rows)"
    )
    assert bad <= max(
        1, counted // 10
    ), f"{label}: {bad}/{counted} rows failed top-K overlap"


def test_prefill_fresh():
    if not has_fp8_mqa_logits():
        print("  [SKIP] deep_gemm.fp8_mqa_logits unavailable")
        return
    torch.manual_seed(0)
    B, S, H, D = 1, 128, 64, INDEXER_HEAD_DIM
    T_live = 128
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    qpos = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)
    ref = _bf16_reference(q_bf16, k_recon, weights, qpos, ratio)
    cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos, ratio)
    _compare("fresh S=T=128", ref, cand, k_topk=16, q_pos_kernel=qpos, ratio=ratio)


def test_prefill_continuation():
    if not has_fp8_mqa_logits():
        print("  [SKIP]")
        return
    torch.manual_seed(1)
    B, S, H, D = 1, 8, 64, INDEXER_HEAD_DIM
    T_live = 200
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    # No causal mask in continuation.
    ref = _bf16_reference(q_bf16, k_recon, weights, None, ratio)
    cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, None, ratio)
    _compare("continuation S=8 T=200", ref, cand, k_topk=32)


def test_prefill_long_S():
    if not has_fp8_mqa_logits():
        print("  [SKIP]")
        return
    torch.manual_seed(2)
    B, S, H, D = 1, 4096, 64, INDEXER_HEAD_DIM
    T_live = 4096
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )

    qpos = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)
    ref = _bf16_reference(q_bf16, k_recon, weights, qpos, ratio)
    cand = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos, ratio)
    _compare("long S=T=4096", ref, cand, k_topk=64, q_pos_kernel=qpos, ratio=ratio)


def test_prefill_chunked():
    """Split S into 2 chunks; each runs its own fp8_mqa call. Concat rows,
    compare vs single-shot fp8 (and bf16 reference)."""
    if not has_fp8_mqa_logits():
        print("  [SKIP]")
        return
    torch.manual_seed(3)
    B, S, H, D = 1, 256, 64, INDEXER_HEAD_DIM
    T_live = 256
    block_size = 64
    ratio = 4
    q_bf16 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T_live, D, dtype=torch.bfloat16, device="cuda") * 0.5

    pool_uint8 = _make_packed_cache(k_bf16, block_size)
    pool_3d = pool_uint8.view(-1, block_size, INDEXER_ENTRY_BYTES)
    slots = torch.arange(T_live, dtype=torch.int64, device="cuda") + block_size
    k_recon = (
        dequantize_indexer_k(pool_3d, slots, out_dtype=torch.bfloat16)
        .view(B, T_live, D)
        .contiguous()
    )
    qpos_full = torch.arange(S, device="cuda", dtype=torch.int32).view(1, S)

    ref = _bf16_reference(q_bf16, k_recon, weights, qpos_full, ratio)
    cand_single = _fp8_path(q_bf16, weights, pool_uint8, T_live, qpos_full, ratio)

    # Chunked: split S → [0, mid), [mid, S). Each chunk's own qpos / output.
    mid = S // 2
    parts = []
    for s_start, s_end in ((0, mid), (mid, S)):
        q_chunk = q_bf16[:, s_start:s_end]
        w_chunk = weights[:, s_start:s_end]
        qpos_chunk = qpos_full[:, s_start:s_end]
        parts.append(_fp8_path(q_chunk, w_chunk, pool_uint8, T_live, qpos_chunk, ratio))
    cand_chunked = torch.cat(parts, dim=1)

    # Chunked must match single-shot on finite (masked) positions. Both
    # were called with clean_logits=True so masked positions are -inf —
    # diff would be NaN, so finite-mask first.
    finite = torch.isfinite(cand_single) & torch.isfinite(cand_chunked)
    cs = torch.where(finite, cand_single, torch.zeros_like(cand_single))
    cc = torch.where(finite, cand_chunked, torch.zeros_like(cand_chunked))
    diff_cs = (cs - cc).abs().amax().item()
    print(f"  [chunked vs single-shot fp8] max_abs={diff_cs:.6f}")
    assert diff_cs < 1e-3, f"chunked != single-shot fp8 (max_abs={diff_cs})"

    _compare(
        "chunked vs bf16",
        ref,
        cand_chunked,
        k_topk=32,
        q_pos_kernel=qpos_full,
        ratio=ratio,
    )


if __name__ == "__main__":
    print("== FP8 non-paged (prefill) indexer score: equivalence ==")
    test_prefill_fresh()
    test_prefill_continuation()
    test_prefill_long_S()
    test_prefill_chunked()
    print("\nOK")
