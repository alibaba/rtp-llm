"""UT for ``indexer_q_fp8_quant_fold``.

Validates the per-(token, head) FP8 quant + scale-fold math used by the
DeepGEMM ``fp8_paged_mqa_logits`` path:
   sum_h ReLU(real_q · k) * w  ==  sum_h ReLU(q_fp8 · k_dequant) * w_fold

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_indexer_q_fp8_quant.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4._indexer_q_fp8_quant_triton import (
    FP8_E4M3_MAX,
    INDEXER_HEAD_DIM,
    indexer_q_fp8_quant_fold,
)


def test_quant_round_trip():
    """q_dequant = q_fp8 * scale recovers q_bf16 within fp8 precision."""
    torch.manual_seed(0)
    B, S, H, D = 1, 1, 64, INDEXER_HEAD_DIM
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q, w)
    # Recover scale = w_fold / w (where w != 0).
    safe_w = w.float()
    safe_w = torch.where(safe_w.abs() > 1e-3, safe_w, torch.ones_like(safe_w))
    scale_recov = w_fold / safe_w
    q_dequant = q_fp8.float() * scale_recov.unsqueeze(-1)
    rel = (q_dequant - q.float()).abs() / (q.float().abs() + 1e-6)
    print(f"  [round-trip] max_rel={rel.amax().item():.4f}")
    assert rel.amax().item() < 0.15, f"fp8 precision exceeded: {rel.amax().item()}"


def test_score_equivalence():
    """Verify the math identity:
       sum_h ReLU(q_real · k) * w == sum_h ReLU(q_fp8 · k_dequant) * w_fold
    on a small synthetic example where ``q_dequant ≈ q_real``."""
    torch.manual_seed(1)
    B, S, T, H, D = 1, 1, 32, 8, INDEXER_HEAD_DIM
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda") * 0.5
    w = torch.randn(B, S, H, dtype=torch.float32, device="cuda")

    # Reference: bf16 path.
    score_ref = torch.einsum("bshd,btd->bsht", q.float(), k.float())  # [B,S,H,T]
    score_ref = torch.relu(score_ref) * w.unsqueeze(-1)
    score_ref = score_ref.sum(dim=2)  # [B,S,T]

    # FP8 quant + fold.
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q, w)
    score_fp8 = torch.einsum("bshd,btd->bsht", q_fp8.float(), k.float())  # [B,S,H,T]
    score_fp8 = torch.relu(score_fp8) * w_fold.unsqueeze(-1)
    score_fp8 = score_fp8.sum(dim=2)  # [B,S,T]

    # FP8 round-trip noise per element ≈6%; sum over H=8 heads
    # propagates linearly. Compare against the score *magnitude scale*,
    # not per-element rel-error (near-zero positions dominate that ratio).
    diff = (score_ref - score_fp8).abs()
    score_scale = score_ref.abs().amax().item()
    print(
        f"  [score equiv] max_abs={diff.amax().item():.4f}  "
        f"mean_abs={diff.mean().item():.4f}  "
        f"score_max={score_scale:.4f}"
    )
    assert diff.mean().item() < 0.10 * score_scale, (
        f"mean abs {diff.mean().item()} > 10% of score range {score_scale} — "
        "fold math broken"
    )
    # Top-K agreement (what the indexer actually uses).
    k = min(8, score_ref.shape[-1])
    top_ref = score_ref.topk(k, dim=-1)[1].squeeze().tolist()
    top_fp8 = score_fp8.topk(k, dim=-1)[1].squeeze().tolist()
    overlap = len(set(top_ref) & set(top_fp8))
    print(f"  [top-{k} overlap] {overlap}/{k}")
    assert overlap >= k - 2, f"top-{k} overlap only {overlap}/{k}"


def test_negative_weights_ok():
    """Scale is always positive so ReLU commutes; negative weights still
    fold correctly (w_fold = w * scale; sign carried by w)."""
    torch.manual_seed(2)
    B, S, T, H, D = 1, 1, 16, 4, INDEXER_HEAD_DIM
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.3
    k = torch.randn(B, T, D, dtype=torch.bfloat16, device="cuda") * 0.3
    w = torch.randn(B, S, H, dtype=torch.float32, device="cuda")  # signed
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q, w)

    score_ref = torch.einsum("bshd,btd->bsht", q.float(), k.float())
    score_ref = (torch.relu(score_ref) * w.unsqueeze(-1)).sum(dim=2)

    score_fp8 = torch.einsum("bshd,btd->bsht", q_fp8.float(), k.float())
    score_fp8 = (torch.relu(score_fp8) * w_fold.unsqueeze(-1)).sum(dim=2)

    diff = (score_ref - score_fp8).abs()
    score_scale = score_ref.abs().amax().item()
    print(
        f"  [signed weights] mean_abs={diff.mean().item():.4f} "
        f"score_max={score_scale:.4f}"
    )
    assert diff.mean().item() < 0.10 * score_scale


def test_zero_input():
    """All-zero q → scale clamped, fp8 codes all zero."""
    B, S, H, D = 1, 1, 4, INDEXER_HEAD_DIM
    q = torch.zeros(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    w = torch.ones(B, S, H, dtype=torch.bfloat16, device="cuda")
    q_fp8, w_fold = indexer_q_fp8_quant_fold(q, w)
    assert (q_fp8.view(torch.uint8) == 0).all(), "all-zero q should give zero codes"
    print("  [zero input] OK")


if __name__ == "__main__":
    print("== Correctness ==")
    test_quant_round_trip()
    test_score_equivalence()
    test_negative_weights_ok()
    test_zero_input()
    print("\nOK")
