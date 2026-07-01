"""Three-tier orthogonal precision metrics, pure-CPU core (importable on a
machine without a GPU, no GPU or rtp_llm dependency).

The decode gate reuses this module to judge candidate vs golden. It depends only
on torch and math and imports no rtp_llm or CUDA modules, so it runs on
GPU-less machines and in CI. Metric formulas, threshold derivations, and SNR
tiering are documented in the comments of the individual three-tier metric
functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch

BF16_EPS = 2.0**-7  # BF16 machine epsilon (7-bit mantissa)


# ═══════════════════════════════════════════════════════════════════════════════
# THREE-TIER ORTHOGONAL PRECISION METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tier 1: Cosine similarity — directional agreement.

    Catches: wrong indices, swapped channels, buffer overflow → direction changes.
    Blind spot: pure scale error (doesn't change direction).
    Normal backends: ~0.99999
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    valid = torch.isfinite(a_flat) & torch.isfinite(b_flat)
    if valid.sum() == 0:
        return float("nan")
    a_flat = a_flat[valid]
    b_flat = b_flat[valid]
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def normalized_rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tier 2: NRMSE — scale-invariant signal-level error.

    NRMSE = RMSE(a, b) / RMS_signal, RMS_signal = sqrt(mean(a^2 + b^2) / 2)

    Catches: systematic bias (scale error, offset), whole-page corruption.
    Property: single outlier doesn't dominate (averaged over all elements).
    Normal backends: BF16 ~0.003-0.005, FP8 ~0.02-0.04
    """
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    valid = torch.isfinite(a_f) & torch.isfinite(b_f)
    if valid.sum() == 0:
        return float("nan")
    a_f = a_f[valid]
    b_f = b_f[valid]
    diff = a_f - b_f
    rmse = torch.sqrt(torch.mean(diff**2))
    rms_signal = torch.sqrt(torch.mean(a_f**2 + b_f**2) / 2)
    return (rmse / torch.clamp(rms_signal, min=1e-30)).item()


def mean_relative_ulp(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    """Tier 3: Mean relative ULP — per-element average precision.

    rel_ulp[i] = |a[i] - b[i]| / (max(|a[i]|, |b[i]|) * BF16_EPS)

    Catches: fine-grained precision degradation across the board.
    Normal backends: BF16 ~1.2-1.8, FP8 ~8-13
    Returns: (mean_ulp, max_ulp)
    """
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    valid = torch.isfinite(a_f) & torch.isfinite(b_f)
    if valid.sum() == 0:
        return float("nan"), float("nan")
    a_f = a_f[valid]
    b_f = b_f[valid]
    abs_diff = (a_f - b_f).abs()
    magnitude = torch.maximum(a_f.abs(), b_f.abs())
    ulp = torch.clamp(magnitude, min=1e-30) * BF16_EPS
    rel_ulp = abs_diff / ulp
    return rel_ulp.mean().item(), rel_ulp.max().item()


def absolute_error_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Absolute-scale error view — the antidote to the small-||output|| trap.

    cos_sim and nrmse are BOTH relative: when the output norm is tiny (layer-0 /
    short kv_len, V vectors largely cancel), a small absolute deviation divided
    by a small signal makes relative metrics explode even though nothing is
    wrong.  Returns raw RMS error, max element error, and reference RMS so the
    SNR gate can fall back to absolute error when signal is weak.
    """
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    valid = torch.isfinite(a_f) & torch.isfinite(b_f)
    if valid.sum() == 0:
        return {
            "rms_abs_err": float("nan"),
            "max_abs_err": float("nan"),
            "ref_rms": float("nan"),
        }
    a_f = a_f[valid]
    b_f = b_f[valid]
    diff = (a_f - b_f).abs()
    return {
        "rms_abs_err": torch.sqrt(torch.mean(diff**2)).item(),
        "max_abs_err": diff.max().item(),
        "ref_rms": torch.sqrt(torch.mean(b_f**2)).item(),
    }


def per_head_cosine(
    a: torch.Tensor, b: torch.Tensor, num_heads: int, head_dim: int
) -> dict:
    """Per-head directional agreement + a norm-weighted aggregate (diagnosis)."""
    a_f = a.reshape(-1, num_heads, head_dim).float()
    b_f = b.reshape(-1, num_heads, head_dim).float()
    dot = (a_f * b_f).sum(dim=-1)
    na = a_f.norm(dim=-1)
    nb = b_f.norm(dim=-1)
    cos = dot / (na * nb).clamp_min(1e-30)
    cos_flat = cos.flatten()
    nb_flat = nb.flatten()
    valid = torch.isfinite(cos_flat)
    if valid.sum() == 0:
        return {
            "cos_perhead_median": float("nan"),
            "cos_perhead_min": float("nan"),
            "cos_normweighted": float("nan"),
        }
    cos_v = cos_flat[valid]
    w = nb_flat[valid]
    cos_nw = (cos_v * w).sum() / w.sum().clamp_min(1e-30)
    return {
        "cos_perhead_median": cos_v.median().item(),
        "cos_perhead_min": cos_v.min().item(),
        "cos_normweighted": cos_nw.item(),
    }


# ─── Threshold config ─────────────────────────────────────────────────────────
THRESHOLDS = {
    "BF16": {"cos_sim": 0.9999, "nrmse": 0.010, "mean_ulp": 2.5},
    # FP8 cross-backend is deprecated: attention kernel output is always BF16,
    # FP8 only affects the KV cache; FP8 quantization quality is covered by
    # evaluate_fp8_quality / FP8_QUALITY_THRESHOLDS.
}

FP8_QUALITY_THRESHOLDS = {
    "cos_sim": 0.998,
    "nrmse": 0.20,  # calibrated on 5424 real-tensor samples: P50=0.089, P95=0.166, P99=0.289
}

# ─── SNR gating config (per comparison mode) ─────────────────────────────────
# abs_floor calibrated from 16272 real-tensor samples (calibrate_abs_floor.py,
# 2026-06-22), ≈ P99.5 of rms_abs_err.
SNR_GATE_CONFIG = {
    "cross_backend_BF16": {
        "cos_threshold": 0.9999,
        "abs_floor": 0.02,
        "rel_ratio": 1.2,
    },
    "fp8_quality": {"cos_threshold": 0.998, "abs_floor": 0.70, "rel_ratio": 1.5},
    "fp32_ref": {"cos_threshold": 0.99, "abs_floor": 0.05, "rel_ratio": 1.2},
}


def snr_gated_judgment(
    rms_abs_err: float,
    ref_rms: float,
    cos_sim: float,
    cos_threshold: float,
    abs_floor: float,
    rel_ratio: float,
) -> dict:
    """SNR-based pass/fail: when signal is weak, fall back to absolute error only.

    SNR = ref_rms / rms_abs_err; SNR_threshold = cos_t / sqrt(1 - cos_t^2).
    HIGH_SNR → cos participates + abs gate; LOW_SNR → cos skipped, abs only.
    """
    snr = ref_rms / (rms_abs_err + 1e-30)
    snr_threshold = cos_threshold / math.sqrt(1 - cos_threshold**2)

    abs_threshold = max(abs_floor, rel_ratio * ref_rms)
    pass_abs = rms_abs_err < abs_threshold

    if snr >= snr_threshold:
        snr_regime = "HIGH_SNR"
        pass_cos = (not math.isnan(cos_sim)) and cos_sim >= cos_threshold
        overall = pass_abs and pass_cos
    else:
        snr_regime = "LOW_SNR"
        pass_cos = True
        overall = pass_abs

    reasons = []
    if not pass_abs:
        reasons.append(f"rms_abs={rms_abs_err:.2e}>=threshold({abs_threshold:.2e})")
    if snr >= snr_threshold and not pass_cos:
        reasons.append(f"cos={cos_sim:.6f}<{cos_threshold}")

    return {
        "snr": snr,
        "snr_regime": snr_regime,
        "snr_threshold": snr_threshold,
        "pass_abs": pass_abs,
        "pass_cos": pass_cos,
        "overall_pass": overall,
        "abs_threshold": abs_threshold,
        "fail_reasons": reasons,
    }


@dataclass
class PrecisionResult:
    """Result of one precision comparison (cross-backend BF16 or FP8 quality)."""

    cos_sim: float
    nrmse: float
    mean_ulp: float
    max_ulp: float
    rms_abs_err: float
    ref_rms: float
    snr: float
    snr_regime: str  # "HIGH_SNR" | "LOW_SNR"
    tier1_pass: bool  # cos_sim (SNR-gated)
    tier2_pass: bool  # nrmse
    tier3_pass: bool  # mean_ulp (FP8: always True, not used)
    pass_abs: bool
    overall_pass: bool
    fail_reason: str


def evaluate_precision(a: torch.Tensor, b: torch.Tensor, kv_dtype_str: str) -> dict:
    """BF16 cross-backend judgment: three tiers + SNR-gated cos.

    Used for the BASE (bf16 KV cache) gate case: candidate(bf16) vs golden(bf16).
    """
    cos = cosine_similarity_flat(a, b)
    nrmse = normalized_rmse(a, b)
    m_ulp, mx_ulp = mean_relative_ulp(a, b)
    abs_m = absolute_error_metrics(a, b)

    thresh = THRESHOLDS[kv_dtype_str]
    nrmse_valid = not math.isnan(nrmse)
    mulp_valid = not math.isnan(m_ulp)

    gate_cfg = SNR_GATE_CONFIG[f"cross_backend_{kv_dtype_str}"]
    gate = snr_gated_judgment(
        abs_m["rms_abs_err"],
        abs_m["ref_rms"],
        cos,
        gate_cfg["cos_threshold"],
        gate_cfg["abs_floor"],
        gate_cfg["rel_ratio"],
    )

    t1 = gate["pass_cos"]
    t2 = nrmse_valid and nrmse <= thresh["nrmse"]
    t3 = mulp_valid and m_ulp <= thresh["mean_ulp"]
    overall = gate["overall_pass"] and t2 and t3

    reasons = list(gate["fail_reasons"])
    if not t2:
        reasons.append(f"nrmse={nrmse:.6f}>{thresh['nrmse']}")
    if not t3:
        reasons.append(f"mean_ulp={m_ulp:.3f}>{thresh['mean_ulp']}")

    return {
        "cos_sim": cos,
        "nrmse": nrmse,
        "mean_ulp": m_ulp,
        "max_ulp": mx_ulp,
        "rms_abs_err": abs_m["rms_abs_err"],
        "ref_rms": abs_m["ref_rms"],
        "snr": gate["snr"],
        "snr_regime": gate["snr_regime"],
        "tier1_pass": t1,
        "tier2_pass": t2,
        "tier3_pass": t3,
        "pass_abs": gate["pass_abs"],
        "overall_pass": overall,
        "fail_reason": "; ".join(reasons) if reasons else "",
    }


def evaluate_fp8_quality(a: torch.Tensor, b: torch.Tensor) -> dict:
    """FP8 quality judgment: cos (SNR-gated) + nrmse only; mean_ulp NOT used.

    Used for the FP8 gate case: candidate(reads FP8 cache) vs golden(ideal
    FP32→bf16).  The deviation bundles FP8 quantization loss + kernel error;
    this is a coarse filter.  Same logic as bench's run_fp8_vs_bf16_baseline.
    """
    cos = cosine_similarity_flat(a, b)
    nrmse_val = normalized_rmse(a, b)
    m_ulp, mx_ulp = mean_relative_ulp(a, b)
    abs_m = absolute_error_metrics(a, b)

    nrmse_valid = not math.isnan(nrmse_val)
    nrmse_pass = nrmse_valid and nrmse_val <= FP8_QUALITY_THRESHOLDS["nrmse"]

    gate_cfg = SNR_GATE_CONFIG["fp8_quality"]
    gate = snr_gated_judgment(
        abs_m["rms_abs_err"],
        abs_m["ref_rms"],
        cos,
        gate_cfg["cos_threshold"],
        gate_cfg["abs_floor"],
        gate_cfg["rel_ratio"],
    )
    overall = gate["overall_pass"] and nrmse_pass

    reasons = list(gate["fail_reasons"])
    if not nrmse_pass:
        reasons.append(f"nrmse={nrmse_val:.6f}>{FP8_QUALITY_THRESHOLDS['nrmse']}")

    return {
        "cos_sim": cos,
        "nrmse": nrmse_val,
        "mean_ulp": m_ulp if not math.isnan(m_ulp) else 0.0,
        "max_ulp": mx_ulp,
        "rms_abs_err": abs_m["rms_abs_err"],
        "ref_rms": abs_m["ref_rms"],
        "snr": gate["snr"],
        "snr_regime": gate["snr_regime"],
        "tier1_pass": gate["pass_cos"],
        "tier2_pass": nrmse_pass,
        "tier3_pass": True,  # mean_ulp not used cross-precision
        "pass_abs": gate["pass_abs"],
        "overall_pass": overall,
        "fail_reason": "; ".join(reasons) if reasons else "",
    }
