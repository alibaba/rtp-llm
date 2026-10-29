"""Decode gate metrics. Thresholds originate from 6c0f2c5f8a."""

import math

import torch


def evaluate_precision(
    candidate: torch.Tensor, golden: torch.Tensor, fp8: bool
) -> dict:
    if candidate.shape != golden.shape or candidate.dtype != golden.dtype:
        raise ValueError("candidate and golden output shape/dtype differ")
    if not candidate.numel():
        raise ValueError("empty attention output")
    a, b = candidate.flatten().float(), golden.flatten().float()
    if not bool((torch.isfinite(a) & torch.isfinite(b)).all()):
        return {"overall_pass": False, "reason": "non-finite output"}
    error = (a - b).square().mean().sqrt().item()
    reference = b.square().mean().sqrt().item()
    signal = ((a.square() + b.square()).mean() / 2).sqrt().item()
    nrmse = error / max(signal, 1e-30)
    cosine = (torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30)).item()
    ulp = (
        ((a - b).abs() / (torch.maximum(a.abs(), b.abs()).clamp_min(1e-30) * 2**-7))
        .mean()
        .item()
    )
    cos_limit = 0.998 if fp8 else 0.9999
    snr = reference / (error + 1e-30)
    low_snr = snr < cos_limit / math.sqrt(1 - cos_limit**2)
    absolute_ok = error < max(0.70 if fp8 else 0.02, (1.5 if fp8 else 1.2) * reference)
    passed = absolute_ok if low_snr else cosine >= cos_limit
    passed = passed and nrmse <= (0.20 if fp8 else 0.010)
    passed = passed and (fp8 or ulp <= 2.5)
    return {
        "overall_pass": passed,
        "cos_sim": cosine,
        "nrmse": nrmse,
        "mean_ulp": ulp,
        "snr": snr,
        "low_snr": low_snr,
        "rms_error": error,
    }
