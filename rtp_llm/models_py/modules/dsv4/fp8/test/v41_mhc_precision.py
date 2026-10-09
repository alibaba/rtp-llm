"""Test-only FP32 delayed-mHC projection for reference accuracy diagnosis."""

from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.hc import v41_prenorm


def fp32_mhc_prenorm(residual, weight, eps):
    flattened = residual.flatten(-2).float()
    return F.linear(flattened, weight) * torch.rsqrt(
        flattened.square().mean(-1, keepdim=True) + eps
    )


@contextmanager
def mhc_fp32(enabled):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("FP32 mHC diagnostic is qualified only on SM120")
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with patch.object(v41_prenorm, "prenorm", fp32_mhc_prenorm):
            yield
    finally:
        torch.set_float32_matmul_precision(previous)


def reference_mhc_post(owner, x, residual, post, comb):
    """Released explicit FP32 post equation, with native residual writeback."""
    mixed = (comb.float().unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(dim=-3)
    result = post.float() * x.float().unsqueeze(-2) + mixed
    residual.copy_(result.to(residual.dtype))
    return residual


@contextmanager
def mhc_post_reference(enabled):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("mHC post diagnostic is qualified only on SM120")
    from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit

    with patch.object(DelayedHCUnit, "_post_impl", reference_mhc_post):
        yield
