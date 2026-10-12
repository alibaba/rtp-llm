"""Scoped released mHC collapse equation for SM120 correctness diagnostics."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc import delayed


def reference_collapse(residual, pre_mix):
    if pre_mix.shape != residual.shape[:-1]:
        raise ValueError("Released collapse coefficients must match residual streams")
    if residual.dtype != torch.bfloat16 or pre_mix.dtype != torch.float32:
        raise TypeError(
            "Released collapse requires BF16 residual and FP32 coefficients"
        )
    return (pre_mix.unsqueeze(-1) * residual.float()).sum(-2).to(residual.dtype)


@contextmanager
def mhc_collapse_reference(enabled):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("mHC collapse reference is qualified only on SM120")
    # DelayedHCUnit's identity entry read remains outside collapse_delayed.
    with patch.object(delayed, "collapse_delayed", reference_collapse):
        yield
