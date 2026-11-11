"""Fused MXFP8 SiLU must stay off devices without native FP8 GEMM.

The kernel writes ``tl.float8e4nv`` and Triton rejects that dtype on SM80.
A100 also runs V41MXFP8Linear's reference dequant path, so the fused gate
must require both linears to be on their native (non-fallback) path.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.moe import shared_expert
from rtp_llm.models_py.modules.dsv41.moe.shared_expert import W13SharedExpert
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def _linear(fallback: bool) -> V41MXFP8Linear:
    linear = object.__new__(V41MXFP8Linear)
    linear._fallback = fallback
    return linear


def _expert(w13: V41MXFP8Linear, w2: V41MXFP8Linear) -> W13SharedExpert:
    expert = object.__new__(W13SharedExpert)
    torch.nn.Module.__init__(expert)
    expert.w13 = w13
    expert.w2 = w2
    return expert


def _x():
    return SimpleNamespace(
        is_cuda=True,
        dtype=torch.bfloat16,
        dim=lambda: 2,
    )


def test_native_fp8_device_can_use_fused_silu():
    with patch.object(shared_expert, "_MXFP8_FUSED_SILU_OK", True):
        assert _expert(_linear(False), _linear(False))._can_run_mxfp8_fused_silu(_x())


def test_a100_fallback_linear_disables_fused_silu():
    with patch.object(shared_expert, "_MXFP8_FUSED_SILU_OK", True):
        assert not _expert(_linear(True), _linear(False))._can_run_mxfp8_fused_silu(_x())
        assert not _expert(_linear(False), _linear(True))._can_run_mxfp8_fused_silu(_x())
        assert not _expert(_linear(True), _linear(True))._can_run_mxfp8_fused_silu(_x())


def test_unavailable_kernel_stays_disabled():
    with patch.object(shared_expert, "_MXFP8_FUSED_SILU_OK", False):
        assert not _expert(_linear(False), _linear(False))._can_run_mxfp8_fused_silu(_x())
