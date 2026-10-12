"""Scope an imported FP32 predecessor to one eager delayed-HC stage entry."""

from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit


class _ImportedPredecessor:
    def __init__(self, pre_mix):
        self.pre_mix_out = pre_mix.clone()


@contextmanager
def imported_delayed_pre_mix(unit, residual, pre_mix):
    """Bind after the caller validates the PP frame and current CP row identity.

    The receiving unit retains ownership of prenorm, split, Sinkhorn and
    readout arithmetic. A copy keeps subsequent receive-buffer reuse from
    changing this forward. This is an adapter only; it does not admit PP.
    """
    if not isinstance(unit, DelayedHCUnit):
        raise TypeError("Imported pre-mix requires the maintained delayed-HC unit")
    if unit._previous_ref is not None:
        raise ValueError(
            "Imported pre-mix requires a stage-entry unit without a local predecessor"
        )
    if (
        not isinstance(residual, torch.Tensor)
        or residual.ndim != 3
        or residual.shape[1:] != (unit.hc_mult, unit.dim)
        or residual.dtype != torch.bfloat16
        or residual.device != unit.fn.device
        or not residual.is_contiguous()
    ):
        raise ValueError(
            "Imported pre-mix requires contiguous BF16 [rows, streams, dim] residuals on the unit device"
        )
    if (
        not isinstance(pre_mix, torch.Tensor)
        or pre_mix.shape != residual.shape[:-1]
        or pre_mix.dtype != torch.float32
        or pre_mix.device != residual.device
        or not pre_mix.is_contiguous()
    ):
        raise ValueError(
            "Imported pre-mix requires contiguous FP32 [rows, streams] coefficients on the residual device"
        )
    if residual.is_cuda and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "Imported delayed-HC predecessor requires eager execution"
        )
    if not bool(torch.isfinite(pre_mix).all()):
        raise ValueError("Imported delayed-HC coefficients must be finite")
    predecessor = _ImportedPredecessor(pre_mix)
    unit.set_previous(predecessor)
    try:
        yield
    finally:
        unit.set_previous(None)
