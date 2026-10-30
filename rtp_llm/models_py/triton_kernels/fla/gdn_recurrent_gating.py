"""Fuse the private GDN gate/beta producer into small-batch recurrent decode."""

import os

import torch

from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)

_ENABLED = os.getenv("RTP_LLM_SM120_FUSION", "0") == "1"


def is_enabled_for_shape(q):
    if not _ENABLED or q.ndim != 4:
        return False
    batch, seq = q.shape[:2]
    return seq == 1 and 0 < batch <= 32 or seq == 5 and 0 < batch <= 4


def fused_recurrent_with_gating(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    initial_state,
    block_map,
    seq_size_per_block,
    sequence_lengths,
):
    """Return (output, state), or None for the original two-kernel path."""
    if not is_enabled_for_shape(q) or not q.is_cuda:
        return None
    batch, seq, heads, dim = q.shape
    if (
        torch.version.hip is not None
        or torch.cuda.get_device_capability(q.device) != (12, 0)
        or q.dtype != torch.bfloat16
        or (heads, dim) != (8, 128)
        or k.shape != q.shape
        or v.shape != (batch, seq, 24, 128)
        or not (seq == 1 and 0 < batch <= 32 or seq == 5 and 0 < batch <= 4)
        or seq_size_per_block not in (8, 64)
        or initial_state is None
        or initial_state.dtype not in (torch.bfloat16, torch.float32)
        or initial_state.ndim != 4
        or initial_state.shape[-3:] != (24, 128, 128)
        or initial_state.stride(-3) != 128 * 128
        or initial_state.stride(-1) != 1
        or initial_state.stride(-2) != 128
        or block_map is None
        or block_map.ndim != 2
        or block_map.shape[0] != batch
        or block_map.stride(1) != 1
        or block_map.dtype not in (torch.int32, torch.int64)
        or sequence_lengths is None
        or sequence_lengths.shape != (batch,)
        or sequence_lengths.stride(0) != 1
        or sequence_lengths.dtype not in (torch.int32, torch.int64)
        or a is None
        or b is None
        or a.shape != (batch * seq, 24)
        or b.shape != a.shape
        or a.stride(1) != 1
        or b.stride(1) != 1
        or a.stride(0) != b.stride(0)
        or A_log.shape != (24,)
        or dt_bias.shape != (24,)
        or not A_log.is_contiguous()
        or not dt_bias.is_contiguous()
    ):
        return None
    if any(
        t.device != q.device
        for t in (
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            initial_state,
            block_map,
            sequence_lengths,
        )
    ):
        return None
    if any(t.stride(0) != seq * t.stride(1) for t in (q, k, v)):
        return None
    if q.stride(-1) != 1:
        return None
    if any(t.dtype != q.dtype or t.stride(-1) != 1 for t in (k, v, a, b)):
        return None
    if any(t.dtype not in (torch.bfloat16, torch.float32) for t in (A_log, dt_bias)):
        return None
    return fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=a,
        beta=b,
        scale=dim**-0.5,
        initial_state=initial_state,
        inplace_final_state=True,
        block_map=block_map,
        seq_size_per_block=seq_size_per_block,
        sequence_lengths=sequence_lengths,
        use_qk_l2norm_in_kernel=True,
        raw_gating=(A_log, dt_bias),
    )
