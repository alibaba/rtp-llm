"""Build diagnostic V4.1 rotary tables on explicit SM120 attention devices."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import rope
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8


@contextmanager
def cuda_frequency_reference(enabled, *, attentions=()):
    if type(enabled) is not bool:
        raise ValueError("CUDA frequency selection requires a bool")
    if not enabled:
        yield
        return
    attentions = tuple(attentions)
    if not attentions or len({id(owner) for owner in attentions}) != len(attentions):
        raise ValueError("Select distinct V4.1 attention owners")
    for owner in attentions:
        if (
            type(owner) is not AttentionV41FP8
            or owner.dim not in (512, 5120)
            or (owner.head_dim, owner.rope_head_dim) != (512, 64)
            or owner.compress_ratio not in (0, 1, 2)
            or owner.freqs_cis.dtype != torch.complex64
            or owner.freqs_cis.shape != (owner._rope_max_seq_len, 32)
            or not owner.freqs_cis.is_cuda
            or not owner.freqs_cis.is_contiguous()
            or torch.cuda.get_device_capability(owner.freqs_cis.device) != (12, 0)
        ):
            raise ValueError("CUDA frequency selection requires admitted SM120 tables")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare CUDA frequency tables before graph capture")
    tables = []
    # A fresh memo also avoids an earlier CPU-constructed table with the same
    # target-device key. Restore the original memo before executing any owner.
    with patch.object(rope, "_FREQS_CIS_CACHE", {}):
        for owner in attentions:
            device = owner.freqs_cis.device
            with torch.device(device):
                tables.append(
                    rope.precompute_freqs_cis(
                        owner._rope_dim,
                        owner._rope_max_seq_len,
                        owner._rope_o_seq_len,
                        owner._rope_base,
                        owner._rope_factor,
                        owner._rope_beta_fast,
                        owner._rope_beta_slow,
                        device=device,
                    )
                )
    with ExitStack() as stack:
        for owner, table in zip(attentions, tables):
            stack.enter_context(patch.object(owner, "freqs_cis", table))
        yield
