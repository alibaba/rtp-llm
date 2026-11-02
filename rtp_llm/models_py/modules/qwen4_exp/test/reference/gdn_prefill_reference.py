"""Prepare reference GDN inputs for the production SM120 prefill adapter."""

import torch

from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
    sm120_prefill_prepared,
)
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


def sm120_prefill_with_checkpoints(
    q, k, v, g, beta, initial, cu_seqlens, lengths, page_size=128
):
    """Log gates in, FP32 final/page states out; lengths are existing CPU metadata."""

    q = l2norm_fwd(q).squeeze(0).contiguous()
    k = l2norm_fwd(k).squeeze(0).contiguous()
    v = v.squeeze(0).contiguous()
    alpha = torch.exp(g.squeeze(0).float()).contiguous()
    beta_fp32 = beta.squeeze(0).float().contiguous()
    return sm120_prefill_prepared(
        q, k, v, alpha, beta_fp32, initial, cu_seqlens, lengths, page_size
    )
