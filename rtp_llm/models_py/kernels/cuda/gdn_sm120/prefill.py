"""SM120 GDN prefill with FP32 final states and 128-token checkpoints."""

import functools
import logging
from importlib import metadata

import torch

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()


@functools.lru_cache(maxsize=1)
def sm120_dsl_available():
    try:
        if metadata.version("nvidia-cutlass-dsl") != "4.4.2":
            return False
        from rtp_llm.models_py.utils.cutlass import cuda13_cutlass_is_intact

        if not cuda13_cutlass_is_intact():
            return False
        from .delta_rule_cp_sm120 import cp_delta_rule_dsl_sm120

        return callable(cp_delta_rule_dsl_sm120)
    except (ImportError, AttributeError, metadata.PackageNotFoundError) as error:
        logging.warning("SM120 GDN DSL unavailable; using native prefill: %s", error)
        return False


def sm120_prefill_prepared(
    q, k, v, alpha, beta_fp32, initial, cu_seqlens, lengths, page_size=128
):
    from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
        checkpoint_starts as make_checkpoint_starts,
    )

    heads = max(q.shape[1], v.shape[1])
    if page_size != 128:
        raise ValueError("SM120 prefill checkpoint pages must contain 128 tokens")
    checkpoint_starts = make_checkpoint_starts(cu_seqlens, len(lengths))
    checkpoints = torch.empty(
        (sum(n // page_size for n in lengths), heads, 128, 128),
        device=q.device,
        dtype=torch.float32,
    )
    final = torch.empty(
        (len(lengths), heads, 128, 128), device=q.device, dtype=torch.float32
    )
    output = torch.empty((q.shape[0], heads, 128), device=q.device, dtype=q.dtype)
    initial_fp32 = initial.float().contiguous() if initial is not None else None
    from .delta_rule_cp_sm120 import cp_delta_rule_dsl_sm120

    cp_delta_rule_dsl_sm120(
        output,
        final,
        q,
        k,
        v,
        alpha,
        beta_fp32,
        cu_seqlens,
        128**-0.5,
        initial_state=initial_fp32,
        state_checkpoints=checkpoints,
        checkpoint_cu_starts=checkpoint_starts,
        checkpoint_every_n_tokens=page_size,
        max_seqlen=max(lengths),
    )
    return output.unsqueeze(0), checkpoints, final, checkpoint_starts
