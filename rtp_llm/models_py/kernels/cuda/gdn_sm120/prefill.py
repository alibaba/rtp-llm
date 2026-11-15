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
    q,
    k,
    v,
    alpha,
    beta_fp32,
    initial,
    cu_seqlens,
    lengths=None,
    page_size=128,
    *,
    prepared=None
):
    """Run SM120 GDN; prepared mode may zero caller-owned Q/K/V padding."""
    from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
        checkpoint_starts as make_checkpoint_starts,
    )

    heads = max(q.shape[1], v.shape[1])
    if page_size != 128:
        raise ValueError("SM120 prefill checkpoint pages must contain 128 tokens")
    if prepared is None:
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
        max_seqlen = max(lengths)
    else:
        if (
            q.shape[0] != prepared.token_capacity
            or cu_seqlens.data_ptr() != prepared.cu_seqlens.data_ptr()
        ):
            raise ValueError(
                "SM120 prepared GDN requires its fixed token capacity and cu buffer"
            )
        prepared.bind_sm120_cp(heads)
        from rtp_llm.models_py.triton_kernels.fla.gdn_prefill_prepare import (
            clear_sm120_prefill_padding,
        )

        # TMA descriptors cover capacity, so NaN padding would enter partial
        # tiles before arithmetic masking. Restore eager's zero OOB values;
        # live rows are never written, and this launch/grid is capture-stable.
        clear_sm120_prefill_padding(q, k, v, prepared.num_tokens)
        checkpoint_starts = prepared.checkpoint_starts
        checkpoints = prepared.get_workspace(
            "gdn_sm120_checkpoints",
            (prepared.token_capacity // page_size, heads, 128, 128),
            torch.float32,
            q.device,
        )
        final = prepared.get_workspace(
            "gdn_sm120_final",
            (prepared.batch_capacity, heads, 128, 128),
            torch.float32,
            q.device,
        )
        output = prepared.get_workspace(
            "gdn_sm120_output", (q.shape[0], heads, 128), q.dtype, q.device
        )
        # Only live rows are produced by the DSL; keep downstream padding inert.
        output.zero_()
        max_seqlen = prepared.token_capacity
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
        max_seqlen=max_seqlen,
        workspace=prepared,
    )
    return output.unsqueeze(0), checkpoints, final, checkpoint_starts
