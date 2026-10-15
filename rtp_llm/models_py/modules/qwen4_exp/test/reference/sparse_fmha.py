import torch


def sparse_prefill_attn_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Matches :func:`sparse_prefill_attn` up to fp rounding."""
    from rtp_llm.models_py.modules.qwen4_exp.test.reference.sparse_attention import (
        gather_and_attend,
    )

    return gather_and_attend(q, k, v, selected)
