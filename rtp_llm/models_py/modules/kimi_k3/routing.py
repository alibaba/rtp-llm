"""K3 routing: native fused CUDA operation and a CPU reference path."""

import torch


def grouped_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    top_k: int,
    groups: int,
    top_groups: int,
    renormalize: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.is_cuda:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        op = getattr(rtp_llm_ops, "kimi_k3_grouped_topk", None)
        if op is None:
            raise RuntimeError(
                "K3 fused routing requires the CUDA13 native routing binding"
            )
        return op(logits, bias, groups, top_groups, top_k, renormalize, scale)

    scores = logits.sigmoid()
    choice = scores + bias
    if groups > top_groups:
        grouped = choice.reshape(logits.shape[0], groups, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        selected = group_scores.topk(top_groups, sorted=False).indices
        mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
            1, selected, True
        )
        choice = choice.masked_fill(
            ~mask.unsqueeze(-1).expand_as(grouped).reshape_as(choice), float("-inf")
        )
    ids = choice.topk(top_k, sorted=False).indices
    routing = scores.gather(1, ids)
    if renormalize and top_k > 1:
        routing = routing / (routing.sum(-1, keepdim=True) + 1e-20)
    return routing * scale, ids
