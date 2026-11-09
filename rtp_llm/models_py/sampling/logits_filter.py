# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Logits-domain filtering with RTP threshold-tie semantics.

The two apply_* functions below are copied unchanged from vLLM revision
 af261bab7f717b6ece30f5ed3213c43974e4a6c4,
vllm/v1/sample/ops/topk_topp_sampler.py. Only imports are adapted.
"""

import torch


def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    # Set non-top-k rows to 1 so that we can gather.
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    # topk.values tensor has shape [batch_size, max_top_k].
    # Convert top k to 0-based index in range [0, max_top_k).
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    # Handle non-topk rows.
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def filter_logits(
    logits: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor
) -> torch.Tensor:
    """Filter CUDA FP32 [B,V] logits using CPU int32/FP32 [B] parameters.

    Zero top-k disables truncation; k above V is clipped to V. Top-p is
    normalized over the top-k survivors. All tokens tied at either cutoff
    survive, matching RTP's probability threshold filters. At p=0 the maximum
    tie group survives. Rows must contain a finite logit and no NaN/+inf.
    The input is never modified. An unfiltered batch can alias the input.
    """
    assert logits.is_cuda and logits.dtype == torch.float32 and logits.ndim == 2
    batch, vocab = logits.shape
    assert vocab > 0
    assert top_k.device.type == "cpu" and top_k.dtype == torch.int32
    assert top_p.device.type == "cpu" and top_p.dtype == torch.float32
    assert top_k.shape == top_p.shape == (batch,)
    assert bool((top_k >= 0).all())
    assert bool(((top_p >= 0) & (top_p <= 1)).all())
    k = top_k.clamp(max=vocab).masked_fill(top_k == 0, vocab)
    use_k = bool((k < vocab).any())
    use_p = bool((top_p < 1).any())
    if batch == 0 or not (use_k or use_p):
        return logits
    topk_logits = apply_top_k_top_p_pytorch(
        logits.clone(), k.to(device=logits.device) if use_k else None, None
    )
    if not use_p:
        return topk_logits
    p = top_p.to(device=logits.device)
    filtered = apply_top_k_top_p_pytorch(topk_logits.clone(), None, p)
    # Upstream top-p selects individual sorted tokens. RTP retains the entire
    # probability threshold group; restore that group using original logits.
    threshold = filtered.masked_fill(torch.isneginf(filtered), float("inf")).amin(
        dim=-1, keepdim=True
    )
    restored = topk_logits.masked_fill(topk_logits < threshold, -float("inf"))
    # The upstream cumsum can mask probabilities that underflow to zero even
    # with p=1. Preserve the unfiltered top-k support for these rows.
    return torch.where((p >= 1).unsqueeze(1), topk_logits, restored)
