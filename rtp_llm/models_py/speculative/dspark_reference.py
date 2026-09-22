"""Small torch reference for DSpARK's sequential low-rank Markov head.

This is a correctness oracle and weight-free simulation path. Serving uses the
fused C++/CUDA sampler once that part of main is migrated; keeping the formula
here makes the contract testable without a checkpoint or GPU kernel build.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def sample_markov_drafts(
    base_logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    *,
    temperature: Union[float, torch.Tensor] = 1.0,
    draft_to_target: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the runtime ``W1[x_prev] @ W2.T`` chain and sample each row.

    Runtime weight layout is ``W1[target_vocab, rank]`` and
    ``W2[draft_vocab, rank]``. Returned token ids are in target-vocabulary
    space; probabilities remain in draft-vocabulary space, matching the C++
    sampler contract. ``temperature == 0`` is the reference shorthand for the
    serving greedy limit and produces an exact argmax point mass.
    """
    if base_logits.dim() != 3:
        raise ValueError("base_logits must be [batch, width, vocab]")
    batch, width, padded_draft_vocab = base_logits.shape
    if anchors.shape != (batch,):
        raise ValueError(f"anchors must be [{batch}], got {tuple(anchors.shape)}")
    if markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("markov_w1 and markov_w2 must be matrices")
    draft_vocab = int(markov_w2.shape[0])
    if draft_vocab <= 0 or padded_draft_vocab < draft_vocab:
        raise ValueError("base logits do not cover the Markov draft vocabulary")
    if markov_w1.shape[1] != markov_w2.shape[1]:
        raise ValueError("Markov low-rank dimensions do not match")
    if anchors.numel() and (
        int(anchors.min()) < 0 or int(anchors.max()) >= markov_w1.shape[0]
    ):
        raise ValueError("anchor token is outside the Markov vocabulary")

    if draft_to_target is None:
        if draft_vocab > markov_w1.shape[0]:
            raise ValueError("identity draft vocabulary exceeds target vocabulary")
        draft_to_target = torch.arange(draft_vocab, dtype=torch.long)
    if draft_to_target.dim() != 1 or draft_to_target.numel() != draft_vocab:
        raise ValueError("draft_to_target must contain one target id per draft id")
    draft_to_target = draft_to_target.to(device=base_logits.device, dtype=torch.long)
    if draft_to_target.numel() and (
        int(draft_to_target.min()) < 0
        or int(draft_to_target.max()) >= markov_w1.shape[0]
    ):
        raise ValueError("draft_to_target contains an invalid target token id")

    if isinstance(temperature, torch.Tensor):
        temperatures = temperature.to(device=base_logits.device, dtype=torch.float32)
        if temperatures.dim() == 0:
            temperatures = temperatures.expand(batch)
        if temperatures.shape != (batch,):
            raise ValueError(f"temperature must be scalar or [{batch}]")
    else:
        temperatures = torch.full(
            (batch,), float(temperature), device=base_logits.device
        )
    if not torch.isfinite(temperatures).all():
        raise ValueError("temperature must be finite")
    if (temperatures < 0.0).any():
        raise ValueError("temperature must be non-negative")

    previous = anchors.to(device=base_logits.device, dtype=torch.long)
    w1 = markov_w1.to(device=base_logits.device, dtype=torch.float32)
    w2 = markov_w2.to(device=base_logits.device, dtype=torch.float32)
    corrected_rows = []
    sampled_rows = []
    greedy_rows = temperatures == 0.0
    safe_temperatures = torch.where(
        greedy_rows, torch.ones_like(temperatures), temperatures
    )
    for step in range(width):
        bias = w1.index_select(0, previous) @ w2.transpose(0, 1)
        corrected = base_logits[:, step, :draft_vocab].float() + bias
        probabilities = torch.softmax(
            corrected / safe_temperatures.unsqueeze(1), dim=-1
        )
        greedy_tokens = corrected.argmax(dim=-1)
        if greedy_rows.any():
            greedy_probabilities = torch.nn.functional.one_hot(
                greedy_tokens, num_classes=draft_vocab
            ).to(torch.float32)
            probabilities = torch.where(
                greedy_rows.unsqueeze(1), greedy_probabilities, probabilities
            )
        sampled_draft = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).squeeze(1)
        sampled = draft_to_target.index_select(0, sampled_draft)
        corrected_rows.append(probabilities)
        sampled_rows.append(sampled)
        previous = sampled
    return torch.stack(sampled_rows, dim=1), torch.stack(corrected_rows, dim=1)
