"""Qwen3 DSpark draft model (models_py) — DFlash base + Markov head.

Extends the DFlash block-diffusion draft (qwen3_dflash.py) with a low-rank
Markov transition-bias head that overrides stage D: from the anchor, each
position adds a low-rank bias (from the previously sampled token) to the base
logits before the greedy pick, recovering the "what did the previous token
actually sample" serial dependency that the one-shot block forward drops.

Mirrors upstream vLLM's split (qwen3_dspark.py: DSparkMarkovHead +
Qwen3DSparkModel(DFlashQwen3Model)).  Stages A/B/C and the propose/forward
skeleton are inherited unchanged from Qwen3DFlashModel; only markov_correct is
overridden.  The sampling loop lives in the model (not the executor) per the
phase-1 design's G3 contract (docs/dspark-two-phase-plan-2026-07-14.md §2.3):
that keeps it inside the draft CUDA-graph capture boundary and lets the
phase-2 confidence head co-locate with the Markov head.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.qwen3_dflash import (
    DSparkDraftParams,
    DSparkProposal,
    Qwen3DFlashModel,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


class DSparkMarkovHead(nn.Module):
    """Low-rank Markov transition-bias head (DSpark stage D).

    markov_w1[token] embeds the previously sampled token (target vocab,
    [V1, rank]); markov_w2 projects it to a draft-vocab bias ([V2, rank]).
    Kept in fp32 like the reference oracle: the per-position argmax feeds the
    next step, so the correction chain is decision-critical.
    """

    def __init__(self, markov_w1: torch.Tensor, markov_w2: torch.Tensor):
        super().__init__()
        self.markov_w1 = markov_w1  # [V1, rank]
        self._markov_w2_f32 = markov_w2.float()  # [V2, rank]

    def bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """[B] int64 previously-sampled tokens -> [B, V2] fp32 transition bias."""
        return F.linear(self.markov_w1[prev_tokens.long()].float(), self._markov_w2_f32)


class Qwen3DSparkModel(Qwen3DFlashModel):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        assert self.dspark_params.markov_rank > 0, (
            "Qwen3DSparkModel requires markov_rank > 0; register a markov-free "
            "draft as qwen_3_dflash (Qwen3DFlashModel)"
        )
        markov_w1 = weights.get_global_weight_or_none(W.dspark_markov_w1)
        markov_w2 = weights.get_global_weight_or_none(W.dspark_markov_w2)
        assert markov_w1 is not None and markov_w2 is not None, (
            "markov_rank > 0 but markov head weights missing from ckpt"
        )
        self.markov_head = DSparkMarkovHead(markov_w1, markov_w2)

    def markov_correct(
        self, base_logits: torch.Tensor, anchor_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DSpark stage D: left-to-right greedy chain with low-rank bias.

        base_logits: [B, k, V]; anchor_ids: [B] int64.
        Returns (tokens [B, k] int64, corrected_logits [B, k, V] fp32).
        Plain python loop over k — small, and CUDA-graph capturable later
        (vLLM captures the same loop in its FULL graph).
        """
        corrected = base_logits.float()
        k = corrected.shape[1]
        prev = anchor_ids.long()
        tokens = torch.empty(
            (corrected.shape[0], k), dtype=torch.int64, device=corrected.device
        )
        for i in range(k):
            corrected[:, i] += self.markov_head.bias(prev)
            prev = corrected[:, i].argmax(dim=-1)
            tokens[:, i] = prev
        return tokens, corrected


__all__ = [
    "DSparkDraftParams",
    "DSparkProposal",
    "DSparkMarkovHead",
    "Qwen3DSparkModel",
]
