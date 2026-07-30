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
phase-1 design's G3 contract:
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
    Weights stay in the ckpt dtype (bf16): the GEMM accumulates in fp32 and
    only the [B, V2] output is upcast, so the serial chain reads k * V2 * rank
    bf16 bytes per round instead of fp32 — at rank 256 that halves ~1.1GB/round
    of weight traffic, the largest single cost in the draft tail after lm_head.
    The += into the fp32 corrected logits still happens in fp32; on the real
    golden decision path the bf16 bias perturbation (~3e-2) sits well inside
    the per-step argmax margins (0.44+), and a flipped draft token can only
    lower the acceptance rate, never break output correctness (upstream vLLM /
    SGLang run this GEMM in bf16 as well).
    """

    def __init__(self, markov_w1: torch.Tensor, markov_w2: torch.Tensor):
        super().__init__()
        self.markov_w1 = markov_w1  # [V1, rank]
        self.markov_w2 = markov_w2  # [V2, rank]

    def bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """[B] int64 previously-sampled tokens -> [B, V2] fp32 transition bias."""
        emb = self.markov_w1[prev_tokens.long()]
        return F.linear(emb, self.markov_w2).float()


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
        Plain python loop over k — small, pure tensor ops, captured in the
        default full-tail draft graph (vLLM captures the same loop in its
        FULL graph).
        """
        corrected = base_logits.float()
        if corrected is base_logits:
            # .float() on an already-fp32 lm_head (enable_fp32_lm_head) returns
            # an alias; the in-place += below must not mutate the caller's
            # pre-correction base_logits.
            corrected = corrected.clone()
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
