# Copyright (c) 2026 Alibaba Cloud.
#
# RTP-LLM first-party Gemma4 full-attention kernel package (split-D prototype).
# Kernel logic is adapted from FlashInfer's CuTe-DSL Blackwell FMHA (see
# cute_fmha_split_d.py header for the per-file source mapping); those files
# carry NVIDIA's BSD-3-Clause notice.

from .cute_fmha_split_d import LOG2_E, Gemma4SplitDApplyKernel, Gemma4SplitDStatsKernel
from .gemma4_split_d_prefill import (
    Gemma4SplitDPrefillWrapper,
    gemma4_split_d_attention,
    gemma4_split_d_attention_support,
    gemma4_split_d_stats,
    gemma4_split_d_support,
    run_gemma4_split_d_prefill,
)

__all__ = [
    "Gemma4SplitDApplyKernel",
    "Gemma4SplitDStatsKernel",
    "Gemma4SplitDPrefillWrapper",
    "gemma4_split_d_attention",
    "gemma4_split_d_attention_support",
    "gemma4_split_d_stats",
    "gemma4_split_d_support",
    "run_gemma4_split_d_prefill",
    "LOG2_E",
]
