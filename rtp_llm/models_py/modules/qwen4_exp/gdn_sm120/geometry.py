# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlashInfer team.
# Extracted from FlashInfer 0.7.0 delta_rule_sm120.py; see PROVENANCE.json.


def can_implement(
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    element_size: int,
) -> bool:
    ratio = (
        num_q_heads // num_v_heads
        if num_q_heads > num_v_heads
        else num_v_heads // num_q_heads
    )
    is_gva_enabled = num_v_heads > num_q_heads
    is_gqa_like = (
        num_k_heads == num_v_heads
        and num_q_heads == ratio * num_k_heads
        and num_q_heads == ratio * num_v_heads
    )
    is_gva_like = (
        num_q_heads == num_k_heads
        and num_v_heads == ratio * num_q_heads
        and num_v_heads == ratio * num_k_heads
    )
    alignment = 16 // element_size
    return (
        ((not is_gva_enabled and is_gqa_like) or (is_gva_enabled and is_gva_like))
        and head_size <= 128
        and head_size % alignment == 0
    )
