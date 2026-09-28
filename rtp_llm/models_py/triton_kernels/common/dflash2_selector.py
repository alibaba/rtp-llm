# SPDX-License-Identifier: Apache-2.0
# DFlash2's candidate lattice / conditional walk follows z-lab/dflash
# (07ebd93db9f472af339b644bb70221ad8428328a) and SGLang PR #35371
# (e5a3e4d30fa7abda95bafd2d697f9f9c48566114). This implementation uses
# FP32 edge arithmetic and supports both CUDA and HIP through Triton.

import torch
import triton
import triton.language as tl


@triton.jit
def _edge_scores(
    projected,
    predecessor_table,
    successor_table,
    candidates,
    unary,
    anchors,
    scores,
    SLOTS: tl.constexpr,
    K: tl.constexpr,
    RANK: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    previous_index = tl.program_id(2)
    candidate_offset = (row * SLOTS + slot) * K
    if slot == 0:
        previous = tl.load(anchors + row).to(tl.int64)
    else:
        previous = tl.load(candidates + candidate_offset - K + previous_index)
    r = tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    valid_r = r < RANK
    h = tl.load(projected + (row * SLOTS + slot) * RANK + r, valid_r, 0).to(tl.float32)
    # Invalid anchors may occur in graph padding rows. Never read outside the
    # table. Active request anchors are checked by the caller's token contract.
    a = tl.load(
        predecessor_table + previous * RANK + r,
        valid_r & (previous >= 0) & (previous < VOCAB),
        0,
    ).to(tl.float32)
    successor = tl.load(candidates + candidate_offset + k, k < K, 0)
    b = tl.load(
        successor_table + successor[:, None] * RANK + r[None, :],
        (k[:, None] < K) & valid_r[None, :],
        0,
    ).to(tl.float32)
    edge = tl.sum((a * h)[None, :] * b, axis=1)
    value = tl.load(unary + candidate_offset + k, k < K, 0).to(tl.float32)
    output = ((row * SLOTS + slot) * K + previous_index) * K + k
    tl.store(scores + output, value + edge, k < K)


@triton.jit
def _conditional_walk(
    candidates,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
    tokens,
    probabilities,
    SLOTS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    temperature = tl.load(temperatures + row).to(tl.float32)
    greedy = tl.load(greedy_mask + row) != 0
    previous_index = tl.full((), 0, tl.int32)
    for slot in range(SLOTS):
        base = (row * SLOTS + slot) * K
        value = tl.load(scores + (base + previous_index) * K + k, k < K, float("-inf"))
        best = tl.max(value, axis=0)
        if greedy:
            index = tl.min(tl.where((k < K) & (value == best), k, K), axis=0)
            q = tl.where(k == index, 1.0, 0.0)
        else:
            # Subtract before dividing: finite logits and a small positive
            # temperature must not overflow into inf - inf. Padding rows whose
            # logits are all -inf, and +inf maxima, remain finite as well.
            shifted = tl.where(value == best, 0.0, value - best)
            # A decimal below FLT_MIN is inferred as FP64 by Triton even
            # when the temperature tensor is FP32. Type the bound explicitly
            # so mixed greedy/stochastic control flow keeps q in FP32.
            minimum_temperature = tl.full((), 1.1754943508222875e-38, tl.float32)
            weight = tl.exp(shifted / tl.maximum(temperature, minimum_temperature))
            weight = tl.where(k < K, weight, 0.0)
            q = weight / tl.sum(weight, axis=0)
            u = tl.load(uniforms + row * SLOTS + slot).to(tl.float32)
            index = tl.sum(((u >= tl.cumsum(q, axis=0)) & (k < K)).to(tl.int32), axis=0)
            index = tl.minimum(index, K - 1)
        chosen = tl.load(candidates + base + index)
        tl.store(tokens + row * SLOTS + slot, chosen.to(tl.int32))
        tl.store(probabilities + base + k, q, k < K)
        previous_index = index


def select_candidates(
    projected: torch.Tensor,
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary: torch.Tensor,
    anchors: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    uniforms: torch.Tensor,
):
    """Internal contiguous-tensor entry point; no collectives or host reads.

    Torch allocations during graph capture belong to the graph pool. Every
    result, including every conditional probability, is rewritten on replay.
    """
    batch, slots, top_k = candidate_ids.shape
    rank = projected.shape[-1]
    scores = torch.empty(
        (batch, slots, top_k, top_k), device=projected.device, dtype=torch.float32
    )
    tokens = torch.empty((batch, slots), device=projected.device, dtype=torch.int32)
    q = torch.empty_like(unary, dtype=torch.float32)
    if batch == 0 or slots == 0:
        return tokens, q
    _edge_scores[(batch, slots, top_k)](
        projected,
        predecessor_table,
        successor_table,
        candidate_ids,
        unary,
        anchors,
        scores,
        SLOTS=slots,
        K=top_k,
        RANK=rank,
        VOCAB=predecessor_table.shape[0],
        BLOCK_K=triton.next_power_of_2(top_k),
        BLOCK_R=triton.next_power_of_2(rank),
        num_warps=4,
    )
    _conditional_walk[(batch,)](
        candidate_ids,
        scores,
        uniforms,
        temperatures,
        greedy_mask,
        tokens,
        q,
        SLOTS=slots,
        K=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )
    return tokens, q
