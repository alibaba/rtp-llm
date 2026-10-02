"""CPU planning for K3 whole-model prefill rounds.

Adapted from the K3 dev planner. Alignment is the caller's ordinary joint cache
checkpoint unit, independent of TP. SP padding is applied separately per round.
This module plans real rows only; it does not execute or publish cache state.
"""

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class KimiK3ChunkSlice:
    original_batch_idx: int
    source_start: int
    source_end: int
    prefix_length: int
    processed_length: int
    new_length: int
    absolute_start: int
    absolute_end: int
    terminal: bool


@dataclass(frozen=True)
class KimiK3ChunkRound:
    slices: tuple[KimiK3ChunkSlice, ...]

    @property
    def token_count(self) -> int:
        return sum(item.new_length for item in self.slices)


def _source_offsets(lengths: Sequence[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        if int(length) <= 0:
            raise ValueError(f"K3 chunk input lengths must be positive, got {length}")
        offsets.append(offsets[-1] + int(length))
    return offsets


def plan_kimi_k3_chunk_rounds(
    input_lengths: Sequence[int],
    prefix_lengths: Sequence[int],
    *,
    chunk_budget: int,
    alignment_tokens: int,
) -> tuple[KimiK3ChunkRound, ...]:
    """Split a packed Prefill batch at absolute joint-checkpoint boundaries."""

    lengths = [int(value) for value in input_lengths]
    prefixes = [int(value) for value in prefix_lengths]
    if len(lengths) != len(prefixes):
        raise ValueError(
            "K3 chunk input/prefix batch sizes differ: "
            f"input={len(lengths)} prefix={len(prefixes)}"
        )
    if not lengths:
        raise ValueError("K3 chunk planner requires at least one request")
    if chunk_budget <= 0:
        raise ValueError(f"K3 chunk budget must be positive, got {chunk_budget}")
    if alignment_tokens <= 0:
        raise ValueError(
            f"K3 chunk checkpoint alignment must be positive, got {alignment_tokens}"
        )
    if any(prefix < 0 for prefix in prefixes):
        raise ValueError(f"K3 prefix lengths must be non-negative, got {prefixes}")
    if any(prefix % alignment_tokens for prefix in prefixes):
        raise ValueError(
            "whole-model K3 chunk Prefill requires checkpoint-aligned prefixes: "
            f"prefixes={prefixes} alignment={alignment_tokens}"
        )
    if chunk_budget < alignment_tokens and any(
        length > chunk_budget for length in lengths
    ):
        raise ValueError(
            "K3 chunk budget must reach one joint checkpoint when a request "
            f"spans rounds: budget={chunk_budget}, checkpoint={alignment_tokens}"
        )

    source_offsets = _source_offsets(lengths)
    processed = [0] * len(lengths)
    rounds: list[KimiK3ChunkRound] = []
    while any(done < total for done, total in zip(processed, lengths)):
        available = chunk_budget
        round_slices: list[KimiK3ChunkSlice] = []
        for request_idx, total_length in enumerate(lengths):
            done = processed[request_idx]
            remaining = total_length - done
            if remaining <= 0 or available <= 0:
                continue

            terminal = remaining <= available
            if terminal:
                take = remaining
            else:
                absolute_start = prefixes[request_idx] + done
                aligned_end = (
                    (absolute_start + available) // alignment_tokens
                ) * alignment_tokens
                take = aligned_end - absolute_start
                if take <= 0:
                    continue

            absolute_start = prefixes[request_idx] + done
            absolute_end = absolute_start + take
            source_start = source_offsets[request_idx] + done
            round_slices.append(
                KimiK3ChunkSlice(
                    original_batch_idx=request_idx,
                    source_start=source_start,
                    source_end=source_start + take,
                    prefix_length=prefixes[request_idx],
                    processed_length=done,
                    new_length=take,
                    absolute_start=absolute_start,
                    absolute_end=absolute_end,
                    terminal=terminal,
                )
            )
            processed[request_idx] += take
            available -= take

        if not round_slices:
            raise RuntimeError("K3 chunk planner made no progress")
        rounds.append(KimiK3ChunkRound(tuple(round_slices)))
    return tuple(rounds)
