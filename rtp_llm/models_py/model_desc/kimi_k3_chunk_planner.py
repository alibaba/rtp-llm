"""Pure-Python planning for Kimi K3 whole-model chunk Prefill."""

from __future__ import annotations

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
    page_size: int,
) -> tuple[KimiK3ChunkRound, ...]:
    """Split a packed Prefill batch without stranding reusable linear state.

    Every non-terminal slice ends at an absolute ``page_size`` boundary. A
    terminal slice may end in the request's partial tail block.
    """

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
    if page_size <= 0:
        raise ValueError(f"K3 linear page size must be positive, got {page_size}")
    if any(prefix < 0 for prefix in prefixes):
        raise ValueError(f"K3 prefix lengths must be non-negative, got {prefixes}")
    if chunk_budget < page_size and any(length > chunk_budget for length in lengths):
        raise ValueError(
            "K3 chunk budget must cover one linear page when a request spans "
            f"rounds: budget={chunk_budget}, page={page_size}"
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
                aligned_end = ((absolute_start + available) // page_size) * page_size
                take = aligned_end - absolute_start
                if take <= 0:
                    continue

            absolute_start = prefixes[request_idx] + done
            absolute_end = absolute_start + take
            if not terminal and absolute_end % page_size:
                raise AssertionError(
                    "non-terminal K3 chunk slice is not page aligned: "
                    f"request={request_idx} end={absolute_end} page={page_size}"
                )
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
            pending = [
                idx for idx, (done, total) in enumerate(zip(processed, lengths))
                if done < total
            ]
            raise RuntimeError(
                "K3 chunk budget cannot advance any pending request to a linear "
                f"page boundary: budget={chunk_budget}, page={page_size}, "
                f"pending={pending}"
            )
        rounds.append(KimiK3ChunkRound(tuple(round_slices)))

    return tuple(rounds)


__all__ = [
    "KimiK3ChunkRound",
    "KimiK3ChunkSlice",
    "plan_kimi_k3_chunk_rounds",
]
