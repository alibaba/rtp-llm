"""Pure-Python planning for bounded dense-FlashMLA prefix expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FlashMLAPrefixChunkSpec:
    """One packed historical-prefix chunk before device materialization."""

    request_indices: tuple[int, ...]
    prefix_starts: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    q_start: int
    q_tokens: int

    @property
    def kv_tokens(self) -> int:
        return sum(self.prefix_lens)


def _indptr(lengths: Sequence[int]) -> list[int]:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return values


def plan_flashmla_prefix_chunks(
    q_lens: Sequence[int],
    prefix_lens: Sequence[int],
    *,
    chunk_tokens: int,
    page_size: int,
) -> tuple[FlashMLAPrefixChunkSpec, ...]:
    """Pack page-aligned prefix ranges under a strict total-token capacity."""

    q_values = [int(value) for value in q_lens]
    prefix_values = [int(value) for value in prefix_lens]
    if len(q_values) != len(prefix_values) or not q_values:
        raise ValueError(
            "FlashMLA prefix chunk planner requires matching non-empty Q/prefix "
            f"lengths, got q={q_values} prefix={prefix_values}"
        )
    if any(value <= 0 for value in q_values):
        raise ValueError(f"FlashMLA Q lengths must be positive, got {q_values}")
    if any(value < 0 for value in prefix_values):
        raise ValueError(
            f"FlashMLA prefix lengths must be non-negative, got {prefix_values}"
        )
    if chunk_tokens < 0:
        raise ValueError(
            f"FlashMLA prefix chunk capacity must be non-negative, got {chunk_tokens}"
        )
    if chunk_tokens == 0 or not any(prefix_values):
        return ()
    if page_size <= 0:
        raise ValueError(f"FlashMLA page size must be positive, got {page_size}")
    if chunk_tokens < page_size or chunk_tokens % page_size:
        raise ValueError(
            "FlashMLA prefix chunk capacity must be at least one cache page "
            f"and divisible by it, got chunk={chunk_tokens} page={page_size}"
        )

    q_offsets = _indptr(q_values)
    chunks: list[FlashMLAPrefixChunkSpec] = []
    request_indices: list[int] = []
    prefix_starts: list[int] = []
    chunk_lens: list[int] = []
    capacity = chunk_tokens

    def flush() -> None:
        nonlocal capacity
        if not request_indices:
            return
        first_request = request_indices[0]
        last_request = request_indices[-1]
        chunks.append(
            FlashMLAPrefixChunkSpec(
                request_indices=tuple(request_indices),
                prefix_starts=tuple(prefix_starts),
                prefix_lens=tuple(chunk_lens),
                q_start=q_offsets[first_request],
                q_tokens=q_offsets[last_request + 1] - q_offsets[first_request],
            )
        )
        request_indices.clear()
        prefix_starts.clear()
        chunk_lens.clear()
        capacity = chunk_tokens

    for request_idx, prefix_len in enumerate(prefix_values):
        if prefix_len == 0:
            # Keep every chunk's Q rows contiguous without adding a zero-KV row.
            flush()
            continue
        prefix_start = 0
        while prefix_start < prefix_len:
            if request_indices and request_idx != request_indices[-1] + 1:
                flush()
            remaining = prefix_len - prefix_start
            if remaining <= capacity:
                take = remaining
            else:
                take = (capacity // page_size) * page_size
                if take == 0:
                    flush()
                    continue
            request_indices.append(request_idx)
            prefix_starts.append(prefix_start)
            chunk_lens.append(take)
            prefix_start += take
            capacity -= take
            if capacity == 0 or prefix_start < prefix_len:
                flush()
    flush()

    for chunk in chunks:
        if chunk.kv_tokens <= 0 or chunk.kv_tokens > chunk_tokens:
            raise AssertionError(
                "FlashMLA prefix chunk exceeds its token capacity: "
                f"tokens={chunk.kv_tokens} capacity={chunk_tokens}"
            )
        if any(start % page_size for start in chunk.prefix_starts):
            raise AssertionError(
                f"FlashMLA prefix chunk starts are not page aligned: {chunk}"
            )
    return tuple(chunks)


__all__ = ["FlashMLAPrefixChunkSpec", "plan_flashmla_prefix_chunks"]
