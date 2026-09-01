from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ZigzagRelayStep:
    """One contiguous stateful-attention call in global token order."""

    owner_rank: int
    first_global_segment: int
    first_local_segment: int
    segment_count: int = 1

    def valid_token_count(self, segment_valid_lengths: tuple[int, ...]) -> int:
        end = self.first_global_segment + self.segment_count
        if len(segment_valid_lengths) < end:
            raise ValueError(
                f"segment_valid_lengths must contain at least {end} entries, "
                f"got {len(segment_valid_lengths)}"
            )
        return sum(segment_valid_lengths[self.first_global_segment : end])


@dataclass(frozen=True)
class ZigzagCPPlan:
    """Topology shared by stateful linear-attention CP implementations.

    A sequence is split into ``2 * cp_size`` segments. Rank ``r`` stores
    segment ``r`` followed by segment ``2 * cp_size - 1 - r``. Stateful
    attention must nevertheless consume the segments in global order.
    """

    cp_size: int
    cp_rank: int

    def __post_init__(self) -> None:
        if self.cp_size < 2:
            raise ValueError(f"cp_size must be at least 2, got {self.cp_size}")
        if not 0 <= self.cp_rank < self.cp_size:
            raise ValueError(
                f"cp_rank must be in [0, {self.cp_size}), got {self.cp_rank}"
            )

    @property
    def global_segment_count(self) -> int:
        return 2 * self.cp_size

    @property
    def local_global_segments(self) -> tuple[int, int]:
        return self.cp_rank, self.global_segment_count - 1 - self.cp_rank

    @property
    def halo_sources(
        self,
    ) -> tuple[Optional[tuple[int, int]], tuple[int, int]]:
        """Return predecessor ``(rank, local_segment)`` for both local segments.

        ``None`` denotes the sequence boundary, whose halo comes from the
        reusable prefix state (or zeros when no prefix exists).
        """
        front_source = None if self.cp_rank == 0 else (self.cp_rank - 1, 0)
        back_source = (
            (self.cp_rank, 0)
            if self.cp_rank == self.cp_size - 1
            else (self.cp_rank + 1, 1)
        )
        return front_source, back_source

    @property
    def relay_steps(self) -> tuple[ZigzagRelayStep, ...]:
        """Return the minimal ordered relay schedule for this CP size.

        The innermost rank owns the two adjacent middle segments, so they are
        combined into one call. Every other rank contributes one front and one
        back call, yielding ``2 * cp_size - 1`` calls and one fewer transfers.
        """
        front = tuple(
            ZigzagRelayStep(
                owner_rank=rank,
                first_global_segment=rank,
                first_local_segment=0,
            )
            for rank in range(self.cp_size - 1)
        )
        middle = (
            ZigzagRelayStep(
                owner_rank=self.cp_size - 1,
                first_global_segment=self.cp_size - 1,
                first_local_segment=0,
                segment_count=2,
            ),
        )
        back = tuple(
            ZigzagRelayStep(
                owner_rank=rank,
                first_global_segment=self.global_segment_count - 1 - rank,
                first_local_segment=1,
            )
            for rank in range(self.cp_size - 2, -1, -1)
        )
        return front + middle + back


def get_segment_valid_lengths(
    actual_tokens: int, segment_tokens: int, cp_size: int
) -> tuple[int, ...]:
    """Return real-token counts for every padded global zigzag segment."""
    if actual_tokens < 0:
        raise ValueError(f"actual_tokens must be non-negative, got {actual_tokens}")
    if segment_tokens <= 0:
        raise ValueError(f"segment_tokens must be positive, got {segment_tokens}")
    if cp_size < 2:
        raise ValueError(f"cp_size must be at least 2, got {cp_size}")
    capacity = 2 * cp_size * segment_tokens
    if actual_tokens > capacity:
        raise ValueError(
            f"actual_tokens ({actual_tokens}) exceeds CP capacity ({capacity})"
        )

    return tuple(
        max(0, min(segment_tokens, actual_tokens - i * segment_tokens))
        for i in range(2 * cp_size)
    )
