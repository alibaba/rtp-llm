"""CPU interval plans between padded natural shards and shared Zigzag rows."""

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class RowSegment:
    """Receive ``rows`` from a peer's local source into local destination rows."""

    peer: int
    source_offset: int
    destination_offset: int
    rows: int


def _coalesce(segments):
    ordered = sorted(segments, key=lambda segment: segment.destination_offset)
    result = []
    for segment in ordered:
        if result:
            previous = result[-1]
            if (
                previous.peer == segment.peer
                and previous.source_offset + previous.rows == segment.source_offset
                and previous.destination_offset + previous.rows
                == segment.destination_offset
            ):
                result[-1] = RowSegment(
                    previous.peer,
                    previous.source_offset,
                    previous.destination_offset,
                    previous.rows + segment.rows,
                )
                continue
        result.append(segment)
    return tuple(result)


class NaturalZigzagPlan:
    """Describe receives into one rank, retaining every physical padding row.

    Natural physical rows concatenate each padded request chronologically,
    then split into equal contiguous rank shards. Each Zigzag rank receives
    its front and mirrored back interval from each request, in request order,
    matching ZigZagProcessor.generateQKVRestoreIndices.

    ``to_zigzag`` names natural source peers; ``to_natural`` names Zigzag
    source peers. Both tuples use local offsets and ascending destination
    order, include self copies and coalesce contiguous transfers.
    """

    def __init__(self, lengths, size, rank):
        self.size, self.rank = index(size), index(rank)
        if self.size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= self.rank < self.size:
            raise ValueError("rank must be in [0, size)")
        self.lengths = tuple(index(length) for length in lengths)
        if any(length < 0 for length in self.lengths):
            raise ValueError("request lengths must be nonnegative")

        alignment = 2 * self.size
        self.padded_lengths = tuple(
            (length + alignment - 1) // alignment * alignment for length in self.lengths
        )
        self.physical_rows = sum(self.padded_lengths)
        self.local_rows = self.physical_rows // self.size
        to_zigzag, to_natural = [], []
        natural_shard_start = self.rank * self.local_rows
        natural_shard_end = natural_shard_start + self.local_rows
        request_start = chunk_offset = 0

        for padded_length in self.padded_lengths:
            pair = padded_length // alignment
            if not pair:
                continue
            for zigzag_rank in range(self.size):
                starts = (
                    request_start + zigzag_rank * pair,
                    request_start + padded_length - (zigzag_rank + 1) * pair,
                )
                for half, start in enumerate(starts):
                    end = start + pair
                    zigzag_offset = chunk_offset + half * pair

                    # Split only at natural-shard boundaries, never per row.
                    if zigzag_rank == self.rank:
                        cursor = start
                        while cursor < end:
                            peer, source_offset = divmod(cursor, self.local_rows)
                            rows = min(end - cursor, self.local_rows - source_offset)
                            to_zigzag.append(
                                RowSegment(
                                    peer,
                                    source_offset,
                                    zigzag_offset + cursor - start,
                                    rows,
                                )
                            )
                            cursor += rows

                    # The inverse receives exactly the overlap with our shard.
                    overlap_start = max(start, natural_shard_start)
                    overlap_end = min(end, natural_shard_end)
                    if overlap_start < overlap_end:
                        to_natural.append(
                            RowSegment(
                                zigzag_rank,
                                zigzag_offset + overlap_start - start,
                                overlap_start - natural_shard_start,
                                overlap_end - overlap_start,
                            )
                        )
            request_start += padded_length
            chunk_offset += pair * 2

        self.to_zigzag = _coalesce(to_zigzag)
        self.to_natural = _coalesce(to_natural)
