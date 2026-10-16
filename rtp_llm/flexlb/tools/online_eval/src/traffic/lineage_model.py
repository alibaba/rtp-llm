"""Prefix identity model with explicit token lengths and allocation semantics."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator

MAX_LINEAGE_EVENTS = 1_000_000
DEFAULT_PREFIX_BLOCK_SIZE = 512


def prefix_block_size(value):
    """Replay supports power-of-two prefix blocks from 64 to 4096 tokens."""
    if type(value) is not int or not 64 <= value <= 4096 or value & (value - 1):
        raise ValueError("block_size must be a power of two in [64, 4096]")
    return value


class LengthSemantics(str, Enum):
    BLOCK_ALIGNED = "BLOCK_ALIGNED_LOSSY_TOKENS"
    EXACT = "EXACT_LENGTH_PRIVATE_PARTIAL_BLOCK"


@dataclass(frozen=True)
class LineageMetadata:
    version: int
    block_size: int
    count: int
    realism: str
    tail: LengthSemantics
    arrival: str
    provenance: dict
    token_adjustment: dict | None = None
    allocation_bitmap: str | None = None

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict):
            raise ValueError("invalid lineage metadata")
        try:
            result = cls(**dict(value, tail=LengthSemantics(value["tail"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid lineage metadata") from exc
        prefix_block_size(result.block_size)
        if (type(result.version) is not int or result.version not in (2, 3)
                or type(result.count) is not int
                or not 1 <= result.count <= MAX_LINEAGE_EVENTS or not isinstance(result.provenance, dict)
                or result.tail != (LengthSemantics.BLOCK_ALIGNED if result.version == 2 else LengthSemantics.EXACT)):
            raise ValueError("invalid lineage metadata")
        return result

    def to_dict(self):
        result = dict(version=self.version, block_size=self.block_size, count=self.count,
                      realism=self.realism, tail=self.tail.value, arrival=self.arrival,
                      provenance=self.provenance)
        if self.token_adjustment is not None:
            result["token_adjustment"] = self.token_adjustment
        if self.allocation_bitmap is not None:
            result["allocation_bitmap"] = self.allocation_bitmap
        return result


@dataclass(frozen=True)
class LineageEvent:
    timestamp_ms: int
    input_tokens: int
    parent_index: int
    shared_full_blocks: int
    reserved_labels: int = 0


@dataclass(frozen=True)
class LineageModel:
    metadata: LineageMetadata
    events: tuple[LineageEvent, ...]

    def __post_init__(self):
        if len(self.events) != self.metadata.count:
            raise ValueError("lineage model count mismatch")
        previous = 0
        for index, event in enumerate(self.events):
            if (any(type(value) is not int for value in (event.timestamp_ms, event.input_tokens,
                    event.parent_index, event.shared_full_blocks, event.reserved_labels))
                    or not previous <= event.timestamp_ms <= 9223372036854775807
                    or not 1 <= event.input_tokens <= 2147483647
                    or not -1 <= event.parent_index < index
                    or not 0 <= event.shared_full_blocks <= event.input_tokens // self.metadata.block_size
                    or event.reserved_labels not in (0, 1)):
                raise ValueError("invalid lineage event")
            if ((event.parent_index == -1 and event.shared_full_blocks)
                    or (event.parent_index >= 0 and event.shared_full_blocks >
                        self.events[event.parent_index].input_tokens // self.metadata.block_size)):
                raise ValueError("invalid lineage shared prefix")
            if self.metadata.tail == LengthSemantics.BLOCK_ALIGNED:
                if event.input_tokens % self.metadata.block_size:
                    raise ValueError("block-aligned lineage requires whole blocks")
            elif event.reserved_labels:
                raise ValueError("exact lineage cannot reserve discarded labels")
            previous = event.timestamp_ms

    @classmethod
    def from_wire(cls, metadata, events):
        header = LineageMetadata.read(metadata)
        if header.version == 2:
            typed = tuple(LineageEvent(ts, blocks * header.block_size, parent, shared, reserve)
                          for ts, blocks, parent, shared, reserve in events)
        else:
            typed = tuple(LineageEvent(*event) for event in events)
        return cls(header, typed)

    def to_wire(self):
        if self.metadata.version == 2:
            events = [(event.timestamp_ms, event.input_tokens // self.metadata.block_size, event.parent_index,
                       event.shared_full_blocks, event.reserved_labels) for event in self.events]
        else:
            events = [[event.timestamp_ms, event.input_tokens, event.parent_index,
                       event.shared_full_blocks] for event in self.events]
        return self.metadata.to_dict(), events


@dataclass(frozen=True)
class ExpandedRequest:
    source_index: int
    timestamp_ms: int
    input_tokens: int
    block_labels: tuple[int, ...]


@dataclass(frozen=True)
class LineageExpander:
    model: LineageModel
    label_limit: int = 2147483647
    retained_blocks: int | None = None

    def __post_init__(self):
        if self.retained_blocks is not None and (type(self.retained_blocks) is not int or self.retained_blocks < 1):
            raise ValueError("retained_blocks must be positive")

    def __iter__(self) -> Iterator[ExpandedRequest]:
        paths: list[tuple[int, ...]] = []
        next_label = 1
        for index, event in enumerate(self.model.events):
            prefix = paths[event.parent_index][:event.shared_full_blocks] if event.parent_index >= 0 else ()
            fresh = (event.input_tokens + self.model.metadata.block_size - 1) // self.model.metadata.block_size - event.shared_full_blocks
            if next_label + fresh + event.reserved_labels > self.label_limit:
                raise ValueError("lineage label budget exceeded")
            retained = fresh if self.retained_blocks is None else max(0, self.retained_blocks - len(prefix))
            labels = prefix + tuple(range(next_label, next_label + min(fresh, retained)))
            next_label += fresh + event.reserved_labels
            paths.append(labels)
            yield ExpandedRequest(index, event.timestamp_ms, event.input_tokens, labels)
