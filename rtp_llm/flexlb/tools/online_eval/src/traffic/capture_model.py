"""Anonymous capture records shared by extraction and fitting.

Timestamps are arrival epoch ms. Keys are cumulative SHA256 prefixes (128 bits)
from little-endian int32 tokens; only full blocks participate in sharing.
Missing observations remain null. Hashes are anonymization, not encryption.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import math
import re

from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE, prefix_block_size

SCHEMA_VERSION = 1
HEX = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True)
class CaptureMetadata:
    schema_version: int = SCHEMA_VERSION
    block_size: int = DEFAULT_PREFIX_BLOCK_SIZE

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        prefix_block_size(self.block_size)

    @classmethod
    def read(cls, value, location="capture metadata"):
        try:
            return cls(value["schema_version"], value["block_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{location}: invalid capture metadata: {exc}") from None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CaptureRecord:
    schema_version: int
    block_size: int
    ts: int
    il: int
    ol: int | None
    keys: tuple[str, ...]
    tail_hash: str
    rid: str
    status: str | None
    error: int | str | None
    cached: int | None
    aux_reuse: int | None
    latency_ms: float | None
    priority: int | None
    max_new_tokens: int | None
    timeout_ms: int | None

    def __post_init__(self):
        CaptureMetadata(self.schema_version, self.block_size)
        for name in ('ts', 'il'):
            if type(getattr(self, name)) is not int:
                raise ValueError('illegal type for ' + name)
        if not 1 <= self.il <= 2147483647 or not 0 <= self.ts <= 9223372036854775807:
            raise ValueError('invalid il/ts range')
        for name in ('ol', 'cached', 'aux_reuse', 'priority', 'max_new_tokens', 'timeout_ms'):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ValueError('illegal type for ' + name)
            if value is not None and not 0 <= value <= 2147483647:
                raise ValueError('invalid range for ' + name)
        if self.priority is not None and not 1 <= self.priority <= 100:
            raise ValueError('invalid range for priority')
        for name in ('cached', 'aux_reuse'):
            if getattr(self, name) is not None and getattr(self, name) > self.il:
                raise ValueError('invalid range for ' + name)
        if self.status is not None and type(self.status) is not str:
            raise ValueError('illegal type for status')
        if self.error is not None and type(self.error) not in (int, str):
            raise ValueError('illegal type for error')
        if self.latency_ms is not None and (type(self.latency_ms) not in (int, float)
                or not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError('invalid latency_ms')
        if type(self.keys) is not tuple or len(self.keys) != self.il // self.block_size:
            raise ValueError('keys/il block count mismatch')
        if any(type(v) is not str or not HEX.fullmatch(v)
               for v in (*self.keys, self.tail_hash, self.rid)):
            raise ValueError('invalid 128-bit digest')

    @classmethod
    def read(cls, value, location="capture row", *, metadata=None):
        if not isinstance(value, dict):
            raise ValueError(f"{location}: expected JSON object")
        expected = {f.name for f in fields(cls)}
        missing, unknown = expected - value.keys(), value.keys() - expected
        if missing:
            raise ValueError(f"{location}: missing required field {sorted(missing)[0]}")
        if unknown:
            raise ValueError(f"{location}: unknown field {sorted(unknown)[0]}")
        try:
            if type(value['keys']) is not list:
                raise ValueError('illegal type for keys')
            result = cls(**dict(value, keys=tuple(value['keys'])))
            if metadata is not None and CaptureMetadata(result.schema_version, result.block_size) != metadata:
                raise ValueError('row metadata differs from capture summary')
            return result
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{location}: {exc}") from None

    def to_dict(self):
        result = asdict(self)
        result['keys'] = list(self.keys)
        return result
