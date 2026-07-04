"""Mock latency and cache-key helpers for DeepSeek-V4-Flash replay."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

BLOCK_SIZE = 1024
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT64_MOD = 1 << 64

PREFILL_BASE_MS = 213.058760744
PREFILL_H2048_COEF = 0.000420120401621
PREFILL_H24576_COEF = 0.00817215761679
PREFILL_HIT_COEF = -0.000373217058264
PREFILL_HAS_HIT_COEF = -10.6141559328
PREFILL_CP_COEF = 2.84762280669e-08

DEFAULT_DECODE_BATCH = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_DECODE_STEP = [
    13.04,
    13.04,
    14.06,
    15.33,
    14.53,
    16.99,
    19.51,
    22.45,
    26.05,
    36.88,
]


@dataclass
class RequestShape:
    request_id: int
    input_len: int
    output_len: int
    block_keys: List[int]
    hit_tokens: int = 0


class PerformanceModel:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.block_size = int(cfg.get("block_size", BLOCK_SIZE))
        self.sleep_scale = float(cfg.get("sleep_scale", 1.0))
        prefill = cfg.get("prefill", {})
        decode = cfg.get("decode", {})
        self.prefill_fixed_ms = _optional_float(prefill.get("fixed_ms"))
        self.prefill_scale = float(prefill.get("scale", 1.0))
        pairs = decode.get("step_ms_by_batch")
        if pairs:
            pairs = sorted((int(x), float(y)) for x, y in pairs)
            self.decode_batch = [x for x, _ in pairs]
            self.decode_step = [y for _, y in pairs]
        else:
            self.decode_batch = list(DEFAULT_DECODE_BATCH)
            self.decode_step = list(DEFAULT_DECODE_STEP)
        self.decode_scale = float(decode.get("scale", 1.0))

    @classmethod
    def from_file(cls, path: str | None) -> "PerformanceModel":
        if not path:
            return cls({})
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def prefill_ms(self, requests: Sequence[RequestShape]) -> float:
        if not requests:
            return 0.0
        if self.prefill_fixed_ms is not None:
            return self.prefill_fixed_ms * self.prefill_scale

        # Batch prefill is not a sum of per-request latencies. This approximation
        # keeps the dominant long request plus a small aggregate/batch overhead.
        singles = [
            prefill_ttft_ms(r.input_len, r.hit_tokens, self.block_size)
            for r in requests
        ]
        compute_tokens = sum(max(0, r.input_len - r.hit_tokens) for r in requests)
        batch_overhead = 12.0 * math.log2(len(requests) + 1)
        aggregate_overhead = 0.00008 * max(
            0, compute_tokens - max(r.input_len for r in requests)
        )
        return max(singles) * self.prefill_scale + batch_overhead + aggregate_overhead

    def decode_ms(self, output_len: int, active_batch_size: int) -> float:
        if output_len <= 0:
            return 0.0
        return (
            output_len
            * _interp(self.decode_batch, self.decode_step, active_batch_size)
            * self.decode_scale
        )

    def first_decode_step_ms(self, active_batch_size: int) -> float:
        return (
            _interp(self.decode_batch, self.decode_step, active_batch_size)
            * self.decode_scale
        )

    def sleep_seconds(self, latency_ms: float) -> float:
        return max(0.0, latency_ms * self.sleep_scale / 1000.0)


def load_performance_config(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def observed_prefill_reuse_tokens(
    input_len: int, cached_tokens: int, block_size: int = BLOCK_SIZE
) -> int:
    if input_len <= 0 or cached_tokens < block_size:
        return 0
    block_aligned_hit = (cached_tokens // block_size) * block_size
    if block_aligned_hit >= input_len:
        return max(0, input_len - block_size)
    return min(block_aligned_hit, input_len)


def prefill_ttft_ms(
    input_len: int, cached_tokens: int, block_size: int = BLOCK_SIZE
) -> float:
    if input_len <= 0:
        return 0.0
    hit = observed_prefill_reuse_tokens(input_len, cached_tokens, block_size)
    uncached = max(0, input_len - hit)
    has_hit = 1 if hit > 0 else 0
    h2048 = max(0, uncached - 2048)
    h24576 = max(0, uncached - 24576)
    return (
        PREFILL_BASE_MS
        + PREFILL_H2048_COEF * h2048
        + PREFILL_H24576_COEF * h24576
        + PREFILL_HIT_COEF * hit
        + PREFILL_HAS_HIT_COEF * has_hit
        + PREFILL_CP_COEF * uncached * hit
    )


def compute_block_keys(
    token_ids: Sequence[int], block_size: int = BLOCK_SIZE
) -> List[int]:
    """Stable block hash fallback used when a trace has input_ids but no bh."""

    keys: List[int] = []
    for i in range(0, len(token_ids) // block_size * block_size, block_size):
        h = hashlib.blake2b(digest_size=8)
        block = token_ids[i : i + block_size]
        for token in block:
            h.update(int(token).to_bytes(4, byteorder="little", signed=True))
        keys.append(
            to_signed_int64(
                int.from_bytes(h.digest(), byteorder="little", signed=False)
            )
        )
    return keys


def to_signed_int64(value) -> int:
    """Normalize unsigned log values to protobuf int64 two's-complement range."""

    n = int(value)
    if INT64_MIN <= n <= INT64_MAX:
        return n
    n = n % UINT64_MOD
    if n > INT64_MAX:
        n -= UINT64_MOD
    return n


def synthetic_token_ids(input_len: int) -> List[int]:
    # Keep token material simple; route/cache locality comes from bh encoded in
    # GenerateConfigPB.unique_key when the source trace provides block hashes.
    return [0] * max(0, input_len)


def _interp(xs: Sequence[int], ys: Sequence[float], x: int) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, x) - 1
    t = (x - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + t * (ys[i + 1] - ys[i])


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)
