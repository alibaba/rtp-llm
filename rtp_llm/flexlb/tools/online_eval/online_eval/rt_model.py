"""Mock latency and cache-key helpers for DeepSeek-V4-Flash replay."""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

BLOCK_SIZE = 1024
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT64_MOD = 1 << 64

DEFAULT_DECODE_BATCH = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_DECODE_STEP = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
]

logger = logging.getLogger("rt_model")


@dataclass
class RequestShape:
    request_id: int
    input_len: int
    output_len: int
    block_keys: List[int]
    hit_tokens: int = 0


# ---------------------------------------------------------------------------
# Prefill formula — parsed from the Master PREFILL_TIME_FORMULA config string.
#
# The formula string lives in ONE place: the master config JSON
# (data/config/master_fixed_window_220ms.json, env var PREFILL_TIME_FORMULA).
# The mock engine reads it from there (via --master-config) and parses it
# here, so there is no duplicate copy of the coefficients in Python.
#
# Expected formula structure:
#   max(MIN, BASE + BATCH_COEF*log(batchSize + 1)
#     + COEF_i*SCALE_i*log(1 + exp((sum(computeTokens) - THRESH_i)/SCALE_i))  [+ ...]
#     + HIT_COEF*sum(hitCacheTokens)
#     + HAS_HIT_COEF*(sum(hasHitCache)/batchSize)
#     + HIT_RATIO_COEF*(sum(hitCacheTokens/(inputTokens + 1))/batchSize))
# ---------------------------------------------------------------------------

# Regex for a floating-point number (handles decimal and scientific notation).
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"

# Softplus term:  COEF*SCALE*log(1 + exp((sum(computeTokens) - THRESH)/SCALE))
_SOFTPLUS_RE = re.compile(
    rf"({_NUM})\s*\*\s*({_NUM})\s*\*\s*"
    r"log\(\s*1\s*\+\s*exp\(\s*\(\s*sum\(computeTokens\)\s*-\s*"
    rf"({_NUM})\s*\)\s*/\s*"
    rf"({_NUM})\s*\)\s*\)"
)


@dataclass
class PrefillFormulaCoeffs:
    """Coefficients parsed from a PREFILL_TIME_FORMULA string."""

    min_ms: float
    base: float
    batch_coef: float
    softplus_terms: List[tuple]  # list of (coef, scale, threshold)
    hit_coef: float
    has_hit_coef: float
    hit_ratio_coef: float


def parse_prefill_formula(formula_str: str) -> PrefillFormulaCoeffs:
    """Parse a PREFILL_TIME_FORMULA string into coefficients.

    The formula string is the single source of truth — it comes from the
    master config JSON (env var PREFILL_TIME_FORMULA), the same string the
    Java FormulaPredictor parses.  This function extracts the numeric
    coefficients so the mock engine can evaluate the formula in Python.

    Raises ValueError if the formula does not match the expected structure.
    """
    s = re.sub(r"\s+", " ", formula_str).strip()

    # max(MIN, BASE + ...)
    m = re.match(rf"max\(\s*({_NUM})\s*,\s*({_NUM})\s*\+", s)
    if not m:
        raise ValueError(
            f"Cannot parse MIN/BASE from formula (expected 'max(NUM, NUM + ...'): "
            f"{formula_str[:80]!r}"
        )
    min_ms = float(m.group(1))
    base = float(m.group(2))

    # BATCH_COEF — COEF*log(batchSize + 1)
    batch_m = re.search(rf"({_NUM})\s*\*\s*log\(\s*batchSize\s*\+\s*1\s*\)", s)
    if not batch_m:
        raise ValueError(f"Cannot parse BATCH_COEF from formula: {formula_str[:80]!r}")
    batch_coef = float(batch_m.group(1))

    # Softplus terms: COEF*SCALE*log(1+exp((sum(computeTokens)-THRESH)/SCALE))
    softplus_terms: List[tuple] = []
    for match in _SOFTPLUS_RE.finditer(s):
        coef, scale_mult, thresh, scale_div = match.groups()
        sm = float(scale_mult)
        sd = float(scale_div)
        if sm != sd:
            raise ValueError(
                f"Softplus term has inconsistent scale: {sm} vs {sd} in formula"
            )
        softplus_terms.append((float(coef), sm, float(thresh)))

    # HIT_COEF — COEF*sum(hitCacheTokens)
    hit_m = re.search(rf"({_NUM})\s*\*\s*sum\(\s*hitCacheTokens\s*\)", s)
    if not hit_m:
        raise ValueError(f"Cannot parse HIT_COEF from formula: {formula_str[:80]!r}")
    hit_coef = float(hit_m.group(1))

    # HAS_HIT_COEF — COEF*(sum(hasHitCache)/batchSize)
    has_hit_m = re.search(
        rf"({_NUM})\s*\*\s*\(\s*sum\(\s*hasHitCache\s*\)\s*/\s*batchSize\s*\)", s
    )
    if not has_hit_m:
        raise ValueError(
            f"Cannot parse HAS_HIT_COEF from formula: {formula_str[:80]!r}"
        )
    has_hit_coef = float(has_hit_m.group(1))

    # HIT_RATIO_COEF — COEF*(sum(hitCacheTokens/(inputTokens + 1))/batchSize)
    hit_ratio_m = re.search(
        rf"({_NUM})\s*\*\s*\(\s*sum\(\s*hitCacheTokens\s*/\s*"
        rf"\(\s*inputTokens\s*\+\s*1\s*\)\s*\)\s*/\s*batchSize\s*\)",
        s,
    )
    if not hit_ratio_m:
        raise ValueError(
            f"Cannot parse HIT_RATIO_COEF from formula: {formula_str[:80]!r}"
        )
    hit_ratio_coef = float(hit_ratio_m.group(1))

    return PrefillFormulaCoeffs(
        min_ms=min_ms,
        base=base,
        batch_coef=batch_coef,
        softplus_terms=softplus_terms,
        hit_coef=hit_coef,
        has_hit_coef=has_hit_coef,
        hit_ratio_coef=hit_ratio_coef,
    )


def _eval_prefill_formula(
    coeffs: PrefillFormulaCoeffs,
    batch_size: int,
    sum_compute_tokens: float,
    sum_hit_cache_tokens: float,
    sum_has_hit_cache: float,
    sum_hit_cache_over_input: float,
) -> float:
    """Evaluate the prefill formula with the given batch statistics."""
    if batch_size <= 0:
        return 0.0

    latency = coeffs.base + coeffs.batch_coef * math.log(batch_size + 1)
    for coef, scale, threshold in coeffs.softplus_terms:
        latency += (
            coef
            * scale
            * math.log(1 + math.exp((sum_compute_tokens - threshold) / scale))
        )
    latency += coeffs.hit_coef * sum_hit_cache_tokens
    latency += coeffs.has_hit_coef * (sum_has_hit_cache / batch_size)
    latency += coeffs.hit_ratio_coef * (sum_hit_cache_over_input / batch_size)
    return max(coeffs.min_ms, latency)


class PerformanceModel:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.block_size = int(cfg.get("block_size", BLOCK_SIZE))
        self.sleep_scale = float(cfg.get("sleep_scale", 1.0))
        prefill = cfg.get("prefill", {})
        decode = cfg.get("decode", {})
        self.prefill_fixed_ms = _optional_float(prefill.get("fixed_ms"))
        self.prefill_scale = float(prefill.get("scale", 1.0))

        # Prefill formula string — single source of truth is the master config
        # JSON (PREFILL_TIME_FORMULA env var).  The mock engine reads it via
        # --master-config and injects it here as prefill.formula_str.
        formula_str = prefill.get("formula_str")
        if formula_str:
            self._formula_coeffs = parse_prefill_formula(formula_str)
        else:
            self._formula_coeffs = None
            if self.prefill_fixed_ms is None:
                logger.warning(
                    "No prefill formula and no prefill.fixed_ms — "
                    "prefill timing defaults to 300ms"
                )

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

        if self._formula_coeffs is None:
            return 300.0 * self.prefill_scale

        # Aggregate batch statistics for the prefill formula.
        #   computeTokens_i  = max(0, input_len_i - hit_tokens_i)
        #   hitCacheTokens_i = hit_tokens_i
        #   hasHitCache_i    = 1 if hit_tokens_i > 0 else 0
        batch_size = len(requests)
        sum_compute = 0.0
        sum_hit = 0.0
        sum_has_hit = 0.0
        sum_hit_over_input = 0.0
        for r in requests:
            hit = r.hit_tokens
            compute = max(0, r.input_len - hit)
            sum_compute += compute
            sum_hit += hit
            if hit > 0:
                sum_has_hit += 1
            sum_hit_over_input += hit / (r.input_len + 1)

        latency = _eval_prefill_formula(
            self._formula_coeffs,
            batch_size=batch_size,
            sum_compute_tokens=sum_compute,
            sum_hit_cache_tokens=sum_hit,
            sum_has_hit_cache=sum_has_hit,
            sum_hit_cache_over_input=sum_hit_over_input,
        )
        return latency * self.prefill_scale

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


def extract_prefill_formula_from_master_config(
    master_config_path: str | None,
) -> str | None:
    """Read PREFILL_TIME_FORMULA from a master config JSON file.

    The master config has the structure:
      {"zone_process_setting": {"process_info": {"envs": [[key, value], ...]}}}
    """
    if not master_config_path:
        return None
    with open(master_config_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    envs = (
        payload.get("zone_process_setting", {}).get("process_info", {}).get("envs", [])
    )
    for item in envs:
        if (
            isinstance(item, list)
            and len(item) == 2
            and item[0] == "PREFILL_TIME_FORMULA"
        ):
            return item[1]
    return None


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
