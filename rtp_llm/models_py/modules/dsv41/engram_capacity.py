# SPDX-License-Identifier: Apache-2.0
"""Trusted local capacity policy; no CUDA, allocation or checkpoint imports.

The hard ceilings are development safety bounds, not available-memory claims.
Only a local launcher may raise the small defaults. A manifest or launch bundle
must never supply these limits. Budget counts unique padded backing, not RSS,
file cache, other model weights or driver mapping overhead.
"""
from __future__ import annotations

import os
import re


DEFAULT_MAX_ALLOCATION_BYTES = 1 << 30
DEFAULT_MAX_TOTAL_ALLOCATION_BYTES = 8 << 30
HARD_MAX_ALLOCATION_BYTES = 128 << 30
HARD_MAX_TOTAL_ALLOCATION_BYTES = 256 << 30
MAX_ALLOCATION_ENV = "RTP_ENGRAM_MAX_ALLOCATION_BYTES"
MAX_TOTAL_ALLOCATION_ENV = "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"
_DECIMAL = re.compile(r"[1-9][0-9]{0,11}\Z")
_MAX_INTEGER = (1 << 63) - 1


class CapacityError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def validate_capacity(max_allocation_bytes, max_total_allocation_bytes):
    """Return validated local ceilings; a total smaller than per-allocation is legal."""
    for value, ceiling in ((max_allocation_bytes, HARD_MAX_ALLOCATION_BYTES),
                           (max_total_allocation_bytes, HARD_MAX_TOTAL_ALLOCATION_BYTES)):
        if type(value) is not int or not 0 < value <= ceiling:
            raise CapacityError("INVALID_CAPACITY_LIMIT")
    return max_allocation_bytes, max_total_allocation_bytes


def capacity_from_environ(environ=None):
    """Read each trusted local setting once; accept canonical decimal bytes only."""
    env = os.environ if environ is None else environ
    values = []
    missing = object()
    for key, default in ((MAX_ALLOCATION_ENV, DEFAULT_MAX_ALLOCATION_BYTES),
                         (MAX_TOTAL_ALLOCATION_ENV, DEFAULT_MAX_TOTAL_ALLOCATION_BYTES)):
        value = env.get(key, missing)
        if value is missing:
            values.append(default)
        elif type(value) is str and _DECIMAL.fullmatch(value):
            values.append(int(value))
        else:
            raise CapacityError("INVALID_CAPACITY_LIMIT")
    return validate_capacity(*values)


def checked_allocation_bytes(logical_bytes, granularity, max_allocation_bytes):
    """Round without overflow; callers apply total-generation budget separately."""
    if (type(logical_bytes) is not int or not 0 < logical_bytes <= _MAX_INTEGER
            or type(granularity) is not int or not 0 < granularity <= HARD_MAX_ALLOCATION_BYTES
            or type(max_allocation_bytes) is not int
            or not 0 < max_allocation_bytes <= HARD_MAX_ALLOCATION_BYTES):
        raise CapacityError("INVALID_CAPACITY_LIMIT")
    # Avoid computing an unchecked sum in fixed-width implementations as well.
    if logical_bytes > max_allocation_bytes:
        raise CapacityError("ALLOCATION_LIMIT")
    padding = (-logical_bytes) % granularity
    if padding > max_allocation_bytes - logical_bytes:
        raise CapacityError("ALLOCATION_LIMIT")
    return logical_bytes + padding
