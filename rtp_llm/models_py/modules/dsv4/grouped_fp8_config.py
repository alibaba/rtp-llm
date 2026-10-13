"""Host-only capacity settings shared by Hopper execution and JIT warmup."""

import os

DEFAULT_MASKED_MAX_N = 1024
MASKED_MAX_N_ENV = "DSV4_MOE_MASKED_MAX_N"


def masked_max_n() -> int:
    raw = os.environ.get(MASKED_MAX_N_ENV, str(DEFAULT_MASKED_MAX_N))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{MASKED_MAX_N_ENV} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{MASKED_MAX_N_ENV} must be positive, got {value}")
    return value
