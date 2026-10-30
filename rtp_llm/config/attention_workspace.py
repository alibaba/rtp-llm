"""Validation and sizing shared by bounded Prefill KV workspaces."""

import math


def kv_workspace_bytes_from_gib(value: float) -> int:
    try:
        size = float(value) * 1024**3
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Prefill KV workspace must be a positive finite GiB amount"
        ) from error
    if not math.isfinite(size) or size < 0 or (size != 0 and size < 1):
        raise ValueError("Prefill KV workspace must be a nonnegative finite GiB amount")
    return int(size)


def validate_kv_workspace_bytes(value: int, *, allow_unbounded: bool = False) -> int:
    if type(value) is not int or value < 0 or (not allow_unbounded and value == 0):
        raise ValueError("prefill_kv_workspace_budget_bytes must be a positive integer")
    return value
