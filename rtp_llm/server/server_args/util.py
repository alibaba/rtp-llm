import argparse
import math
from typing import Optional

from rtp_llm.ops import CPRotateMethod
from rtp_llm.utils.pre_import_config import str2bool

# Largest MiB count that still converts to a size_t byte count, mirroring
# checkedMiBToBytes in rtp_llm/cpp/cache/MemoryEvaluationHelper.cc: size_t max
# divided by 1024 * 1024 bytes-per-MiB, i.e. 2^64 / 2^20 == 2^44, minus one.
# Shared so argument definitions and their tests cannot drift apart by each
# re-deriving it (tests used to recompute it from struct.calcsize("P")).
MAX_RUNTIME_MEMORY_MIB = (1 << 44) - 1


def nonnegative_int(value: str, max_value: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"value must be an integer, got {value!r}"
        ) from error
    if parsed < 0 or (max_value is not None and parsed > max_value):
        constraint = (
            f"be in [0, {max_value}]"
            if max_value is not None
            else "be non-negative"
        )
        raise argparse.ArgumentTypeError(
            f"value must {constraint}, got {value!r}"
        )
    return parsed


def nonnegative_float(
    value: str,
    max_value: Optional[float] = None,
    max_value_exclusive: bool = False,
) -> float:
    """Parse a finite non-negative float, optionally with an upper bound.

    Bind the bound with functools.partial at the add_argument site rather than
    writing a bespoke validator per argument: a local copy tends to lose the
    isfinite check or the ``got {value!r}`` echo that makes a rejection
    diagnosable from the error text alone.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"value must be a number, got {value!r}"
        ) from error
    too_large = max_value is not None and (
        parsed >= max_value if max_value_exclusive else parsed > max_value
    )
    if not math.isfinite(parsed) or parsed < 0.0 or too_large:
        if max_value is None:
            constraint = "be a finite non-negative number"
        else:
            upper = f"{max_value})" if max_value_exclusive else f"{max_value}]"
            constraint = f"be finite and in [0, {upper}"
        raise argparse.ArgumentTypeError(f"value must {constraint}, got {value!r}")
    return parsed


def greater_than_one_float(value: str) -> float:
    """Parse a finite float strictly greater than 1.0."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"value must be a number, got {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError(
            f"value must be finite and greater than 1.0, got {value!r}"
        )
    return parsed


def str2_cp_rotate_method(value):
    """Convert string to CPRotateMethod enum."""
    if value is None:
        return None
    if isinstance(value, CPRotateMethod):
        return value
    value_upper = value.upper()
    if value_upper == "ALL_GATHER":
        return CPRotateMethod.ALL_GATHER
    elif value_upper == "ALL_GATHER_WITH_OVERLAP":
        return CPRotateMethod.ALL_GATHER_WITH_OVERLAP
    elif value_upper == "ALLTOALL":
        return CPRotateMethod.ALLTOALL
    elif value_upper == "PREFILL_CP":
        return CPRotateMethod.PREFILL_CP
    else:
        raise ValueError(
            f"Invalid cp_rotate_method: {value}. "
            f"Must be one of: ALL_GATHER, ALL_GATHER_WITH_OVERLAP, ALLTOALL, PREFILL_CP"
        )
