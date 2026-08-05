import argparse
import math
from typing import Optional

from rtp_llm.ops import CPRotateMethod
from rtp_llm.utils.pre_import_config import str2bool

# str2bool is a deliberate re-export: argument-group modules import it from here.
# Listing it in __all__ keeps lint autofixes from deleting the "unused" import.
__all__ = [
    "MAX_RUNTIME_MEMORY_MIB",
    "DEFAULT_RESERVER_RUNTIME_MEM_MB",
    "nonnegative_int",
    "nonnegative_float",
    "str2_cp_rotate_method",
    "str2bool",
]

# Largest MiB count that still converts to a size_t byte count, mirroring
# checkedMiBToBytes in rtp_llm/cpp/cache/MemoryEvaluationHelper.cc: size_t max
# divided by 1024 * 1024 bytes-per-MiB, i.e. 2^64 / 2^20 == 2^44, minus one.
# Shared so argument definitions and their tests cannot drift apart by each
# re-deriving it (tests used to recompute it from struct.calcsize("P")).
MAX_RUNTIME_MEMORY_MIB = (1 << 44) - 1

# Default for --reserver_runtime_mem_mb. Unlike the other runtime tuning knobs,
# the default lives at the argparse layer, not in ConfigModules.h: the bare C++
# RuntimeConfig deliberately defaults reserve_runtime_mem_mb to 0 so entrypoints
# bypassing the Python server args reserve nothing implicitly. Comparing against
# a fresh RuntimeConfig() would therefore flag every default deployment as tuned.
DEFAULT_RESERVER_RUNTIME_MEM_MB = 1024


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
    min_value: float = 0.0,
    min_value_exclusive: bool = False,
) -> float:
    """Parse a finite float within [min_value, max_value] (bounds optionally exclusive).

    Defaults keep the historical non-negative contract. Bind bounds with
    functools.partial at the add_argument site rather than writing a bespoke
    validator per argument: a local copy tends to lose the isfinite check or
    the ``got {value!r}`` echo that makes a rejection diagnosable from the
    error text alone.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"value must be a number, got {value!r}"
        ) from error
    too_small = parsed < min_value or (min_value_exclusive and parsed == min_value)
    too_large = max_value is not None and (
        parsed >= max_value if max_value_exclusive else parsed > max_value
    )
    if not math.isfinite(parsed) or too_small or too_large:
        if max_value is None and min_value == 0.0 and not min_value_exclusive:
            constraint = "be a finite non-negative number"
        elif max_value is None:
            bound = "greater than" if min_value_exclusive else "at least"
            constraint = f"be finite and {bound} {min_value}"
        else:
            lower = f"({min_value}" if min_value_exclusive else f"[{min_value}"
            upper = f"{max_value})" if max_value_exclusive else f"{max_value}]"
            constraint = f"be finite and in {lower}, {upper}"
        raise argparse.ArgumentTypeError(f"value must {constraint}, got {value!r}")
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
