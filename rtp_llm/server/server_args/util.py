import argparse
from dataclasses import dataclass
from typing import Any, Optional

from rtp_llm.ops import CPRotateMethod


@dataclass(frozen=True)
class BoundedInt:
    """Argparse integer converter carrying its display and base-type metadata."""

    minimum: int
    maximum: Optional[int] = None

    def __call__(self, value: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError("value must be an integer") from error
        if parsed < self.minimum:
            raise argparse.ArgumentTypeError(f"value must be at least {self.minimum}")
        if self.maximum is not None and parsed > self.maximum:
            raise argparse.ArgumentTypeError(f"value must be at most {self.maximum}")
        return parsed


def bounded_int(minimum: int, maximum: Optional[int] = None) -> BoundedInt:
    """Build an argparse converter with deterministic, source-aware errors."""

    return BoundedInt(minimum, maximum)


def argument_base_type(converter: Any) -> Any:
    """Return the primitive type represented by one argparse converter."""

    return int if isinstance(converter, BoundedInt) else converter


def argument_metavar(converter: Any) -> Optional[str]:
    """Derive a stable display name without probing converter attributes."""

    if isinstance(converter, BoundedInt):
        return "INT"
    if isinstance(converter, type) and issubclass(converter, bool):
        return "BOOL"
    if isinstance(converter, type) and issubclass(converter, int):
        return "INT"
    if isinstance(converter, type) and issubclass(converter, str):
        return "STR"
    return None


positive_int32 = bounded_int(1, 2**31 - 1)
non_negative_int64 = bounded_int(0, 2**63 - 1)


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "on"):
        return True
    if v.lower() in ("no", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


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
