import argparse

from rtp_llm.ops import CPRotateMethod


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


def str2ratio(v):
    """Parse a float restricted to [0.0, 1.0] (rejects nan/inf/out-of-range)."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            value = float(v)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"Float value expected, got: {v!r}")
    else:
        value = float(v)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Value must be within [0.0, 1.0], got: {value}"
        )
    return value


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
