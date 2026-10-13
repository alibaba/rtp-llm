import argparse

from rtp_llm.ops import CPAllGatherImpl, CPRotateMethod


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


def str2_cp_all_gather_impl(value):
    """Convert a string to the configured all-gather CP implementation."""
    if isinstance(value, CPAllGatherImpl):
        return value
    value_upper = value.upper()
    if value_upper == "LEGACY":
        return CPAllGatherImpl.LEGACY
    if value_upper == "FUSED":
        return CPAllGatherImpl.FUSED
    raise argparse.ArgumentTypeError(
        f"Invalid cp_all_gather_impl: {value}. Must be one of: LEGACY, FUSED"
    )
