import argparse

from rtp_llm.ops import CPRotateMethod, CPProcessorType


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


def str2_cp_processor_type(value):
    """Convert string to CPProcessorType enum."""
    if value is None:
        return None
    if isinstance(value, CPProcessorType):
        return value
    value_upper = value.upper()
    if value_upper == "ZIG_ZAG":
        return CPProcessorType.ZIG_ZAG
    elif value_upper == "ROUND_ROBIN":
        return CPProcessorType.ROUND_ROBIN
    else:
        raise ValueError(
            f"Invalid cp_processor_type: {value}. "
            f"Must be one of: ZIG_ZAG, ROUND_ROBIN"
        )


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
