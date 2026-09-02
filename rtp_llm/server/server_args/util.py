import argparse

from rtp_llm.ops import CPRotateMethod, RdmaDeviceHealthFaultHandler


def int_in_range(min_value: int, max_value: int, what: str):
    """Build an argparse ``type`` that parses an int and rejects out-of-range values."""

    def parse(value):
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                f"{what} must be an integer: {value!r}"
            ) from error
        if not min_value <= parsed <= max_value:
            raise argparse.ArgumentTypeError(
                f"{what} must be in [{min_value}, {max_value}], got {parsed}"
            )
        return parsed

    return parse


def str2_rdma_device_health_fault_handler(value):
    """Convert string to RdmaDeviceHealthFaultHandler enum (trim + case-insensitive)."""
    if value is None:
        return None
    if isinstance(value, RdmaDeviceHealthFaultHandler):
        return value
    try:
        return RdmaDeviceHealthFaultHandler.__members__[value.strip().upper()]
    except KeyError as error:
        raise argparse.ArgumentTypeError(
            f"invalid RDMA device health fault handler: {value!r}; expected one of: "
            f"{', '.join(RdmaDeviceHealthFaultHandler.__members__)}"
        ) from error


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
