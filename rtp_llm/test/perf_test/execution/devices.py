"""Resolve perf device slots against an allocation made by the GPU wrapper."""

from __future__ import annotations

import os
from typing import Sequence

from rtp_llm.test.perf_test.errors import PlanningError

GPU_LOCKED_DEVICES_ENV = "RTP_LLM_GPU_LOCKED_DEVICES"


def resolve_allocated_devices(devices: Sequence[int]) -> tuple[int, ...]:
    """Keep standalone physical IDs; map wrapped logical slots to locked IDs.

    Only the wrapper's receipt enables mapping. An ordinary inherited CVD is
    not evidence of a GPU-lock allocation and does not change standalone IDs.
    """
    requested = tuple(devices)
    if not requested or any(
        type(device) is not int or device < 0 for device in requested
    ):
        raise PlanningError(
            "perf devices must be non-empty non-negative integer identifiers"
        )
    if len(set(requested)) != len(requested):
        raise PlanningError("perf devices must contain unique identifiers")
    if GPU_LOCKED_DEVICES_ENV not in os.environ:
        return requested

    receipt = os.environ[GPU_LOCKED_DEVICES_ENV]
    if not receipt.strip():
        raise PlanningError(f"{GPU_LOCKED_DEVICES_ENV} allocation receipt is empty")
    try:
        allocated = tuple(int(device.strip()) for device in receipt.split(","))
    except ValueError as error:
        raise PlanningError(
            f"{GPU_LOCKED_DEVICES_ENV} must contain comma-separated non-negative GPU IDs"
        ) from error
    if any(device < 0 for device in allocated) or len(set(allocated)) != len(allocated):
        raise PlanningError(
            f"{GPU_LOCKED_DEVICES_ENV} must contain unique non-negative GPU IDs"
        )
    if len(requested) > len(allocated):
        raise PlanningError(
            f"perf requires {len(requested)} devices but the wrapper locked only {len(allocated)}"
        )
    if any(slot >= len(allocated) for slot in requested):
        raise PlanningError(
            f"perf device slots {requested} are outside the wrapper's {len(allocated)} locked devices"
        )
    return tuple(allocated[slot] for slot in requested)
