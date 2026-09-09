"""Immutable Device facts shared by startup, model selection and allocation."""

from dataclasses import dataclass

from .device_type import DeviceType, get_device_type


def validate_requested_device(requested):
    if requested != "auto" and requested not in {
        kind.name.lower() for kind in DeviceType
    }:
        raise ValueError(f"Unknown module dispatch platform {requested!r}")


@dataclass(frozen=True)
class DeviceRuntimeContext:
    device_type: DeviceType
    device_name: str
    local_rank: int
    evidence: tuple[tuple[str, str], ...] = ()
    device_string: str = ""
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self):
        if not isinstance(self.device_type, DeviceType):
            raise TypeError("Runtime context requires the existing DeviceType enum")
        if type(self.local_rank) is not int or self.local_rank < 0:
            raise ValueError("Runtime context requires a nonnegative local rank")
        object.__setattr__(
            self, "evidence", tuple(tuple(item) for item in self.evidence)
        )
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))

    @classmethod
    def detect(cls, *, local_rank, requested="auto", created_device=None):
        from rtp_llm.device import get_cached_device_type, get_current_device

        validate_requested_device(requested)
        detected = get_device_type()
        if requested != "auto" and requested != detected.name.lower():
            raise RuntimeError(f"Requested {requested}, detected {detected.name}")
        device = get_current_device()
        for identity in (created_device, get_cached_device_type()):
            if identity is not None and identity != detected:
                raise RuntimeError(
                    f"Created device {identity!r} conflicts with {detected.name}"
                )
        context = device.runtime_context(local_rank)
        if context.device_type != detected or context.local_rank != local_rank:
            raise RuntimeError(
                "Device runtime facts conflict with assigned worker device"
            )
        return context
