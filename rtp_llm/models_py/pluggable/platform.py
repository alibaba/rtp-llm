"""Platform facts captured once, after the worker has assigned its device."""

from dataclasses import dataclass
from typing import Tuple

from rtp_llm.device.device_type import DeviceType, get_device_type


@dataclass(frozen=True)
class PlatformContext:
    device_type: DeviceType
    device_name: str
    local_rank: int
    evidence: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self):
        if not isinstance(self.device_type, DeviceType):
            raise TypeError("PlatformContext requires the existing DeviceType enum")
        if type(self.local_rank) is not int or self.local_rank < 0:
            raise ValueError("PlatformContext requires a nonnegative local rank")
        object.__setattr__(
            self, "evidence", tuple(tuple(item) for item in self.evidence)
        )

    @classmethod
    def detect(cls, *, local_rank: int, requested: str = "auto", created_device=None):
        import os

        import torch

        detected = get_device_type()
        names = {kind.name.lower(): kind for kind in DeviceType}
        if requested != "auto":
            if requested not in names:
                raise ValueError(f"Unknown module platform {requested!r}")
            if names[requested] != detected:
                raise RuntimeError(f"Requested {requested}, detected {detected.name}")
        if created_device is not None and created_device != detected:
            raise RuntimeError(
                f"Created device {created_device!r} conflicts with {detected.name}"
            )
        accelerator = detected in (DeviceType.Cuda, DeviceType.ROCm, DeviceType.Ppu)
        if accelerator and torch.cuda.current_device() != local_rank:
            raise RuntimeError("Capture platform only after setting the worker device")
        return cls(
            detected,
            torch.cuda.get_device_name(local_rank) if accelerator else detected.name,
            local_rank,
            (
                ("detector", "rtp_llm.device.device_type.get_device_type"),
                ("torch_version", str(torch.__version__)),
                ("ppu_home_present", str(bool(os.environ.get("PPU_HOME")))),
            ),
        )
