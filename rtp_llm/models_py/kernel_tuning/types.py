from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KernelTuningStatus:
    overlay: str
    applied: bool
    reason: str
    dependency_version: Optional[str] = None
