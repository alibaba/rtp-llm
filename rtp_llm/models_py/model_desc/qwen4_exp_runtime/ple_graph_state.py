"""Captured stages outlive individual speculative commits."""

from dataclasses import dataclass, field

from .ple_state import _PLETargetTransaction


@dataclass
class PLEGraphRegistry:
    captured: dict[int, _PLETargetTransaction] = field(default_factory=dict)

    def clear(self) -> None:
        self.captured.clear()
