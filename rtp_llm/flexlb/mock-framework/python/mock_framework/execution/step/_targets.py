"""Common target selection argument."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Targets:
    targets: list[str] | None = None

    def __post_init__(self):
        if self.targets is not None and (
            not self.targets or len(set(self.targets)) != len(self.targets)
        ):
            raise ValueError("targets must be nonempty and unique when supplied")
