"""Instance-local loader strategies supplied before any tensor is loaded."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class WeightPreparation:
    split_strategies: tuple[tuple[str, Callable], ...] = ()
    # Same (name, process function, checkpoint count) key as the loader's
    # optional preshard optimization; this does not change split semantics.
    preshard_layouts: tuple[tuple[tuple, tuple], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "split_strategies", tuple(self.split_strategies))
        object.__setattr__(self, "preshard_layouts", tuple(self.preshard_layouts))
        if len(dict(self.split_strategies)) != len(self.split_strategies):
            raise ValueError("Duplicate weight split strategy")
        if any(not callable(fn) for _, fn in self.split_strategies):
            raise TypeError("Weight split strategy must be callable")

    def split_function(self, name, default):
        return dict(self.split_strategies).get(name, default)

    def preshard_layout(self, key, default):
        return dict(self.preshard_layouts).get(key, default)
