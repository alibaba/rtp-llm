"""Strict field and numeric contracts shared by configuration and execution."""

import math
from dataclasses import dataclass, replace
from typing import Optional


def mapping_fields(value, allowed, path, *, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    extra = set(value) - set(allowed)
    if extra:
        raise ValueError(f"{path}: unknown configuration fields {sorted(extra)}")
    missing = set(required) - set(value)
    if missing:
        raise ValueError(f"missing YAML parameter {path}.{sorted(missing)[0]}")
    return value


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class NumberRule:
    integer: bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self):
        if (type(self.integer) is not bool
                or any(v is not None and not finite_number(v)
                       for v in (self.minimum, self.maximum))
                or self.minimum is not None and self.maximum is not None
                   and self.minimum > self.maximum):
            raise ValueError("invalid program number rule")

    def validate(self, value, path):
        if (type(value) not in ((int,) if self.integer else (int, float))
                or not finite_number(value)
                or self.minimum is not None and value < self.minimum
                or self.maximum is not None and value > self.maximum):
            raise ValueError(f"parameter {path!r}: invalid value {value!r}")
        return value

    def narrow(self, spec, path):
        mapping_fields(spec, {"minimum", "maximum"}, "parameter_schema." + path)
        for key, value in spec.items():
            if not finite_number(value):
                raise ValueError("parameter_schema requires finite, increasing bounds")
            base = getattr(self, key)
            if base is not None and (value < base if key == "minimum" else value > base):
                raise ValueError(f"parameter_schema.{path}: cannot weaken the program's {key}")
        try:
            return replace(self, **spec)
        except ValueError as exc:
            raise ValueError("parameter_schema requires finite, increasing bounds") from exc
