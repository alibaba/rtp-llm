"""Strict field and numeric contracts shared by configuration and execution."""

import math
from dataclasses import dataclass, replace
from typing import Optional
from document_io import DocumentError


def fail(path, message):
    raise DocumentError(f"{path}: {message}")


class FieldSetError(ValueError):
    """Structured field differences let input boundaries keep their diagnostics."""
    def __init__(self, path, extra, missing):
        self.extra, self.missing = sorted(extra), sorted(missing)
        message = (f"{path}: unknown configuration fields {self.extra}" if extra
                   else f"missing YAML parameter {path}.{self.missing[0]}")
        super().__init__(message)


def mapping_fields(value, allowed, path, *, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: mapping keys must be strings")
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        raise FieldSetError(path, extra, missing)
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
