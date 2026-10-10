"""One strict field-set check, shared by configuration and action adapters."""


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
    import math
    return type(value) in (int, float) and math.isfinite(value)
