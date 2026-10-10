"""One strict field-set check, shared by configuration and action adapters."""


def mapping_fields(value, allowed, path, *, required=(), error=ValueError):
    if not isinstance(value, dict):
        raise error(f"{path}: expected mapping")
    extra = set(value) - set(allowed)
    if extra:
        raise error(f"{path}: unknown configuration fields {sorted(extra)}")
    missing = set(required) - set(value)
    if missing:
        raise error(f"missing YAML parameter {path}.{sorted(missing)[0]}")
    return value
