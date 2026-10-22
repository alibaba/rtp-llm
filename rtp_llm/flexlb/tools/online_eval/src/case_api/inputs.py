"""Bounded program input groups; case code owns each group's field contract."""

from scenario.config_data import mapping


def fields(data, required, path, *, optional=()):
    """Validate case-owned nested mappings with the same strict field semantics."""
    return mapping(data, set(required) | set(optional), path, required=required)
