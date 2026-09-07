"""Explicit builtin adapter registration. Scenario data never imports Python."""

from .actions.elastic import HANDLERS as ELASTIC_HANDLERS


def handlers():
    result = {}
    for descriptor in ELASTIC_HANDLERS:
        if descriptor.name in result:
            raise ValueError(f"duplicate action {descriptor.name}")
        result[descriptor.name] = descriptor
    return result
