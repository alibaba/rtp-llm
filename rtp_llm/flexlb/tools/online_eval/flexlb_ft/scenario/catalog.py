"""Explicit builtin adapter registration. Scenario data never imports Python."""

from .actions.elastic import HANDLERS as ELASTIC_HANDLERS
from .actions.engine_control import HANDLERS as ENGINE_CONTROL_HANDLERS
from .actions.engine_fault import HANDLERS as ENGINE_FAULT_HANDLERS
from .actions.master import HANDLERS as MASTER_HANDLERS
from .actions.observation import HANDLERS as OBSERVATION_HANDLERS


def handlers():
    result = {}
    for descriptor in [
        *ELASTIC_HANDLERS,
        *ENGINE_CONTROL_HANDLERS,
        *ENGINE_FAULT_HANDLERS,
        *MASTER_HANDLERS,
        *OBSERVATION_HANDLERS,
    ]:
        if descriptor.name in result:
            raise ValueError(f"duplicate action {descriptor.name}")
        result[descriptor.name] = descriptor
    return result
