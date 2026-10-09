"""Foundational actions and capabilities declared by registered Python cases."""

import importlib
from dataclasses import replace

from cases.registry import PROGRAMS
from scenario.actions.engine_control import HANDLERS as ENGINE_CONTROL_HANDLERS
from scenario.actions.engine_fault import HANDLERS as ENGINE_FAULT_HANDLERS
from scenario.actions.environment import HANDLERS as ENVIRONMENT_HANDLERS
from scenario.actions.java_flow import HANDLERS as JAVA_FLOW_HANDLERS
from scenario.actions.master import HANDLERS as MASTER_HANDLERS
from scenario.actions.observation import HANDLERS as OBSERVATION_HANDLERS

FOUNDATION_HANDLERS = (
    *ENGINE_CONTROL_HANDLERS,
    *ENGINE_FAULT_HANDLERS,
    *ENVIRONMENT_HANDLERS,
    *JAVA_FLOW_HANDLERS,
    *MASTER_HANDLERS,
    *OBSERVATION_HANDLERS,
)


def handlers(case=None):
    """Return foundations plus one case's capabilities, or all registered cases."""
    if case is not None and case not in PROGRAMS:
        raise ValueError(f"unknown registered Python case {case!r}")
    result = {}

    def add(descriptor):
        if descriptor.name in result:
            raise ValueError(f"duplicate action {descriptor.name}")
        result[descriptor.name] = descriptor

    for descriptor in FOUNDATION_HANDLERS:
        if descriptor.owners:
            raise ValueError(f"foundational action {descriptor.name!r} has a case owner")
        add(descriptor)
    for name in PROGRAMS if case is None else (case,):
        module = importlib.import_module(PROGRAMS[name])
        for descriptor in getattr(module, "ACTION_HANDLERS", ()):
            if descriptor.owners and descriptor.owners != frozenset({name}):
                raise ValueError(f"action {descriptor.name!r} has a conflicting case owner")
            owned = replace(descriptor, owners=frozenset({name}))
            previous = result.get(descriptor.name)
            if previous is None:
                add(owned)
            elif (
                previous.owners
                and name not in previous.owners
                and replace(previous, owners=frozenset())
                == replace(descriptor, owners=frozenset())
            ):
                # Sharing is explicit: each program declares the same implementation.
                result[descriptor.name] = replace(previous, owners=previous.owners | owned.owners)
            else:
                raise ValueError(f"duplicate action {descriptor.name}")
    return result
