"""Inject immutable registries without maintaining a framework case-name list."""
from pathlib import Path
from case_api import registry


def entry(name, definition, module='fixture.program', path=None):
    return registry.RegisteredCase(name, module, Path(path or __file__), definition)


def snapshot(*entries, include_existing=False):
    current = list(registry.registry().cases.values()) if include_existing else []
    replacements = {case.name for case in entries}
    return registry.CaseRegistry.build([case for case in current if case.name not in replacements] + list(entries))
