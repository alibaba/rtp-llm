"""Declarative scenario APIs; importing primitives does not load the compiler."""

from scenario.errors import ScenarioError

__all__ = ["ScenarioError", "load_scenarios", "compile_scenarios"]


def __getattr__(name):
    if name == "load_scenarios":
        from scenario.loader import load_scenarios
        return load_scenarios
    if name == "compile_scenarios":
        from scenario.compiler import compile_scenarios
        return compile_scenarios
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
