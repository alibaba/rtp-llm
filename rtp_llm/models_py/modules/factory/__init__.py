"""
Factory modules - modules with different implementations based on config/arch.
Provides factories to hide the selection process.
"""

from importlib import import_module

__all__ = [
    "FusedMoeFactory",
    "LinearFactory",
    "AttnImplFactory",
    "FMHAImplBase",
]

_EXPORT_MODULES = {
    "FusedMoeFactory": ".fused_moe",
    "LinearFactory": ".linear",
    "AttnImplFactory": ".attention",
    "FMHAImplBase": ".attention",
}


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
