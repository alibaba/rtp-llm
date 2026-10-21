"""Reusable assemblies of base and factory modules, loaded on first use."""

from importlib import import_module

_MODULES = {
    "CausalAttention": "causal_attention",
    "DenseMLP": "dense_mlp",
    "MlaAttention": "mla_attention",
}
__all__ = list(_MODULES)


def __getattr__(name: str):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
