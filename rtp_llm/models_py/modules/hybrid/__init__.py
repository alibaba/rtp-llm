"""
Hybrid modules - assembly of base/factory modules for reuse across different models.
These modules are architecture-agnostic at this level and compose base/factory modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.hybrid.causal_attention import CausalAttention
    from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
    from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention

_LAZY_IMPORTS = {
    "CausalAttention": (
        "rtp_llm.models_py.modules.hybrid.causal_attention",
        "CausalAttention",
    ),
    "DenseMLP": ("rtp_llm.models_py.modules.hybrid.dense_mlp", "DenseMLP"),
    "MlaAttention": (
        "rtp_llm.models_py.modules.hybrid.mla_attention",
        "MlaAttention",
    ),
}


def __getattr__(name: str):
    target = _LAZY_IMPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "CausalAttention",
    "DenseMLP",
    "MlaAttention",
]
