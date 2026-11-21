"""Direct parameter layers; import only the requested execution dependency.

For example the CPU RMSNorm checkpoint owner does not require distributed
embedding/native bindings. Requesting an embedding or linear still imports its
original implementation and retains all of its backend checks.
"""

from importlib import import_module

__all__ = [
    "LinearBase",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RMSNorm",
    "VocabParallelEmbedding",
    "ParallelLMHead",
]

_MODULES = {
    "RMSNorm": "norm",
    "VocabParallelEmbedding": "embedding",
    "ParallelLMHead": "embedding",
    **{
        name: "linear"
        for name in __all__
        if name.endswith("Linear") or name == "LinearBase"
    },
}


def __getattr__(name):
    module_name = _MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
