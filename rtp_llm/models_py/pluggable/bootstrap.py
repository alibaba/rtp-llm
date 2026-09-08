"""Freeze public and optional extension descriptions once per worker process."""

import importlib
from threading import RLock

from rtp_llm.models_py.pluggable.registry import ModuleRegistry
from rtp_llm.utils.import_util import import_optional_internal_source_entrypoint

_registry = None
_lock = RLock()


def get_module_registry():
    global _registry
    with _lock:
        if _registry is None:
            from rtp_llm.models_py.pluggable.dsv4_specs import register_modules

            registry = ModuleRegistry()
            register_modules(registry)
            from rtp_llm.platforms import register_modules as register_platforms

            register_platforms(registry)
            entry = "models_py.pluggable_register"
            if import_optional_internal_source_entrypoint(entry):
                importlib.import_module(
                    "internal_source.rtp_llm." + entry
                ).register_modules(registry)
            _registry = registry.freeze()
        return _registry
