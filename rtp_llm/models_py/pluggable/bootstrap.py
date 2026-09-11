"""Freeze public and optional extension descriptions once per worker process."""

import importlib
from threading import RLock

from rtp_llm.models_py.pluggable.registry import ModuleRegistry
from rtp_llm.utils.import_util import import_optional_internal_source_entrypoint

_registries = {}
_lock = RLock()


def get_module_registry(register_model_modules):
    with _lock:
        if register_model_modules not in _registries:
            registry = ModuleRegistry()
            register_model_modules(registry)
            from rtp_llm.platforms import register_modules as register_platforms

            register_platforms(registry)
            entry = "models_py.pluggable_register"
            if import_optional_internal_source_entrypoint(entry):
                importlib.import_module(
                    "internal_source.rtp_llm." + entry
                ).register_modules(registry)
            _registries[register_model_modules] = registry.freeze()
        return _registries[register_model_modules]
