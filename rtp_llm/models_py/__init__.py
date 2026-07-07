import logging

from rtp_llm.models_py.model_loader import LoadConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule, rtp_module
from rtp_llm.models_py.registry import MODEL_REGISTRY, get_model_class, register_model, _LAZY_MODEL_REGISTRY

_logger = logging.getLogger(__name__)

# Dynamically gather all unique model class names declared in registry to avoid double-registration.
_DYNAMIC_CLASSES = list(set(class_name for _, class_name in _LAZY_MODEL_REGISTRY.values()))


def __getattr__(name: str):
    # Lookup model class by its name dynamically inside the central registry map.
    for _, (module_path, class_name) in _LAZY_MODEL_REGISTRY.items():
        if class_name == name:
            import importlib
            try:
                mod = importlib.import_module(module_path)
                return getattr(mod, class_name)
            except Exception as e:
                _logger.warning(
                    "Failed to dynamic-import '%s' (from %s.%s): %s",
                    name,
                    module_path,
                    class_name,
                    e,
                )
                raise ImportError(
                    f"Failed to dynamically import {name} from {module_path}: {e}"
                ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "NewModelLoader",
    "LoadConfig",
    "RtpModule",
    "rtp_module",
] + _DYNAMIC_CLASSES
