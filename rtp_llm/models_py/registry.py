import importlib
import inspect
import logging
from typing import Dict, Tuple, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# 仅声明模型名称到其对应的导入路径和类名
_LAZY_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "bert": ("rtp_llm.models_py.new_models.bert", "BertForEmbedding"),
    "roberta": ("rtp_llm.models_py.new_models.bert", "RobertaForEmbedding"),
}



def _registry_keys():
    return tuple(dict.fromkeys([*_LAZY_MODEL_REGISTRY.keys(), *dict.keys(MODEL_REGISTRY)]))


class LazyRegistryDict(dict):
    """A dictionary that lazily imports and registers models on demand.

    This ensures we remain 100% backward compatible with external code that
    directly accesses or iterates over MODEL_REGISTRY, while preventing
    eager imports of all model dependencies during package initialization.
    """

    def __getitem__(self, key):
        if not dict.__contains__(self, key) and key in _LAZY_MODEL_REGISTRY:
            # Trigger lazy load
            cls = get_model_class(key)
            dict.__setitem__(self, key, cls)
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in _LAZY_MODEL_REGISTRY or super().__contains__(key)

    def keys(self):
        return _registry_keys()

    def __iter__(self):
        return iter(_registry_keys())

    def __len__(self):
        return len(_registry_keys())

    def items(self):
        for k in _registry_keys():
            yield k, self[k]

    def values(self):
        for k in _registry_keys():
            yield self[k]


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = LazyRegistryDict()


def _safe_class_file(cls: Type[nn.Module]) -> str:
    try:
        return inspect.getfile(cls)
    except TypeError:
        return "<unknown>"


def register_model(model_type: str):
    def decorator(cls):
        caller = inspect.stack()[1]
        cls_file = _safe_class_file(cls)
        if model_type in MODEL_REGISTRY and model_type not in _LAZY_MODEL_REGISTRY:
            existing_cls = MODEL_REGISTRY[model_type]
            logger.warning(
                "NewModelRegistry duplicate registration: model_type=%s "
                "new_cls=%s new_module=%s new_file=%s caller=%s:%s "
                "existing_cls=%s existing_module=%s existing_file=%s",
                model_type,
                cls.__qualname__,
                cls.__module__,
                cls_file,
                caller.filename,
                caller.lineno,
                existing_cls.__qualname__,
                existing_cls.__module__,
                _safe_class_file(existing_cls),
            )
            raise ValueError(
                f"Model type '{model_type}' is already registered by "
                f"{MODEL_REGISTRY[model_type].__name__}. Cannot register {cls.__name__}."
            )
        logger.info(
            "NewModelRegistry register_model: model_type=%s cls=%s module=%s "
            "file=%s caller=%s:%s",
            model_type,
            cls.__qualname__,
            cls.__module__,
            cls_file,
            caller.filename,
            caller.lineno,
        )
        # Directly set the item in base dict to cache it
        dict.__setitem__(MODEL_REGISTRY, model_type, cls)
        return cls

    return decorator


def get_model_class(model_type: str) -> Type[nn.Module]:
    # Check physical cache first (registered classes)
    if not dict.__contains__(MODEL_REGISTRY, model_type):
        if model_type in _LAZY_MODEL_REGISTRY:
            module_path, class_name = _LAZY_MODEL_REGISTRY[model_type]
            try:
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                dict.__setitem__(MODEL_REGISTRY, model_type, cls)
            except Exception as e:
                logger.error(
                    "NewModelRegistry failed to lazy load model_type=%s from %s.%s: %s",
                    model_type,
                    module_path,
                    class_name,
                    e,
                )
                raise ImportError(
                    f"Failed to dynamically load model '{model_type}' from {module_path}.{class_name}: {e}"
                ) from e
        else:
            raise KeyError(
                f"Model type '{model_type}' not found in registry. "
                f"Available types: {list(_LAZY_MODEL_REGISTRY.keys())}"
            )
    return dict.__getitem__(MODEL_REGISTRY, model_type)


def list_models():
    return list(_registry_keys())
