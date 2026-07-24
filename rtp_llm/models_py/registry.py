import importlib
import inspect
import logging
from typing import Dict, Tuple, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}
_LAZY_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {}


def _validate_model_type(model_type: str) -> str:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("model_type must be a non-empty string")
    return model_type.strip()


def _class_location(cls: Type[nn.Module]) -> str:
    try:
        return inspect.getfile(cls)
    except (TypeError, OSError):
        return "<unknown>"


def register_lazy_model(model_type: str, module_path: str, class_name: str) -> None:
    model_type = _validate_model_type(model_type)
    if not module_path or not class_name:
        raise ValueError("module_path and class_name must be non-empty")
    if model_type in MODEL_REGISTRY:
        raise ValueError(
            f"Model type {model_type!r} is already registered by "
            f"{MODEL_REGISTRY[model_type].__name__}"
        )
    existing = _LAZY_MODEL_REGISTRY.get(model_type)
    candidate = (module_path, class_name)
    if existing is not None and existing != candidate:
        raise ValueError(
            f"Model type {model_type!r} already maps to {existing[0]}.{existing[1]}"
        )
    _LAZY_MODEL_REGISTRY[model_type] = candidate


def register_model(model_type: str):
    model_type = _validate_model_type(model_type)

    def decorator(cls: Type[nn.Module]):
        if not isinstance(cls, type) or not issubclass(cls, nn.Module):
            raise TypeError(f"Registered model must inherit nn.Module, got {cls!r}")
        existing = MODEL_REGISTRY.get(model_type)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Model type {model_type!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        lazy = _LAZY_MODEL_REGISTRY.get(model_type)
        if lazy is not None and lazy != (cls.__module__, cls.__name__):
            raise ValueError(
                f"Model type {model_type!r} already maps to {lazy[0]}.{lazy[1]}"
            )
        MODEL_REGISTRY[model_type] = cls
        logger.info(
            "Registered newloader model %s -> %s.%s (%s)",
            model_type,
            cls.__module__,
            cls.__qualname__,
            _class_location(cls),
        )
        return cls

    return decorator


def get_model_class(model_type: str) -> Type[nn.Module]:
    model_type = _validate_model_type(model_type)
    cached = MODEL_REGISTRY.get(model_type)
    if cached is not None:
        return cached

    target = _LAZY_MODEL_REGISTRY.get(model_type)
    if target is None:
        raise KeyError(
            f"Model type {model_type!r} is not registered. Available: {list_models()}"
        )
    module_path, class_name = target
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Failed to load model {model_type!r} from {module_path}.{class_name}: {exc}"
        ) from exc
    if not isinstance(cls, type) or not issubclass(cls, nn.Module):
        raise TypeError(
            f"Lazy model {module_path}.{class_name} must be an nn.Module subclass"
        )
    MODEL_REGISTRY[model_type] = cls
    return cls


def list_models():
    return sorted(set(MODEL_REGISTRY) | set(_LAZY_MODEL_REGISTRY))


register_lazy_model(
    "qwen_3",
    "rtp_llm.models_py.new_models.qwen3",
    "Qwen3ForCausalLM",
)
register_lazy_model(
    "qwen_3_tool",
    "rtp_llm.models_py.new_models.qwen3",
    "Qwen3ForCausalLM",
)
register_lazy_model(
    "qwen_3_moe",
    "rtp_llm.models_py.new_models.qwen3_moe",
    "Qwen3MoeForCausalLM",
)
register_lazy_model(
    "qwen3_next",
    "rtp_llm.models_py.new_models.qwen3_next",
    "Qwen3NextForCausalLM",
)
register_lazy_model(
    "qwen35_moe",
    "rtp_llm.models_py.new_models.qwen3_next",
    "Qwen3NextForCausalLM",
)
register_lazy_model(
    "qwen3_next_mtp",
    "rtp_llm.models_py.new_models.qwen3_next",
    "Qwen3NextMTPForCausalLM",
)
register_lazy_model(
    "qwen35_moe_mtp",
    "rtp_llm.models_py.new_models.qwen3_next",
    "Qwen35MoeMTPForCausalLM",
)
register_lazy_model(
    "qwen_2",
    "rtp_llm.models_py.new_models.qwen2",
    "Qwen2ForCausalLM",
)
register_lazy_model(
    "qwen_agent",
    "rtp_llm.models_py.new_models.qwen2",
    "Qwen2ForCausalLM",
)
register_lazy_model(
    "qwen_tool",
    "rtp_llm.models_py.new_models.qwen2",
    "Qwen2ForCausalLM",
)
register_lazy_model(
    "qwen_2_moe",
    "rtp_llm.models_py.new_models.qwen2_moe",
    "Qwen2MoeForCausalLM",
)
register_lazy_model(
    "qwen2_vl",
    "rtp_llm.models_py.new_models.qwen2_vl.model",
    "Qwen2VLForCausalLM",
)
register_lazy_model(
    "qwen2_vl_vision",
    "rtp_llm.models_py.new_models.qwen2_vl.vision",
    "Qwen2VLForVisionEmbedding",
)
register_lazy_model(
    "qwen3_vl",
    "rtp_llm.models_py.new_models.qwen3_vl",
    "Qwen3VLForCausalLM",
)
register_lazy_model(
    "qwen3_vl_vision",
    "rtp_llm.models_py.new_models.qwen3_vl.vision",
    "Qwen3VLForVisionEmbedding",
)
