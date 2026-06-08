from typing import Dict, Type

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(model_type: str):
    def decorator(cls):
        if model_type in MODEL_REGISTRY:
            raise ValueError(
                f"Model type '{model_type}' is already registered by "
                f"{MODEL_REGISTRY[model_type].__name__}. Cannot register {cls.__name__}."
            )
        MODEL_REGISTRY[model_type] = cls
        return cls

    return decorator


def get_model_class(model_type: str) -> Type[nn.Module]:
    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Model type '{model_type}' not found in registry. "
            f"Available types: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_type]


def list_models():
    return list(MODEL_REGISTRY.keys())
