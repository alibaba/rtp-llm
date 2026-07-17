from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import torch


class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(
        self,
        layer,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer) -> None:
        raise NotImplementedError


_LINEAR_METHOD_REGISTRY: Dict[str, Type[QuantizeMethodBase]] = {}


def register_quant_method(*keys: str):
    def decorator(cls: Type[QuantizeMethodBase]) -> Type[QuantizeMethodBase]:
        for key in keys:
            existing = _LINEAR_METHOD_REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Quant method {key!r} is already registered to "
                    f"{existing.__name__}"
                )
            _LINEAR_METHOD_REGISTRY[key] = cls
        return cls

    return decorator


class QuantizationConfig:
    """Typed dispatch config for the first unquantized newloader model slice."""

    def __init__(self, quant_type: str = "none"):
        if not isinstance(quant_type, str):
            raise TypeError("quant_type must be a string")
        normalized = quant_type.strip().lower()
        if normalized in ("", "none"):
            normalized = "none"
        self.quant_type = normalized

    def get_quant_method(self, layer, prefix: str = "") -> QuantizeMethodBase:
        from rtp_llm.models_py.quant_methods import unquantized  # noqa: F401

        method_cls = _LINEAR_METHOD_REGISTRY.get(self.quant_type)
        if method_cls is None:
            raise ValueError(
                f"Quantization {self.quant_type!r} is not supported by the "
                "Qwen dense newloader slice"
            )
        return method_cls()
