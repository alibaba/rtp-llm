from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch

if TYPE_CHECKING:
    from rtp_llm.models_py.layers.linear import LinearBase


class QuantizeMethodBase(ABC):

    @abstractmethod
    def create_weights(
        self,
        layer: "LinearBase",
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self, layer: "LinearBase", x: torch.Tensor, bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: "LinearBase"):
        raise NotImplementedError


_RUNTIME_METHOD_REGISTRY: Dict[str, Type[QuantizeMethodBase]] = {}


def register_quant_method(*keys: str):
    def deco(cls: Type[QuantizeMethodBase]) -> Type[QuantizeMethodBase]:
        for k in keys:
            existing = _RUNTIME_METHOD_REGISTRY.get(k)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"quant method key '{k}' already registered to "
                    f"{existing.__name__}, cannot re-register to {cls.__name__}"
                )
            _RUNTIME_METHOD_REGISTRY[k] = cls
        return cls

    return deco


class QuantizationConfig:

    def __init__(self, quant_type: str = "none"):
        self.quant_type = quant_type

    def get_quant_method(self, layer, prefix: str = "") -> QuantizeMethodBase:
        cls = _RUNTIME_METHOD_REGISTRY.get(self.quant_type)
        if cls is None:
            from rtp_llm.models_py.quant_methods.unquantized import (
                UnquantizedLinearMethod,
            )

            return UnquantizedLinearMethod()
        return cls()
