
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict,List

import weakref
import torch

class QuantizationType(str, Enum):
    """
    Enum storing quantization type options
    """

    INT = "int"
    FLOAT = "float"

class QuantizationConfig(ABC):
    """Base class for quantization configs."""
    _registry = weakref.WeakValueDictionary()
    def __init__(self, bits: int, group_size: int, is_quanted:bool, **kwargs: Any):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: Dict[str, List[str]] = dict()
        self._bits = bits
        self._group_size = group_size
        self._is_quanted = is_quanted


    @classmethod
    @abstractmethod
    def get_method(cls) -> str:
        """Name of the quantization method."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_algo(cls) -> str:
        """Name of quant_algo in c++"""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError


    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        for _, c in cls._registry.items():
            if c.get_method().upper() == config.get("method", config.get("quant_algo")).upper():
                return c._from_config(config)
        raise ValueError(f"config: {config}'s method is not support in {cls._registry.keys()}")

    def bits(self)-> int:
        return self._bits

    def is_quanted(self) -> bool:
        return self._is_quanted

    def group_size(self)-> int:
        return self._group_size
    

class WeightOnlyInt8PerChannelQuantConfig(QuantizationConfig):
    def __init__(self):
        super().__init__(bits = 8, group_size=0, is_quanted=False)
        pass

    @classmethod
    def get_method(cls) -> str:
        return "INT8"

    @classmethod
    def get_algo(cls) -> str:
        return "weight_only_per_col"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return WeightOnlyInt8PerChannelQuantConfig()

DEFAULT_WEIGHT_ONLY_INT8_PER_CHANNEL_QUANT_CONFIG = WeightOnlyInt8PerChannelQuantConfig()

class Fp8PerTensorQuantConfig(QuantizationConfig):
    def __init__(self, bits: int=8, group_size: int=0, is_quanted: bool=False, **kwargs: Any):
        assert bits == 8 and group_size == 0, f"invalid params {bits} != 8 or {group_size} != 0"
        super().__init__(bits=8, group_size=0, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "FP8"

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return Fp8PerTensorQuantConfig(**config)

DEFAULT_FP8_PER_TENSOR_QUANT_CONFIG = Fp8PerTensorQuantConfig(is_quanted=False)

class Fp8BlockWiseQuantConfig(QuantizationConfig):
    DEFAULT_FP8_QUANT_BLOCK_SIZE=128
    def __init__(self, bits: int=8, group_size: int=128, is_quanted: bool=False, **kwargs: Any):
        super().__init__(bits=bits, group_size=group_size, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "FP8_PER_BLOCK"

    @classmethod
    def get_algo(cls) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return Fp8BlockWiseQuantConfig(**config)

class Fp8PerChannelQuantConfig(QuantizationConfig):
    def __init__(self, bits: int=8, group_size: int=0, is_quanted: bool=False, **kwargs: Any):
        super().__init__(bits=bits, group_size=0, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "FP8_PER_CHANNEL"

    @classmethod
    def get_algo(cls) -> str:
        return "compressed-tensors"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return Fp8PerChannelQuantConfig(**config)

class SmoothQuantConfig(QuantizationConfig):
    def __init__(self):
        super().__init__(bits=0, group_size=0, is_quanted=True)

    @classmethod
    def get_method(cls) -> str:
        return "smooth_quant"

    @classmethod
    def get_algo(cls) -> str:
        return "smooth_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return SmoothQuantConfig()

class OmniQuantConfig(QuantizationConfig):
    def __init__(self):
        super().__init__(bits=0, group_size=0, is_quanted=True)

    @classmethod
    def get_method(cls) -> str:
        return "omni_quant"

    @classmethod
    def get_algo(cls) -> str:
        return "omni_quant"

    def type(self) -> str:
        return "int"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return OmniQuantConfig()

class Int8PerTensorQuantConfig(QuantizationConfig):
    def __init__(self):
        super().__init__(bits=0, group_size=0, is_quanted=True)

    @classmethod
    def get_method(cls) -> str:
        return "pertensor_quant"

    @classmethod
    def get_algo(cls) -> str:
        return "pertensor_quant"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return Int8PerTensorQuantConfig()

class AWQConfig(QuantizationConfig):
    def __init__(self, bits: int, group_size: int, is_quanted: bool, **kwargs: Any):
        super().__init__(bits=bits, group_size=group_size, is_quanted=is_quanted)


    @classmethod
    def get_method(cls) -> str:
        return "awq"

    @classmethod
    def get_algo(cls) -> str:
        return "awq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return AWQConfig(**config)

class GPTQConfig(QuantizationConfig):
    def __init__(self, bits: int, group_size: int, is_quanted: bool, **kwargs: Any):
        super().__init__(bits=bits, group_size=group_size, is_quanted=is_quanted)

    @classmethod
    def get_method(cls) -> str:
        return "gptq"

    @classmethod
    def get_algo(cls) -> str:
        return "gptq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def _from_config(cls, config: Dict[str, Any]) -> "QuantizationConfig":
        return GPTQConfig(**config)


DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG = Fp8BlockWiseQuantConfig(bits=8, group_size=Fp8BlockWiseQuantConfig.DEFAULT_FP8_QUANT_BLOCK_SIZE, is_quanted=False)
DEFAULT_FP8_PER_CHANNEL_QUANT_CONFIG = Fp8PerChannelQuantConfig(bits=8, is_quanted=False)

preset_quant_config = {
    "INT8": DEFAULT_WEIGHT_ONLY_INT8_PER_CHANNEL_QUANT_CONFIG,
    "FP8": DEFAULT_FP8_PER_TENSOR_QUANT_CONFIG,
    "FP8_PER_BLOCK": DEFAULT_FP8_BLOCK_WISE_QUANT_CONFIG,
    "FP8_PER_CHANNEL": DEFAULT_FP8_PER_CHANNEL_QUANT_CONFIG
}