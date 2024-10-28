import os
from enum import Enum
from typing import Optional, Union, Dict, Any

class WEIGHT_TYPE(Enum):
    INT4 = ["int4"]
    INT8 = ["int8"]
    FP8 = ["fp8", "fp8_e4m3"]
    FP16 = ["fp16", "float16"]
    FP32 = ["fp32", "float32"]
    BF16 = ["bf16", "bfloat16","bp16"]
    @classmethod
    def from_str(cls, value: str) -> 'WEIGHT_TYPE':
        lower_value = value.lower()
        for name, member in cls.__members__.items():
            if lower_value in map(str.lower, member.value):
                return member
        raise ValueError('No enum member with value %s' % value)

    def to_str(self) -> str:
        return self.value[0]

def get_weight_type_from_env(env_param: Dict[str, str]) -> WEIGHT_TYPE:
    weight_type_str = env_param.get("WEIGHT_TYPE", None)
    if weight_type_str:
        weight_type = WEIGHT_TYPE.from_str(weight_type_str)
        return weight_type
    else:
        int8_mode = int(env_param.get("INT8_MODE", "0"))
        if int8_mode == 1:
            return WEIGHT_TYPE.INT8
        return WEIGHT_TYPE.FP16

def get_propose_weight_type_from_env(env_param: Dict[str, str]) -> WEIGHT_TYPE:
    propose_weight_type_str = env_param.get("SP_WEIGHT_TYPE", None)
    if propose_weight_type_str:
        propose_weight_type = WEIGHT_TYPE.from_str(propose_weight_type_str)
    else:
        propose_int8_mode = int(env_param.get('SP_INT8_MODE', '0'))
        propose_weight_type = WEIGHT_TYPE.INT8 if propose_int8_mode == 1 else WEIGHT_TYPE.FP16
    return propose_weight_type
