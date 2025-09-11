from enum import Enum

import torch


class WEIGHT_TYPE(Enum):
    # 每个枚举值现在是 (别名列表, torch dtype) 的元组
    AUTO = [["auto"], None]  # 特殊处理
    INT4 = (["int4"], None)  # 特殊处理
    INT8 = (["int8"], torch.int8)
    FP8 = (["fp8", "fp8_e4m3"], torch.float8_e4m3fn)
    FP16 = (["fp16", "float16"], torch.float16)
    FP32 = (["fp32", "float32"], torch.float32)
    BF16 = (["bf16", "bfloat16", "bp16"], torch.bfloat16)

    @classmethod
    def from_str(cls, value: str) -> "WEIGHT_TYPE":
        lower_value = value.lower()
        for member in cls:
            # 调整这里访问成员值的第一个元素（别名列表）
            if lower_value in map(str.lower, member.value[0]):
                return member
        raise ValueError("No enum member with value %s" % value)

    def to_str(self) -> str:
        return self.value[0][0]

    def to_torch_dtype(self) -> torch.dtype:
        if (dtype := self.value[1]) is None:
            raise NotImplementedError("Torch does not support INT4 dtype directly.")
        return dtype
