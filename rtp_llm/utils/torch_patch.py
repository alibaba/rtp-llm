import logging
from typing import List

import torch

# 1. 保存原始 concat
original_concat = torch.concat


# 2. 定义自定义 concat
def custom_concat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    try:
        return original_concat(tensors, dim=dim).contiguous()
    except RuntimeError as e:
        if "cat_cuda" in str(e) and "Float8_e4m3fn" in str(e):
            logging.info("Caught Float8_e4m3fn concat error. Falling back to CPU.")
            original_device = tensors[0].device
            tensors_cpu = [t.cpu() for t in tensors]
            result_cpu = original_concat(tensors_cpu, dim=dim).contiguous()
            return result_cpu.to(original_device)
        else:
            raise


# 3. 劫持 torch.concat
torch.concat = custom_concat

# 4. 配置 torch cpp_extension 路径（用于 flashinfer 统一 ninja 编译）
try:
    import os

    import flashinfer  # 检查是否安装了 flashinfer
    import torch.utils.cpp_extension as cpp_ext

    # 使用 realpath 解析软链接到实际目标文件
    _HERE = os.path.realpath(cpp_ext.__file__)
    _TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))
    TORCH_LIB_PATH = os.path.join(_TORCH_PATH, "lib")

    # 覆盖 cpp_extension 的路径配置，确保统一的 ninja 编译路径
    cpp_ext._HERE = _HERE
    cpp_ext._TORCH_PATH = _TORCH_PATH
    cpp_ext.TORCH_LIB_PATH = TORCH_LIB_PATH

    logging.debug(f"Patched torch cpp_extension paths for flashinfer: _HERE={_HERE}")
except ImportError:
    # flashinfer 未安装，跳过路径配置
    pass
