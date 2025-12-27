import logging

import torch


def print_tensor(tensor: torch.Tensor, meta: str):
    """打印tensor的详细信息，包括统计值

    Args:
        tensor: 要打印的tensor
        meta: 元数据信息
    """
    # 转换为fp32精度进行计算
    tensor_fp32 = tensor.float()

    # 计算统计值
    tensor_sum = tensor_fp32.sum().item()
    square_sum = (tensor_fp32**2).sum().item()
    mean = tensor_fp32.mean().item()

    # 打印tensor信息（单个logging.info，分3行）
    logging.info(
        f"[{meta}] dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}\n"
        f"  Values: {tensor}\n"
        f"  Stats(fp32): sum={tensor_sum}, square_sum={square_sum}, mean={mean}"
    )
