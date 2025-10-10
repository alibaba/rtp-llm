import logging
from enum import IntEnum

import torch


class MlaOpsType(IntEnum):
    AUTO = 0
    MHA = 1
    FLASH_INFER = 2
    FLASH_MLA = 3


def check_with_info(condition: bool, error_msg: str):
    if not condition:
        raise Exception(error_msg)


def compare_tensor_diff_with_ratio(
    a: torch.Tensor,
    b: torch.Tensor,
    rel_threshold: float = 1e-1,
    abs_threshold: float = 1e-2,
    ratio: float = 0.03,
    name: str = "",
):
    # 检查不超过5%的元素差异（同时考虑绝对误差和相对误差）
    abs_diff = torch.abs(a - b)

    # 计算相对误差，避免除以零
    denominator = torch.max(torch.abs(a), torch.abs(b))
    denominator = torch.where(
        denominator > 1e-10, denominator, torch.ones_like(denominator)
    )
    rel_diff = abs_diff / denominator

    # 分别计算绝对误差和相对误差超过阈值的元素比例
    abs_diff_mask = abs_diff > abs_threshold
    rel_diff_mask = rel_diff > rel_threshold
    combined_diff_mask = abs_diff_mask & rel_diff_mask

    abs_diff_percentage = abs_diff_mask.sum().item() / abs_diff_mask.numel()
    rel_diff_percentage = rel_diff_mask.sum().item() / rel_diff_mask.numel()
    combined_diff_percentage = (
        combined_diff_mask.sum().item() / combined_diff_mask.numel()
    )

    # 打印误差统计信息
    logging.debug(
        "%s - Absolute error > %s percentage: %.2f%%", name, abs_threshold, abs_diff_percentage * 100
    )
    logging.debug(
        "%s - Relative error > %s percentage: %.2f%%", name, rel_threshold, rel_diff_percentage * 100
    )
    logging.debug("%s - Combined error percentage: %.2f%%", name, combined_diff_percentage * 100)
    logging.debug("%s - Max absolute error: %.6f", name, abs_diff.max().item())
    logging.debug("%s - Max relative error: %.6f", name, rel_diff.max().item())

    # 使用组合条件进行断言
    check_with_info(
        combined_diff_percentage <= ratio,
        f"More than {ratio} of elements exceed both absolute error ({abs_threshold}) and relative error ({rel_threshold}) thresholds: {combined_diff_percentage:.2%}",
    )
