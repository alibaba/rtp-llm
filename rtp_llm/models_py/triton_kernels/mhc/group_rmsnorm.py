"""
Group RMSNorm 2D Tiled

优化策略：
1. 使用 2D grid 设计，兼容各种 M 大小
2. 智能选择 BLOCK_SIZE_M 和 BLOCK_SIZE_N
3. M 小时增加并行度，M 大时减少 kernel launch 开销
4. 使用 rsqrt、float32 累加等优化技巧
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def group_rms_norm_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    M,
    N,
    num_groups,
    eps,
    BLOCK_SIZE_M: tl.constexpr,  # 序列维度的block size
    BLOCK_SIZE_N: tl.constexpr,  # hidden维度的block size
):
    """
    2D Tiled Group RMSNorm Kernel

    策略：
    - 小 M (< 64): BLOCK_SIZE_M=1，增加并行度（更多blocks）
    - 中 M (64-512): BLOCK_SIZE_M=4，平衡并行度和效率
    - 大 M (>512): BLOCK_SIZE_M=8-16，减少 kernel launch 开销

    Grid: (num_groups, cdiv(M, BLOCK_SIZE_M))
    每个 block 处理一个 group 的 BLOCK_SIZE_M 个序列位置
    """

    # 2D grid 解码
    group_id = tl.program_id(0)
    m_block_id = tl.program_id(1)

    # 该 block 处理的序列范围
    m_start = m_block_id * BLOCK_SIZE_M
    weight_offset = group_id * N

    # 对该 block 负责的每个序列位置分别处理
    for m_idx in range(BLOCK_SIZE_M):
        m = m_start + m_idx

        if m < M:
            row_offset = group_id * M * N + m * N

            # ========== PASS 1: 计算 sum of squares ==========
            sum_sq = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)

            for block_start in range(0, N, BLOCK_SIZE_N):
                col_offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
                mask = col_offsets < N

                input_ptrs = input_ptr + row_offset + col_offsets
                x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
                sum_sq += tl.where(mask, x * x, 0.0)

            # 计算 1/rms
            total_sum_sq = tl.sum(sum_sq)
            rms_inv = tl.rsqrt(total_sum_sq / N + eps)

            # ========== PASS 2: 归一化 && Weight 乘法 ==========
            for block_start in range(0, N, BLOCK_SIZE_N):
                col_offsets = block_start + tl.arange(0, BLOCK_SIZE_N)
                mask = col_offsets < N

                input_ptrs = input_ptr + row_offset + col_offsets
                x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)

                weight_ptrs = weight_ptr + weight_offset + col_offsets
                w = tl.load(weight_ptrs, mask=mask, other=1.0).to(tl.float32)

                output = x * rms_inv * w

                output_ptrs = output_ptr + row_offset + col_offsets
                tl.store(output_ptrs, output, mask=mask)


def group_rms_norm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Group RMSNorm

    自动根据输入 shape 选择最优的 BLOCK_SIZE 配置

    Args:
        output: 输出 tensor, shape=[num_groups, M, N]
        input: 输入 tensor, shape=[num_groups, M, N]
        weight: 权重 tensor, shape=[num_groups, N]
        eps: epsilon for numerical stability

    Returns:
        output tensor
    """
    assert input.dim() == 3, f"Expected 3D input, got {input.dim()}D"
    assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"

    num_groups, m, n = input.shape
    assert weight.shape == (
        num_groups,
        n,
    ), f"Weight shape {weight.shape} doesn't match input [{num_groups}, {n}]"
    assert (
        output.shape == input.shape
    ), f"Output shape {output.shape} doesn't match input {input.shape}"

    # ========== 智能选择 BLOCK_SIZE_M ==========
    # 小 M: 用 BLOCK_SIZE_M=1，增加并行度
    # 中 M: 用 BLOCK_SIZE_M=4，平衡效率和并行度
    # 大 M: 用 BLOCK_SIZE_M=8-16，减少 kernel launch 开销

    if m <= 32:
        BLOCK_SIZE_M = 1
    elif m <= 128:
        BLOCK_SIZE_M = 2
    elif m <= 512:
        BLOCK_SIZE_M = 4
    elif m <= 2048:
        BLOCK_SIZE_M = 8
    else:
        BLOCK_SIZE_M = 16

    # ========== 智能选择 BLOCK_SIZE_N ==========
    # 原则：
    # 1. 尽量减少循环次数（但不能太大导致寄存器溢出）
    # 2. 考虑 warp size (32) 和 cache line 对齐
    # 3. 通常 512-2048 是比较好的范围

    if n <= 256:
        BLOCK_SIZE_N = 256
    elif n <= 512:
        BLOCK_SIZE_N = 512
    elif n <= 1024:
        BLOCK_SIZE_N = 512  # 避免寄存器压力
    elif n <= 2048:
        BLOCK_SIZE_N = 1024
    elif n <= 4096:
        BLOCK_SIZE_N = 1024
    elif n <= 8192:
        BLOCK_SIZE_N = 2048
    else:
        BLOCK_SIZE_N = 2048  # 上限

    grid = (num_groups, triton.cdiv(m, BLOCK_SIZE_M))

    group_rms_norm_kernel[grid](
        output,
        input,
        weight,
        m,
        n,
        num_groups,
        eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return output


class GroupRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, group_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        self.weight = weight

    def forward(
        self, input: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if output is None:
            output = torch.empty_like(input)
        else:
            assert (
                output.shape == input.shape
            ), f"Output shape {output.shape} doesn't match input {input.shape}"
        return group_rms_norm(output, input, self.weight, self.eps)
