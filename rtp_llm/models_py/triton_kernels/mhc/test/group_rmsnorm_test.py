import sys
import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.triton_kernels.mhc.group_rmsnorm import (
    GroupRMSNorm,
    group_rms_norm,
)


class GroupRMSNormTorch(nn.Module):
    """PyTorch 参考实现 - 对每个 group 分别计算 RMSNorm"""

    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = weight  # [num_groups, N]
        self.variance_epsilon = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [num_groups, M, N]
            weight: [num_groups, N]
        Returns:
            output: [num_groups, M, N]
        """
        num_groups = input.shape[0]
        outputs = []

        # 对每个 group 分别计算 RMSNorm
        for g in range(num_groups):
            hidden_states = input[g]  # [M, N]
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)

            # 计算 variance (对最后一维)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)

            # 归一化
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )

            # 乘以该 group 的 weight
            output = self.weight[g] * hidden_states.to(input_dtype)
            outputs.append(output)

        return torch.stack(outputs, dim=0)


class TestGroupRMSNorm(unittest.TestCase):

    def test_group_rms_norm_module(self):
        """测试 GroupRMSNorm Module"""
        device = torch.device("cuda")
        eps = 1e-6

        # 测试不同的 num_groups 和维度
        test_configs = [
            (2, 4, 128, 1024),  # (num_groups, group_size, M, N/group_size)
            (4, 2, 256, 2048),
            (8, 1, 512, 4096),
        ]

        for num_groups, group_size, m, n_per_group in test_configs:
            with self.subTest(
                num_groups=num_groups, group_size=group_size, m=m, n=n_per_group
            ):
                # 创建 weight
                weight = torch.randn(
                    num_groups, n_per_group, dtype=torch.bfloat16, device=device
                )
                # 创建输入 (实际使用时可能是 [M, total_N] 然后 reshape)
                input_tensor = torch.randn(
                    num_groups, m, n_per_group, dtype=torch.bfloat16, device=device
                )
                # Triton Module
                group_rms_norm_module = GroupRMSNorm(weight, group_size, eps)
                output = group_rms_norm_module(input_tensor)

                # PyTorch 参考实现
                ref_impl = GroupRMSNormTorch(weight, eps)
                ref_output = ref_impl(input_tensor)

                torch.testing.assert_close(
                    output,
                    ref_output,
                    atol=1e-2,
                    rtol=1e-2,
                )


if __name__ == "__main__":
    unittest.main()
