"""
Test for fused scaled dot product gate kernel
"""

import math
import unittest

import torch
import torch.nn as nn

from rtp_llm.models_py.triton_kernels.mhc.scaled_dot_product_gate import (
    scaled_dot_product_gate,
)


class ScaledDotProductGateTorch(nn.Module):
    """PyTorch 参考实现"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, key_norm: torch.Tensor, query_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            key_norm: [hc_mult, num_tokens, hidden_size]
            query_norm: [hc_mult, num_tokens, hidden_size]
        Returns:
            gate: [hc_mult, num_tokens]
        """
        # scaled dot product
        gate = (key_norm * query_norm).sum(dim=-1) / math.sqrt(self.hidden_size)
        # abs().clamp_min(eps).sqrt() * sign()
        gate = gate.abs().clamp_min(self.eps).sqrt() * gate.sign()
        # sigmoid
        gate = gate.sigmoid()
        return gate


class TestScaledDotProductGate(unittest.TestCase):

    def test_scaled_dot_product_gate_function(self):
        """测试 scaled_dot_product_gate 函数"""
        device = torch.device("cuda")
        eps = 1e-6

        # 测试不同的配置
        test_configs = [
            (2, 128, 1024, torch.float16),  # (hc_mult, num_tokens, hidden_size, dtype)
            (4, 256, 2048, torch.float16),
            (8, 512, 4096, torch.bfloat16),
        ]

        for hc_mult, num_tokens, hidden_size, dtype in test_configs:
            with self.subTest(
                hc_mult=hc_mult,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
            ):
                # 创建随机输入
                key_norm = torch.randn(
                    hc_mult, num_tokens, hidden_size, dtype=dtype, device=device
                )
                query_norm = torch.randn(
                    hc_mult, num_tokens, hidden_size, dtype=dtype, device=device
                )

                # Triton 实现
                gate_triton = scaled_dot_product_gate(
                    key_norm, query_norm, hidden_size, eps
                )

                # PyTorch 参考实现
                ref_impl = ScaledDotProductGateTorch(hidden_size, eps)
                gate_ref = ref_impl(key_norm, query_norm)

                # 比较结果
                torch.testing.assert_close(
                    gate_triton,
                    gate_ref,
                    atol=1e-2 if dtype == torch.float16 else 5e-2,
                    rtol=1e-2 if dtype == torch.float16 else 5e-2,
                )

    def test_scaled_dot_product_gate_module(self):
        """测试 ScaledDotProductGate Module"""
        device = torch.device("cuda")
        eps = 1e-6

        # 测试不同的配置
        test_configs = [
            (2, 128, 1024, torch.float16),
            (4, 256, 2048, torch.bfloat16),
        ]

        for hc_mult, num_tokens, hidden_size, dtype in test_configs:
            with self.subTest(
                hc_mult=hc_mult,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                dtype=dtype,
            ):
                key_norm = torch.randn(
                    hc_mult, num_tokens, hidden_size, dtype=dtype, device=device
                )
                query_norm = torch.randn(
                    hc_mult, num_tokens, hidden_size, dtype=dtype, device=device
                )

                gate_triton = scaled_dot_product_gate(
                    key_norm, query_norm, hidden_size, eps
                )

                ref_module = ScaledDotProductGateTorch(hidden_size, eps).to(device)
                gate_ref = ref_module(key_norm, query_norm)

                torch.testing.assert_close(
                    gate_triton,
                    gate_ref,
                    atol=1e-2 if dtype == torch.float16 else 5e-2,
                    rtol=1e-2 if dtype == torch.float16 else 5e-2,
                )


if __name__ == "__main__":
    unittest.main()
