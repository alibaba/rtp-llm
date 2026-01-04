#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 l2norm 两个 kernel 的差异"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.fla.l2norm import L2Norm, l2norm


def l2norm_reference(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch 参考实现"""
    norm = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)
    return x / norm


def compute_max_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """计算最大相对误差"""
    abs_diff = torch.abs(actual - expected)
    abs_expected = torch.abs(expected)
    rel_error = abs_diff / (abs_expected + 1e-10)
    return rel_error.max().item()


def compute_mean_relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """计算平均相对误差"""
    abs_diff = torch.abs(actual - expected)
    abs_expected = torch.abs(expected)
    rel_error = abs_diff / (abs_expected + 1e-10)
    return rel_error.mean().item()


class TestL2Norm(unittest.TestCase):
    """L2Norm 测试类"""

    def setUp(self):
        """每个测试前的设置"""
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.eps = 1e-6

    def test_l2norm_accuracy(self):
        """测试不同维度下的 l2norm 精度"""
        test_cases = [
            (1, 64),  # 小 batch, 小维度
            (2, 128),  # 小 batch, 中等维度
            (8, 256),  # 中等 batch, 中等维度
            (16, 512),  # 中等 batch, 大维度
            (32, 1024),  # 大 batch, 大维度
            (64, 2048),  # 大 batch, 超大维度
            (128, 4096),  # 超大 batch, 超大维度
            (1024, 128),  # 超大 batch, 中等维度
            (4096, 512),  # 超大 batch, 大维度
        ]

        for batch_size, hidden_dim in test_cases:
            with self.subTest(batch_size=batch_size, hidden_dim=hidden_dim):
                torch.manual_seed(42)

                # 生成随机输入数据
                x = torch.randn(
                    batch_size, hidden_dim, device=self.device, dtype=torch.float32
                )

                # 计算参考结果
                y_ref = l2norm_reference(x, eps=self.eps)

                # 计算 Triton kernel 结果
                y_triton = l2norm(x, eps=self.eps)

                # 计算误差
                max_rel_error = compute_max_relative_error(y_triton, y_ref)
                mean_rel_error = compute_mean_relative_error(y_triton, y_ref)
                max_abs_error = torch.abs(y_triton - y_ref).max().item()

                print(f"\n测试用例: batch_size={batch_size}, hidden_dim={hidden_dim}")
                print(f"  最大相对误差: {max_rel_error:.6e}")
                print(f"  平均相对误差: {mean_rel_error:.6e}")
                print(f"  最大绝对误差: {max_abs_error:.6e}")

                # 精度阈值
                self.assertLess(
                    max_rel_error, 1e-5, f"最大相对误差 {max_rel_error} 超过阈值 1e-5"
                )
                self.assertLess(
                    mean_rel_error, 1e-6, f"平均相对误差 {mean_rel_error} 超过阈值 1e-6"
                )
                self.assertTrue(
                    torch.allclose(y_triton, y_ref, rtol=1e-5, atol=1e-6), "结果不匹配"
                )

    def test_l2norm_dtype(self):
        """测试不同数据类型"""
        dtypes = [
            (torch.float32, 1e-5, 1e-6),
            (torch.float16, 1e-3, 1e-3),
            (torch.bfloat16, 1e-2, 1e-2),
        ]

        batch_size, hidden_dim = 32, 512

        for dtype, rtol, atol in dtypes:
            with self.subTest(dtype=dtype):
                torch.manual_seed(42)

                # 生成随机输入数据
                x = torch.randn(batch_size, hidden_dim, device=self.device, dtype=dtype)

                # 计算参考结果（使用 float32）
                y_ref = l2norm_reference(x.float(), eps=self.eps)

                # 计算 Triton kernel 结果
                y_triton = l2norm(x, eps=self.eps)

                max_rel_error = compute_max_relative_error(y_triton.float(), y_ref)
                mean_rel_error = compute_mean_relative_error(y_triton.float(), y_ref)

                print(f"\n测试数据类型: {dtype}")
                print(f"  最大相对误差: {max_rel_error:.6e}")
                print(f"  平均相对误差: {mean_rel_error:.6e}")

                self.assertTrue(
                    torch.allclose(y_triton.float(), y_ref, rtol=rtol, atol=atol),
                    f"{dtype} 结果不匹配",
                )

    def test_l2norm_module(self):
        """测试 L2Norm 模块"""
        batch_size, hidden_dim = 16, 256

        # 创建模块
        l2norm_layer = L2Norm(eps=self.eps).to(self.device)

        # 生成随机输入数据
        x = torch.randn(batch_size, hidden_dim, device=self.device, dtype=torch.float32)

        # 计算参考结果
        y_ref = l2norm_reference(x, eps=self.eps)

        # 计算模块结果
        y_module = l2norm_layer(x)

        # 计算误差
        max_rel_error = compute_max_relative_error(y_module, y_ref)
        mean_rel_error = compute_mean_relative_error(y_module, y_ref)

        print(f"\n测试 L2Norm 模块")
        print(f"  最大相对误差: {max_rel_error:.6e}")
        print(f"  平均相对误差: {mean_rel_error:.6e}")

        self.assertTrue(
            torch.allclose(y_module, y_ref, rtol=1e-5, atol=1e-6),
            "L2Norm 模块结果不匹配",
        )

    def test_l2norm_edge_cases(self):
        """测试边界情况"""
        # 测试全零输入
        x_zeros = torch.zeros(4, 128, device=self.device, dtype=torch.float32)
        y_zeros = l2norm(x_zeros, eps=self.eps)
        self.assertTrue(torch.all(torch.isfinite(y_zeros)), "全零输入产生了无效值")

        # 测试非常小的值
        x_small = torch.ones(4, 128, device=self.device, dtype=torch.float32) * 1e-8
        y_small = l2norm(x_small, eps=self.eps)
        self.assertTrue(torch.all(torch.isfinite(y_small)), "极小值输入产生了无效值")

        # 测试单个样本
        x_single = torch.randn(1, 128, device=self.device, dtype=torch.float32)
        y_single = l2norm(x_single, eps=self.eps)
        y_ref_single = l2norm_reference(x_single, eps=self.eps)
        self.assertTrue(
            torch.allclose(y_single, y_ref_single, rtol=1e-5, atol=1e-6),
            "单样本结果不匹配",
        )

        print("\n边界情况测试通过")

    def test_l2norm_multidim(self):
        """测试多维输入"""
        # 测试 3D 输入 (batch, seq_len, hidden_dim)
        x_3d = torch.randn(4, 16, 128, device=self.device, dtype=torch.float32)
        y_3d = l2norm(x_3d, eps=self.eps)
        y_ref_3d = l2norm_reference(x_3d, eps=self.eps)

        self.assertEqual(x_3d.shape, y_3d.shape, "输出形状应与输入相同")
        self.assertTrue(
            torch.allclose(y_3d, y_ref_3d, rtol=1e-5, atol=1e-6), "3D 输入结果不匹配"
        )

        # 测试 4D 输入 (batch, heads, seq_len, hidden_dim)
        x_4d = torch.randn(2, 8, 16, 64, device=self.device, dtype=torch.float32)
        y_4d = l2norm(x_4d, eps=self.eps)
        y_ref_4d = l2norm_reference(x_4d, eps=self.eps)

        self.assertEqual(x_4d.shape, y_4d.shape, "输出形状应与输入相同")
        self.assertTrue(
            torch.allclose(y_4d, y_ref_4d, rtol=1e-5, atol=1e-6), "4D 输入结果不匹配"
        )

        print("\n多维输入测试通过")


if __name__ == "__main__":
    unittest.main(verbosity=2)
