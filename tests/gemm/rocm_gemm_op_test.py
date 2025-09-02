import os
import unittest

import torch

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"

import os
import unittest
from typing import Tuple

import torch
from torch.testing import assert_close


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (240.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fnuz).view(
        m, n
    ), (x_amax / 240.0).view(m, -1).to(torch.float32)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (240.0 / x_amax)).to(torch.float8_e4m3fnuz)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 240.0).view(
        x_view.size(0), x_view.size(2)
    ).to(torch.float32)


def ceil_div(a, b):
    return (a + b - 1) // b


def dequantize_per_token(fp8_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, n = fp8_tensor.shape
    assert n % 128 == 0 and scale.shape == (m, n // 128)

    fp16_tensor = fp8_tensor.to(torch.bfloat16).view(m, -1, 128)
    scale_expanded = scale.unsqueeze(-1).expand(-1, -1, 128)
    return (fp16_tensor * scale_expanded).view(m, n)


def dequantize_per_block(fp8_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, n = fp8_tensor.shape
    padded_shape = (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128)

    fp8_padded = torch.zeros(
        padded_shape, dtype=torch.float8_e4m3fnuz, device=fp8_tensor.device
    )
    fp8_padded[:m, :n] = fp8_tensor

    fp16_padded = fp8_padded.to(torch.bfloat16)
    num_blocks_m = padded_shape[0] // 128
    num_blocks_n = padded_shape[1] // 128
    fp16_view = fp16_padded.view(num_blocks_m, 128, num_blocks_n, 128)

    scale_reshaped = scale.view(num_blocks_m, num_blocks_n)
    scale_expanded = scale_reshaped.unsqueeze(1).unsqueeze(3).expand(-1, 128, -1, 128)

    dequantized_padded = (fp16_view * scale_expanded).view(padded_shape)
    return dequantized_padded[:m, :n].contiguous()


def detailed_assert_close(a, b, rtol, atol, msg=""):
    mismatch_mask = ~torch.isclose(a, b, rtol=rtol, atol=atol)
    if mismatch_mask.any():
        mismatch_indices = mismatch_mask.nonzero(as_tuple=False)
        error_msg = f"{msg}\n不匹配位置及数值：\n"
        for idx in mismatch_indices[:10]:  # 仅显示前10个不匹配点，避免过多输出
            idx_tuple = tuple(idx.tolist())
            error_msg += (
                f"索引 {idx_tuple}: a={a[idx_tuple]}, b={b[idx_tuple]}, "
                f"绝对误差={torch.abs(a[idx_tuple] - b[idx_tuple])}, "
                f"相对误差={torch.abs((a[idx_tuple] - b[idx_tuple]) / b[idx_tuple])}\n"
            )
        if len(mismatch_indices) > 10:
            error_msg += f"...（共 {len(mismatch_indices)} 处不匹配）"
        raise AssertionError(error_msg)


class TestGemmOp(unittest.TestCase):
    def setUp(self):
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/librocm_test_ops.so"
        )
        self.gemm_op = torch.classes.unittest.GemmOp()

    def _fp8_gemm_ref(self, A, B):
        """PyTorch参考实现：分块量化矩阵乘法"""
        # 量化输入矩阵A和权重矩阵B
        A_quant, A_scales = per_token_cast_to_fp8(A)
        B_quant, B_scales = per_block_cast_to_fp8(B)
        dequant_A = dequantize_per_token(A_quant, A_scales)
        dequant_B = dequantize_per_block(B_quant, B_scales)

        return torch.matmul(dequant_A, dequant_B).to(torch.bfloat16).to("cpu")

    def test_block_gemm(self):
        shapes = [
            (128, 256, 128),  # 小于块大小
            (512, 512, 256),  # 大于块大小但不整除
            (2048, 1024, 896),  # 完美对齐
        ]
        for m, k, n in shapes:
            # 创建测试数据
            input_fp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
            weight_fp = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

            # 计算参考输出
            ref_output_fp8 = self._fp8_gemm_ref(input_fp, weight_fp).to("cpu")
            ref_output = torch.matmul(input_fp, weight_fp).to(torch.bfloat16).to("cpu")

            # 运行自定义kernel（需要根据实际接口调整）

            b_quant, b_scales = per_block_cast_to_fp8(weight_fp)
            b_quant = b_quant.T.contiguous().reshape([k, n])
            b_scales = b_scales.T.contiguous().reshape([k // 128, n // 128])

            custom_output = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
            self.gemm_op.forward(input_fp, b_quant, b_scales, custom_output)
            custom_output = custom_output.to("cpu")
            torch.set_printoptions(threshold=float("inf"))
            assert_close(
                custom_output,
                ref_output_fp8,
                rtol=0.01,
                atol=0.04,
                msg=f"m:{m}, k:{k}, n:{n} 结果不匹配: deep_gemm: {custom_output},\n torch_gemm_fp8:{ref_output_fp8},\n torch_gemm_fp16:{ref_output},",
            )

    def test_uneven_blocks(self):
        """测试非对齐分块"""
        shapes = [(511, 1024, 256), (511, 1024, 896), (127, 256, 128), (385, 512, 256)]
        for m, k, n in shapes:
            # 创建测试数据
            input_fp = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
            weight_fp = torch.randn(k, n, device="cuda", dtype=torch.bfloat16) * 0.1

            # 计算参考输出
            ref_output_fp8 = self._fp8_gemm_ref(input_fp, weight_fp).to("cpu")
            ref_output = torch.matmul(input_fp, weight_fp).to(torch.bfloat16).to("cpu")

            # 运行自定义kernel（需要根据实际接口调整）

            b_quant, b_scales = per_block_cast_to_fp8(weight_fp)
            b_quant = b_quant.T.contiguous().reshape([k, n])
            b_scales = b_scales.T.contiguous().reshape([k // 128, n // 128])

            custom_output = torch.zeros(m, n, device="cuda", dtype=torch.bfloat16)
            self.gemm_op.forward(input_fp, b_quant, b_scales, custom_output)
            custom_output = custom_output.to("cpu")
            # torch.set_printoptions(threshold=float('inf'))
            print(
                f"m:{m}, k:{k}, n:{n} 结果不匹配: deep_gemm: {custom_output},\n torch_gemm_fp8:{ref_output_fp8},\n torch_gemm_fp16:{ref_output},"
            )
            assert_close(
                custom_output,
                ref_output_fp8,
                rtol=0.01,
                atol=0.04,
                msg=f"m:{m}, k:{k}, n:{n} 结果不匹配: deep_gemm: {custom_output},\n torch_gemm_fp8:{ref_output_fp8},\n torch_gemm_fp16:{ref_output},",
            )


if __name__ == "__main__":
    unittest.main()
