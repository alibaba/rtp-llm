# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest

import torch
import torch.nn.functional as F

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"


def per_token_quant_int8(x):
    x = x.to(torch.float32)
    per_token_amax, _ = torch.max(input=torch.abs(x), dim=-1, keepdim=True)
    per_token_scale = per_token_amax / torch.iinfo(torch.int8).max  
    per_token_scale[per_token_scale == 0] = 1
    y = (x / per_token_scale).to(torch.int8)
    y_scale = per_token_scale.to(torch.float32)
    return y, y_scale


def per_tensor_quant_int8(x: torch.Tensor):
    x = x.to(torch.float32)
    amax = torch.max(torch.abs(x))
    int8_max = torch.iinfo(torch.int8).max  # 127
    scale = amax / int8_max

    if scale == 0 or torch.isnan(scale) or torch.isinf(scale):
        scale = torch.tensor(1.0, dtype=torch.float32, device=x.device)
    else:
        scale = scale.to(torch.float32)

    y = (x / scale).to(torch.int8)
    y_scale = scale

    return y, y_scale

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
    
    def _int8_gemm_ref(self, x, x_scale, weight, w_scale, bias):

        x_fp = x.to(torch.float32) * x_scale         # [m,k]
        w_fp = weight.to(torch.float32) * w_scale    # [k,n] * [n] -> [k,n]
        y = F.linear(x_fp, w_fp)                     # [m,n]
        if bias is not None:
            y = y.to(bias) + bias                  
        return y.to(torch.float32)
    
    def test_pertensor_int8_gemm(self):
        for n, k in [
            (9216, 4096),
            (4608, 4096),
        ]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 16384, 32768]:                
                input = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.1
                input_quant, input_scale = per_tensor_quant_int8(input)
                
                weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.1
                weight_quant, weight_scale = per_token_quant_int8(weight)
                
                weight_quant = weight_quant.contiguous()
                weight_scale = weight_scale.squeeze(0).contiguous()
                
                bias = torch.rand([1, n], device="cuda", dtype=torch.float16) * 10
                
                # print(f"[DEBUG_Python] x.shape={tuple(input_quant.shape)}", flush=True)
                # print(f"[DEBUG_Python] x_scale.shape={tuple(input_scale.shape)}", flush=True)
                # print(f"[DEBUG_Python] weight.shape={tuple(weight_quant.shape)}", flush=True)
                # print(f"[DEBUG_Python] w_scale.shape={tuple(weight_scale.shape)}", flush=True)
                # print(f"[DEBUG_Python] bias.shape={tuple(bias.shape) if bias is not None else None}", flush=True)
                                
                torch_output = self._int8_gemm_ref(input_quant, input_scale, weight_quant, weight_scale, bias).to("cpu")

                weight_quant = weight_quant.t()  # k,n
                weight_scale = weight_scale.t()  # 1,n
                
                custom_output = torch.zeros((m, n), device="cuda", dtype=torch.float16)
                
                self.gemm_op.forward_with_input_scale(
                    input_quant, input_scale, weight_quant, weight_scale, custom_output, bias
                )
                custom_output = custom_output.to(torch.float32).to("cpu")
                detailed_assert_close(
                    custom_output,
                    torch_output,
                    rtol=1e-2,
                    atol=4e-2,
                    msg=f"m:{m}, k:{k}, n:{n} 结果不匹配: rtp: {custom_output},\n torch: {torch_output},",
                )
                print(f"dims({m=}, {n=}, {k=}) passed~")

if __name__ == "__main__":
    unittest.main()

