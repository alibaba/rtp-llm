# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os
import unittest

import torch
import torch.nn.functional as F

os.environ["DEVICE_RESERVE_MEMORY_BYTES"] = "128000000"


def per_token_quant_fp8(x):
    x = x.to(torch.float32)
    per_token_amax, _ = torch.max(input=torch.abs(x), dim=-1, keepdim=True)
    per_token_scale = per_token_amax / torch.finfo(torch.float8_e4m3fnuz).max
    per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (x / per_token_scale).to(dtype=torch.float8_e4m3fnuz)
    y_scale = per_token_scale.to(torch.float32)
    return y, y_scale


def shuffle_weight(x, layout=(16, 16), use_int4=False):
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    return x_

def calculate_k_for_swizzling(dtype: torch.dtype):
    if dtype == torch.float32:
        MiK, MiKv = 4, 1
    elif dtype in (torch.float16, torch.half, torch.bfloat16):
        MiK, MiKv = 16, 4
    elif dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz):
        MiK, MiKv = 32, 8
    else:
        raise ValueError(f"unsupported datatype in calculateKforSwizzling: {dtype}")
    elem_size = torch.zeros((), dtype=dtype).element_size()
    PackK = 16 // MiKv // elem_size
    return MiK, MiKv, PackK

def swizzle_tensor(
    src: torch.Tensor,
    col_maj: bool = False,
    MiM: int = 16) -> torch.Tensor:
    tmp = src.clone()

    if col_maj:
        k, m = src.shape
        tmp = tmp.view(k, m).permute(1, 0).contiguous()
    else:
        m, k = src.shape

    MiK, MiKv, PackK = calculate_k_for_swizzling(src.dtype)

    if (MiK == 16):
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 32 == 0, f"swizzle shape k = {k} must be divisible by 32"
    elif (MiK == 32):
        assert m % 16 == 0, f"swizzle shape m = {m} must be divisible by 16"
        assert k % 64 == 0, f"swizzle shape k = {k} must be divisible by 64"

    tmp = tmp.view(m // MiM, MiM, k // (MiK * PackK), MiK // MiKv, MiKv * PackK)
    tmp = tmp.permute(0, 2, 3, 1, 4).contiguous()

    dst = tmp.clone()
    return dst.view(src.shape)
    

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

    def _fp8_a8w8_gemm_ref(self, input_quant, input_scale, weight_quant, weight_scale):
        # quant and dequant input
        input_ = input_quant.to(torch.float32) * input_scale

        weight = weight_quant.to(torch.float32) * weight_scale
        out = F.linear(input_, weight)
        return out

    def test_ptpc_gemm(self):
        for n, k in [
            (9216, 4096),
            (4608, 4096),
        ]:
            for m in [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 16384, 32768]:
                input = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
                input_quant, input_scale = per_token_quant_fp8(input)
                
                weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
                weight_quant, weight_scale = per_token_quant_fp8(weight)

                torch_output = self._fp8_a8w8_gemm_ref(input_quant, input_scale, weight_quant, weight_scale).to(
                    "cpu"
                )
                if os.environ.get("TEST_SWIZZLEA", None) == "1":
                    weight_quant_swizzle = swizzle_tensor(weight_quant, False)
                    weight_quant_swizzle = weight_quant_swizzle.t()
                    weight_scale = weight_scale.t()
                    custom_output = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)

                    self.gemm_op.forward_with_input_scale(
                        input_quant, input_scale, weight_quant_swizzle, weight_scale, custom_output, None
                    )
                else:
                    weight_quant_shuffle = shuffle_weight(weight_quant)
                    weight_quant_shuffle = weight_quant_shuffle.t()  # k,n

                    weight_scale = weight_scale.t()  # 1,n
                    custom_output = torch.zeros((m, n), device="cuda", dtype=torch.bfloat16)

                    self.gemm_op.forward_with_input_scale(
                        input_quant, input_scale, weight_quant_shuffle, weight_scale, custom_output, None
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
