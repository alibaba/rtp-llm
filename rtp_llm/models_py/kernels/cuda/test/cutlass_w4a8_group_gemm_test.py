# !/usr/bin/python3
# coding=utf-8

import itertools
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.kernels.cuda.w4a8_kernel import (
    w4a8_group_gemm_ptpc,
)

from rtp_llm.ops.compute_ops import (
    unified_encode_int4b,
    pack_scale_fp8,
    initialize_tensor,
    dequantize_int4b_to_fp8,
    block_compare_relative,
)


def torch_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    output_dtype: torch.dtype
) -> torch.Tensor:
    output = torch.matmul(a.to(output_dtype), b.to(output_dtype))
    output = output * a_scales.to(output_dtype)
    return output


class W4A8GroupGemmOpTest(TestCase):
    NUM_EXPERT = [5, 128]
    GROUP_SIZE = [128]
    OUTPUT_TYPE = [torch.bfloat16, torch.float16]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")
        self.device = "cuda"

    def _run_w4a8_group_gemm_op_test(
        self,
        num_expert: int,
        group_size: int,
        output_dtype: torch.dtype
    ):
        m = 1024
        n = 4096
        k = 2048
        assert k % group_size == 0

        expert_offsets = torch.zeros(
            (num_expert + 1), dtype=torch.int32, device=self.device)

        problem_sizes = torch.zeros(
            (num_expert, 3), dtype=torch.int32, device=self.device)

        a = torch.empty(
            (num_expert * m, k), dtype=torch.float8_e4m3fn, device=self.device)
        initialize_tensor(a, -0.1, 0.1)

        b = torch.empty(
            (num_expert, n, k // 2), dtype=torch.int8, device=self.device)
        initialize_tensor(b)
        b_unified = unified_encode_int4b(b)

        b_scales = torch.empty(
            (num_expert, n, k // group_size), dtype=torch.float8_e4m3fn, device=self.device)
        initialize_tensor(b_scales, -0.1, 0.1)
        b_packed_scales = pack_scale_fp8(b_scales)

        b_zero = torch.empty(
            (num_expert, n, k // group_size), dtype=torch.float8_e4m3fn, device=self.device)
        initialize_tensor(b_zero, 0., 0.)

        a_out_scales = torch.ones(
            (num_expert * m, 1), dtype=torch.float32, device=self.device)

        ab_strides = torch.full(
            (num_expert,), k, dtype=torch.int64, device=self.device)
        b_scales_strides = torch.tensor(
            [n, 0], dtype=torch.int64, device=self.device).unsqueeze(0).repeat(num_expert, 1, 1)
        c_strides = torch.full(
            (num_expert,), n, dtype=torch.int64, device=self.device)

        output = torch.zeros(
            (num_expert * m, n), dtype=output_dtype, device=self.device)

        ref = torch.zeros(
            (num_expert * m, n), dtype=output_dtype, device=self.device)

        for e in range(num_expert):
            expert_offsets[e + 1] = (e + 1) * m
            problem_sizes[e][0] = n
            problem_sizes[e][1] = m
            problem_sizes[e][2] = k

            a_e = a[e * m:(e + 1) * m]
            b_e = b[e].squeeze(0)
            b_scales_e = b_scales[e].squeeze(0)
            b_zero_e = b_zero[e].squeeze(0)
            a_out_scales_e = a_out_scales[e * m:(e + 1) * m]

            b_e_fp8 = dequantize_int4b_to_fp8(
                b_e, b_scales_e, b_zero_e, group_size)

            ref[e * m:(e + 1) * m] = torch_ref(a_e,
                                               b_e_fp8.T, a_out_scales_e, output_dtype)

        w4a8_group_gemm_ptpc(
            output,
            a,
            b_unified,
            b_packed_scales,
            a_out_scales,
            expert_offsets[:-1],
            problem_sizes,
            ab_strides,
            ab_strides,
            b_scales_strides,
            c_strides,
            group_size
        )

        for e in range(num_expert):
            o = output[e * m:(e + 1) * m]
            r = ref[e * m:(e + 1) * m]
            if not block_compare_relative(o, r):
                # print(f"o[{e}]: {o}")
                # print(f"r[{e}]: {r}")
                torch.testing.assert_close(o, r, rtol=5e-2, atol=1e-2)

    def test_w4a8_group_gemm(self):
        for params in itertools.product(
            self.NUM_EXPERT, self.GROUP_SIZE, self.OUTPUT_TYPE
        ):
            with self.subTest(
                num_expert=params[0],
                group_size=params[1],
                output_dtype=params[2]
            ):
                self._run_w4a8_group_gemm_op_test(*params)


if __name__ == "__main__":
    main()
