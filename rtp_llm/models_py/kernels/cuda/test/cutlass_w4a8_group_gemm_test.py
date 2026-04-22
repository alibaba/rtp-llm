# !/usr/bin/python3
# coding=utf-8

import itertools
from unittest import SkipTest, TestCase, main

import torch

from rtp_kernel.w4a8_group_gemm import (
    w4a8_group_gemm_ptpc,
    unified_encode_int4b,
    reorder_tensor,
    compute_reorder_stride,
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


class W4a8GroupGemmOpTest(TestCase):
    NUM_EXPERT = [5, 128]
    M = [1, 8, 16, 32, 64, 128, 1024]
    GROUP_SIZE = [128]
    OUTPUT_TYPE = [torch.bfloat16]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")
        self.device = "cuda"

    def _run_w4a8_group_gemm_op_test(
        self,
        num_expert: int,
        m: int,
        group_size: int,
        output_dtype: torch.dtype
    ):
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
        b_unified = reorder_tensor(b_unified)

        b_scales = torch.empty(
            (num_expert, k // group_size, n), dtype=torch.float8_e4m3fn, device=self.device)
        initialize_tensor(b_scales, -0.1, 0.1)
        b_packed_scales = pack_scale_fp8(b_scales)

        b_zero = torch.empty(
            (num_expert, k // group_size, n), dtype=torch.float8_e4m3fn, device=self.device)
        initialize_tensor(b_zero, 0., 0.)

        a_out_scales = torch.ones(
            (num_expert * m, 1), dtype=torch.float32, device=self.device)

        a_strides = torch.full(
            (num_expert,), k, dtype=torch.int64, device=self.device)
        b_strides = compute_reorder_stride(num_expert, n, k)
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
            a_strides,
            b_strides,
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
            self.NUM_EXPERT, self.M, self.GROUP_SIZE, self.OUTPUT_TYPE
        ):
            with self.subTest(
                num_expert=params[0],
                m=params[1],
                group_size=params[2],
                output_dtype=params[3]
            ):
                self._run_w4a8_group_gemm_op_test(*params)

    def _run_pack_scale_fp8_gpu_test(self, m: int, n: int):
        input = torch.rand((m, n), device=self.device).to(torch.float8_e4m3fn)
        output = pack_scale_fp8(input)  # cuda version
        output_ref = pack_scale_fp8(input.cpu()).to(self.device)  # cpu version
        torch.testing.assert_close(output, output_ref)

    def test_pack_scale_fp8_gpu(self):
        for params in itertools.product(
            [1024, 2048, 4096],
            [1024, 2048, 4096],
        ):
            with self.subTest(m=params[0], n=params[1]):
                self._run_pack_scale_fp8_gpu_test(*params)

    def _run_unified_encode_int4b_gpu_test(self, m: int, n: int):
        input = torch.rand((m, n), device=self.device).to(torch.int8)
        output = unified_encode_int4b(input)  # cuda version
        output_ref = unified_encode_int4b(
            input.cpu()).to(self.device)  # cpu version
        torch.testing.assert_close(output, output_ref)

    def test_unified_encode_int4b_gpu(self):
        for params in itertools.product(
            [1024, 2048, 4096],
            [1024, 2048, 4096],
        ):
            with self.subTest(m=params[0], n=params[1]):
                self._run_unified_encode_int4b_gpu_test(*params)

    def _run_reorder_tensor_3d_gpu_test(self, num_expert: int, n: int, k: int):
        input = torch.rand(
            (num_expert, n, k // 2), device=self.device).to(torch.int8)
        output = reorder_tensor(input)  # cuda version
        output_ref = reorder_tensor(
            input.cpu()).to(self.device)  # cpu version
        torch.testing.assert_close(output, output_ref)

    def test_reorder_tensor_3d_gpu(self):
        for params in itertools.product(
            [1, 5],
            [1024, 4096],
            [1024, 2048],
        ):
            with self.subTest(num_expert=params[0], n=params[1], k=params[2]):
                self._run_reorder_tensor_3d_gpu_test(*params)

    def _run_reorder_tensor_2d_gpu_test(self, n: int, k: int):
        input = torch.rand((n, k // 2), device=self.device).to(torch.int8)
        output = reorder_tensor(input)  # cuda version, 2D
        output_ref = reorder_tensor(
            input.cpu()).to(self.device)  # cpu version, 2D
        torch.testing.assert_close(output, output_ref)

        # 2D result must match the corresponding slice of the 3D path.
        output_3d = reorder_tensor(input.unsqueeze(0)).squeeze(0)
        torch.testing.assert_close(output, output_3d)

    def test_reorder_tensor_2d_gpu(self):
        for params in itertools.product(
            [1024, 4096],
            [1024, 2048],
        ):
            with self.subTest(n=params[0], k=params[1]):
                self._run_reorder_tensor_2d_gpu_test(*params)


if __name__ == "__main__":
    main()
