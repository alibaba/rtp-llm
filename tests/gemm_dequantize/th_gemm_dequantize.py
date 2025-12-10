import os
import unittest

import numpy as np
import torch

from rtp_llm.device import get_current_device
from rtp_llm.device.device_impl import GpuImpl


def random_tensor(shape, dtype, device, mean=0, std=1):
    return torch.empty(shape, dtype=dtype, device=device).normal_(mean, std)


from rtp_llm.ops.compute_ops import rtp_llm_ops


class TestGemmDequantize(unittest.TestCase):
    def setUp(self) -> None:
        torch.classes.load_library(
            os.environ["TEST_SRCDIR"] + "/rtp_llm/tests/libtest_ops.so"
        )
        self.fused_gemm_dq = torch.ops.gemm_dq_unit_ops.fused_gemm_dq
        self.bench = torch.ops.gemm_dq_unit_ops.benchmark_against_cublas_fp
        self.unpack_packed_int4s = torch.ops.gemm_dq_unit_ops.unpack_int4_packed_tensor_to_int8

        self.device = get_current_device()
        assert isinstance(self.device, GpuImpl)
        self.pack_int4s = self.device.pack_int8_tensor_to_packed_int4
        self.preprocess_weights_for_mixed_gemm = (
            self.device.preprocess_weights_for_mixed_gemm
        )
        self.symmetric_quantizer = (
            self.device.symmetric_quantize_last_axis_of_batched_matrix
        )

        torch.manual_seed(734876213)

    def gemm_dequant_test_helper(
        self,
        compute_type,
        weight_dtype,
        gemm_ms,
        gemm_ns,
        gemm_ks,
        rtol,
        atol,
        use_tensor_core,
        quant_tol=(1e-2, 0.1, 0.2, 0.05),
        benchmark=False,
    ):
        assert (
            weight_dtype == torch.int8 or weight_dtype == torch.quint4x2
        ), "Weight must be quantized"

        for gemm_k in gemm_ks:
            for gemm_n in gemm_ns:
                torch_weights = random_tensor(
                    (gemm_k, gemm_n),
                    dtype=compute_type,
                    device="cuda",
                    mean=0,
                    std=0.002,
                )
                ref_torch_weights, processed_torch_weights, torch_weight_scales = (
                    self.symmetric_quantizer(torch_weights, weight_dtype, True)
                )
                ref_torch_weights = (
                    self.unpack_packed_int4s(ref_torch_weights.cpu())
                    if weight_dtype == torch.quint4x2
                    else ref_torch_weights
                )
                ref_torch_weights = ref_torch_weights.to("cuda")
                processed_torch_weights = processed_torch_weights.to("cuda")
                torch_weight_scales = torch_weight_scales.to("cuda")
                zeros = torch.Tensor().half()

                # dequantized and compare diff
                scales_unsqueezed = torch_weight_scales.unsqueeze(0)
                dequantized_weights = torch.multiply(
                    ref_torch_weights, scales_unsqueezed
                )

                mse_tol, max_err_tol, rel_err_tol, bad_tol = quant_tol
                diff = dequantized_weights - torch_weights
                mse = diff.pow(2).mean().sqrt()
                assert mse < mse_tol, f"RMSE to large: {mse}"

                max_err = diff.abs().max()
                assert (
                    max_err < max_err_tol
                ), f"Max absolute error exceeds threshold: {max_err}"

                relative_err = diff.abs() / (torch_weights.abs() + 1e-6)
                bad_ratio = (
                    (relative_err > rel_err_tol).float().mean()
                )  # proportion of elements with >20% relative error
                assert (
                    bad_ratio < bad_tol
                ), f"Too many elements with high relative error: {bad_ratio:.2%}"

                for num_rows in gemm_ms:
                    torch_activations = torch.randn(
                        size=(num_rows, gemm_k), dtype=compute_type, device="cuda"
                    )

                    dequantized_weights = dequantized_weights.to(
                        torch_activations.dtype
                    )

                    if benchmark:
                        torch.cuda.profiler.start()
                        times, results = self.bench(
                            torch_activations,
                            processed_torch_weights,
                            torch_weight_scales,
                            dequantized_weights,
                            200,
                        )
                        torch.cuda.profiler.stop()
                        times = times[0]
                        cublas_time = times[0].item()
                        ft_time = times[1].item()
                        ft_speedup = cublas_time / ft_time
                        print(
                            "{},{},{},{},{},{}".format(
                                num_rows,
                                gemm_n,
                                gemm_k,
                                cublas_time,
                                ft_time,
                                ft_speedup,
                            )
                        )
                        reference_result = results[0]
                        ft_result = results[1]
                    else:
                        reference_result = torch.matmul(
                            torch_activations, dequantized_weights
                        )
                        ft_result = self.fused_gemm_dq(
                            torch_activations,
                            processed_torch_weights,
                            torch_weight_scales,
                            zeros,
                            gemm_k,
                            use_tensor_core,
                        )

                    msg = "FC1 Failed on m={}, n={}, k={}".format(
                        num_rows, gemm_n, gemm_k
                    )
                    torch.testing.assert_close(
                        ft_result,
                        reference_result,
                        rtol=rtol,
                        atol=atol,
                        msg=msg,
                        check_dtype=False,
                    )

    def test_fp16_int8_gemv(self):
        self.gemm_dequant_test_helper(
            torch.float16,
            torch.int8,
            gemm_ms=[1, 2, 3, 4],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[512, 768, 1024, 4096, 11008],
            rtol=0.001,
            atol=0.002,
            quant_tol=(1e-2, 0.1, 0.2, 0.05),
            use_tensor_core=False,
        )

    def test_fp16_int8_gemm(self):
        self.gemm_dequant_test_helper(
            torch.float16,
            torch.int8,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            rtol=0.001,
            atol=0.002,
            quant_tol=(1e-2, 0.1, 0.2, 0.05),
            use_tensor_core=True,
        )

    def woq_groupwise_extract_int4(self, w_packed, uint4_input=False):
        w_packed_int8 = w_packed.T.contiguous().view(torch.uint8)
        w_unpacked_int4 = torch.stack(
            ((w_packed_int8 % 16).view(-1, 1), (w_packed_int8 // 16).view(-1, 1)), dim=1
        )
        # Unpacked uint4s
        w_unpacked_int4 = (
            w_unpacked_int4.flatten().view(w_packed.shape[1], -1).T.contiguous().int()
        )
        if not uint4_input:
            w_unpacked_int4 -= 8
        return w_unpacked_int4

    def woq_assert_colwise_near_eq(self, ref, act):
        bits_in_type = 4
        quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

        # check each column independently
        if ref.shape[0] > 1:
            for col_idx in range(ref.shape[-1]):
                col = ref[:, col_idx]
                max_val = torch.max(col).item()
                atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
                np.testing.assert_allclose(
                    col.cpu().numpy(), act[:, col_idx].cpu().numpy(), atol=atol
                )
        else:
            max_val = torch.max(ref).item()
            atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
            np.testing.assert_allclose(ref.cpu().numpy(), act.cpu().numpy(), atol=atol)

    def groupwise_gemm_dequant_test_helper(
        self, compute_type, gemm_ms, gemm_ns, gemm_ks, group_size
    ):
        uint4_input = 1
        for gemm_m in gemm_ms:
            for gemm_k in gemm_ks:
                for gemm_n in gemm_ns:
                    torch.manual_seed(0)
                    activation = (
                        torch.rand((gemm_m, gemm_k), dtype=compute_type) * 2 - 1.0
                    )
                    qweight_unprocessed = torch.randint(
                        -(2**31), 2**31, (gemm_k // 8, gemm_n)
                    ).int()
                    scale = (
                        torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type)
                        * 2
                    )
                    zero = (
                        torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type)
                        * 2
                    )

                    qweight_int8 = self.woq_groupwise_extract_int4(
                        qweight_unprocessed, uint4_input
                    ).char()
                    qweight_int4x2_interleaved = self.preprocess_weights_for_mixed_gemm(
                        self.pack_int4s(qweight_int8 - uint4_input * 8),
                        torch.quint4x2,
                    )

                    ref_th_weight = qweight_int8.half() * scale.repeat_interleave(
                        group_size, dim=0
                    ) - uint4_input * 8 * scale.repeat_interleave(group_size, dim=0)

                    ref_th_weight += zero.repeat_interleave(group_size, dim=0)

                    ft_result = self.fused_gemm_dq(
                        activation.cuda(),
                        qweight_int4x2_interleaved.cuda(),
                        scale.cuda(),
                        zero.cuda(),
                        group_size,
                        True,
                    )

                    reference_result = activation.cuda().matmul(
                        ref_th_weight.cuda().to(compute_type)
                    )
                    self.woq_assert_colwise_near_eq(reference_result, ft_result)

    def test_fp16_int4_gemm(self):
        self.groupwise_gemm_dequant_test_helper(
            torch.float16,
            gemm_ms=[1, 16, 32, 44, 256, 37],
            gemm_ns=[64, 128, 1024, 2048, 4096],
            gemm_ks=[64, 128, 1024, 4096],
            group_size=64,
        )

    def test_fp16_int4_gemm2(self):
        self.groupwise_gemm_dequant_test_helper(
            torch.float16,
            gemm_ms=[1, 16, 32, 44, 256, 37],
            gemm_ns=[64, 128, 1024, 2048, 4096],
            gemm_ks=[128, 1024, 4096],
            group_size=128,
        )

    @unittest.skip("Not test yet")
    def test_bf16_int4_gemm(self):
        self.groupwise_gemm_dequant_test_helper(
            torch.bfloat16,
            gemm_ms=[256, 177, 195, 125, 66, 33, 8, 2, 1],
            gemm_ns=[1024, 2048, 4096],
            gemm_ks=[4096, 8192, 16384],
            group_size=128,
        )

    def bench_helper(self, act_type, quant_type, rtol, atol):
        # Warm, using bfloat here since it seems to reliably use cublas.
        x = random_tensor([20480, 20480], torch.bfloat16, device="cuda")
        warm_iters = 30
        for iter in range(warm_iters):
            res = x @ x

        m_shapes = torch.arange(0, 12)
        m_shapes = 2**m_shapes

        self.gemm_dequant_test_helper(
            act_type,
            quant_type,
            gemm_ms=[128],
            gemm_ns=[1536],
            gemm_ks=[12288],
            rtol=rtol,
            atol=atol,
            use_tensor_core=True,
            benchmark=True,
        )

    @unittest.skip("This is a benchmark so don't run by default")
    def test_fp16_int8_cublas(self):
        self.bench_helper(torch.float16, torch.int8, 1e-3, 0.002)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_bf16_int8_cublas(self):
        self.bench_helper(torch.bfloat16, torch.int8, 1e-2, 1e-2)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_fp16_int4_cublas(self):
        self.bench_helper(torch.float16, torch.quint4x2, 1e-3, 0.002)

    @unittest.skip("This is a benchmark so don't run by default")
    def test_bf16_int4_cublas(self):
        self.bench_helper(torch.bfloat16, torch.quint4x2, 1e-2, 1e-2)


if __name__ == "__main__":
    unittest.main()
