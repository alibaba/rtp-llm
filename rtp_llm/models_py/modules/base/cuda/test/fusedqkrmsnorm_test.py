import itertools
from unittest import SkipTest, TestCase, main

import numpy as np
import torch
from torch import dtype as _dtype

try:
    import flashinfer
    from flashinfer.testing.utils import bench_gpu_time
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

from rtp_llm.models_py.modules import FusedQKRMSNorm, QKRMSNorm


class FusedQKRMSNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HEAD_NUM = [40]
    KV_HEAD_NUM = [40, 8, 4]
    SIZE_PER_HEAD = [128]

    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [4096]
    # HEAD_NUM = [40]
    # KV_HEAD_NUM =  [40]
    # SIZE_PER_HEAD = [128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_qk_rmsnorm_test(
        self,
        num_tokens: int,
        head_num: int,
        kv_head_num: int,
        size_per_head: int,
        dtype: _dtype,
    ):
        torch.manual_seed(0)

        hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head

        q_weight = torch.randn(size_per_head, dtype=dtype)
        k_weight = torch.randn(size_per_head, dtype=dtype)

        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        fused_qkrmsnorm = FusedQKRMSNorm(
            q_weight, k_weight, head_num, kv_head_num, size_per_head
        )

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     # out = qkrmsnorm(x)
        #     out = fused_qkrmsnorm(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         # out = qkrmsnorm(x)
        #         out = fused_qkrmsnorm(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qkrmsnorm(x), fused_qkrmsnorm(x), atol=1e-2, rtol=1e-2)
        )

    def test_fusedqkrmsnorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HEAD_NUM,
            self.KV_HEAD_NUM,
            self.SIZE_PER_HEAD,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                head_num=params[1],
                kv_head_num=params[2],
                size_per_head=params[3],
                dtype=params[4],
            ):
                self._run_fused_qk_rmsnorm_test(*params)

    def test_bench_fusedqkrmsnorm(self):
        """Benchmark and compare flashinfer rmsnorm, rtp-llm FusedQKRMSNorm, and rtp-llm QKRMSNorm"""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not FLASHINFER_AVAILABLE:
            raise SkipTest("flashinfer is not available")

        torch.set_default_device("cuda")

        # Test configurations
        num_tokens_list = [4000]
        head_num = 32
        kv_head_num = 4
        size_per_head = 128
        dtypes = [torch.bfloat16]
        dry_run_time_ms = 25
        repeat_time_ms = 100

        print("=" * 150)
        print(
            f"{'Num Tokens':<12} {'Dtype':<12} {'Operator':<30} "
            f"{'Latency (us)':<15} {'Throughput (GB/s)':<20} {'Bandwidth (GB/s)':<20}"
        )
        print("=" * 150)

        for num_tokens in num_tokens_list:
            for dtype in dtypes:
                dtype_str = "float16" if dtype == torch.half else "bfloat16"
                torch.manual_seed(0)

                hidden_size = head_num * size_per_head + 2 * kv_head_num * size_per_head
                q_size = head_num * size_per_head
                kv_size = kv_head_num * size_per_head

                # Prepare weights
                q_weight = torch.randn(size_per_head, dtype=dtype, device="cuda")
                k_weight = torch.randn(size_per_head, dtype=dtype, device="cuda")

                # Prepare input
                x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")

                # Initialize operators
                qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
                fused_qkrmsnorm = FusedQKRMSNorm(
                    q_weight, k_weight, head_num, kv_head_num, size_per_head
                )

                # Benchmark flashinfer rmsnorm (apply to Q and K separately)
                # Keep a copy of x for flashinfer to ensure consistent input
                x_flashinfer = x.clone()
                # Reshape Q and K for per-head normalization
                # Q: [num_tokens, head_num * size_per_head] -> [num_tokens * head_num, size_per_head]
                # K: [num_tokens, kv_head_num * size_per_head] -> [num_tokens * kv_head_num, size_per_head]
                q_input = x_flashinfer[:, :q_size].reshape(-1, size_per_head)
                k_input = x_flashinfer[:, q_size:q_size + kv_size].reshape(-1, size_per_head)

                @torch.cuda.nvtx.range(
                    f"flashinfer_rmsnorm num_tokens={num_tokens}, dtype={dtype_str}"
                )
                def fn_flashinfer() -> None:
                    # Use fresh input for each iteration from the cloned x
                    flashinfer.norm.rmsnorm(q_input, q_weight, enable_pdl = True)
                    flashinfer.norm.rmsnorm(k_input, k_weight, enable_pdl = True)

                measurements_flashinfer = bench_gpu_time(
                    fn_flashinfer,
                    dry_run_time_ms=dry_run_time_ms,
                    repeat_time_ms=repeat_time_ms,
                )
                latency_ms_flashinfer = np.median(measurements_flashinfer)
                latency_us_flashinfer = latency_ms_flashinfer * 1e3

                # Calculate throughput for flashinfer
                # Input: q_reshaped + k_reshaped, weights: q_weight + k_weight, output: q_out + k_out
                total_elements_flashinfer = (
                    q_reshaped.numel() * 2 + k_reshaped.numel() * 2 + q_weight.numel() + k_weight.numel()
                )
                total_bytes_flashinfer = total_elements_flashinfer * x.element_size()
                throughput_gb_s_flashinfer = total_bytes_flashinfer / (latency_ms_flashinfer * 1e-3) * 1e-9

                # Effective bandwidth (read + write)
                io_bytes_flashinfer = (q_reshaped.numel() + k_reshaped.numel()) * x.element_size() * 2
                bandwidth_gb_s_flashinfer = io_bytes_flashinfer / (latency_ms_flashinfer * 1e-3) * 1e-9

                print(
                    f"{num_tokens:<12} {dtype_str:<12} {'flashinfer rmsnorm':<30} "
                    f"{latency_us_flashinfer:<15.2f} {throughput_gb_s_flashinfer:<20.3f} {bandwidth_gb_s_flashinfer:<20.3f}"
                )

                # Benchmark rtp-llm QKRMSNorm
                # QKRMSNorm returns a new tensor, so no need to clone
                @torch.cuda.nvtx.range(
                    f"rtp_llm_qkrmsnorm num_tokens={num_tokens}, dtype={dtype_str}"
                )
                def fn_qkrmsnorm() -> None:
                    _ = qkrmsnorm(x)

                measurements_qkrmsnorm = bench_gpu_time(
                    fn_qkrmsnorm,
                    dry_run_time_ms=dry_run_time_ms,
                    repeat_time_ms=repeat_time_ms,
                )
                latency_ms_qkrmsnorm = np.median(measurements_qkrmsnorm)
                latency_us_qkrmsnorm = latency_ms_qkrmsnorm * 1e3

                # Calculate throughput for QKRMSNorm
                # Input: x, weights: q_weight + k_weight, output: x (in-place or new)
                total_elements_qkrmsnorm = (
                    x.numel() * 2 + q_weight.numel() + k_weight.numel()
                )
                total_bytes_qkrmsnorm = total_elements_qkrmsnorm * x.element_size()
                throughput_gb_s_qkrmsnorm = total_bytes_qkrmsnorm / (latency_ms_qkrmsnorm * 1e-3) * 1e-9

                # Effective bandwidth
                io_bytes_qkrmsnorm = x.numel() * x.element_size() * 2
                bandwidth_gb_s_qkrmsnorm = io_bytes_qkrmsnorm / (latency_ms_qkrmsnorm * 1e-3) * 1e-9

                print(
                    f"{num_tokens:<12} {dtype_str:<12} {'rtp-llm QKRMSNorm':<30} "
                    f"{latency_us_qkrmsnorm:<15.2f} {throughput_gb_s_qkrmsnorm:<20.3f} {bandwidth_gb_s_qkrmsnorm:<20.3f}"
                )

                @torch.cuda.nvtx.range(
                    f"rtp_llm_fused_qkrmsnorm num_tokens={num_tokens}, dtype={dtype_str}"
                )
                def fn_fused_qkrmsnorm() -> None:
                    _ = fused_qkrmsnorm(x)

                measurements_fused = bench_gpu_time(
                    fn_fused_qkrmsnorm,
                    dry_run_time_ms=dry_run_time_ms,
                    repeat_time_ms=repeat_time_ms,
                )
                latency_ms_fused = np.median(measurements_fused)
                latency_us_fused = latency_ms_fused * 1e3

                # Calculate throughput for FusedQKRMSNorm
                total_elements_fused = (
                    x.numel() * 2 + q_weight.numel() + k_weight.numel()
                )
                total_bytes_fused = total_elements_fused * x.element_size()
                throughput_gb_s_fused = total_bytes_fused / (latency_ms_fused * 1e-3) * 1e-9

                # Effective bandwidth
                io_bytes_fused = x.numel() * x.element_size() * 2
                bandwidth_gb_s_fused = io_bytes_fused / (latency_ms_fused * 1e-3) * 1e-9

                print(
                    f"{num_tokens:<12} {dtype_str:<12} {'rtp-llm FusedQKRMSNorm':<30} "
                    f"{latency_us_fused:<15.2f} {throughput_gb_s_fused:<20.3f} {bandwidth_gb_s_fused:<20.3f}"
                )

            print("-" * 150)

        print("=" * 150)
        print("Benchmark completed!")


if __name__ == "__main__":
    main()
