import gc
import itertools
import os
import random
import shutil
import unittest
from typing import Callable, Optional

import matplotlib.pyplot as plt
import torch

from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_mul_bf16_deep_gemm_masked,
    silu_mul_fp8_quant_deep_gemm_masked,
    silu_mul_masked_bf16_no_post_quant_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.test.utils.bench_util import bench_compute_op
from rtp_llm.test.utils.cuda_graph_util import capture_graph
from rtp_llm.test.utils.numeric_util import calc_diff, per_token_cast_back


class SiluMulMaskedTest(unittest.TestCase):

    MAX_NUM_LOCAL_EXPERTS = 256
    MAX_EXPECTED_M = 1024
    MAX_MOE_INTERMEDIATE_SIZE = 5120

    STEP_SIZE_NUM_LOCAL_EXPERTS = 8
    STEP_SIZE_EXPECTED_M = 32
    STEP_SIZE_MOE_INTERMEDIATE_SIZE = 128

    SEARCH_NUM_LOCAL_EXPERTS_LIST = [1, 2, 4, 8, 16, 20, 32, 40] + list(
        range(48, 256, 64)
    )
    SEARCH_EXPECTED_M_LIST = (
        list(range(1, 16, 4)) + list(range(16, 512, 64)) + list(range(512, 1024, 256))
    )
    SEARCH_MOE_INTERMEDIATE_SIZE_LIST = list(range(128, 2560, 256)) + list(
        range(2560, 5120, 512)
    )

    NUM_LOCAL_EXPERTS = 4
    EXPECTED_M = 256
    MOE_INTERMEDIATE_SIZE = 2560

    # @classmethod
    # def setUpClass(cls) -> None:
    #     cls.output_dir = r"./silu_mul_masked_test_output"
    #     if os.path.exists(cls.output_dir):
    #         shutil.rmtree(cls.output_dir)
    #     os.makedirs(cls.output_dir)

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")
        # self.output_dir = SiluMulMaskedTest.output_dir

    def _generate_ref_output(self, up_gate_output: torch.Tensor) -> torch.Tensor:
        N = up_gate_output.shape[2]
        up = up_gate_output[..., : N // 2].to(torch.float32)
        gate = up_gate_output[..., N // 2 :].to(torch.float32)
        gate = gate * (1.0 / (1.0 + torch.exp(-gate)))
        ref_output = (gate * up).to(torch.bfloat16)
        return ref_output

    def _generate_test_data(
        self,
        num_local_experts: int,
        expected_m: int,
        moe_intermediate_size: int,
        is_fp8: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Generate random masked_m
        masked_m = torch.empty((num_local_experts,), device="cuda", dtype=torch.int)
        for i in range(num_local_experts):
            masked_m[i] = max(1, int(expected_m * random.uniform(0.7, 1.3)))
        max_m = (masked_m.max().item() + 127) // 128 * 128
        # Generate random up_gate_output
        up_gate_output = torch.randn(
            (num_local_experts, max_m, moe_intermediate_size * 2),
            device="cuda",
            dtype=torch.bfloat16,
        )
        # Mask out tokens beyond masked_m for each expert
        for i in range(num_local_experts):
            up_gate_output[i, masked_m[i] :, :] = 0
        # Generate test output
        if is_fp8:
            test_new_output = torch.zeros(
                (num_local_experts, max_m, moe_intermediate_size),
                device="cuda",
                dtype=torch.float32,
            ).to(torch.float8_e4m3fn)
            test_new_output_scale = torch.zeros(
                (num_local_experts, max_m, moe_intermediate_size // 128),
                device="cuda",
                dtype=torch.float32,
            )
            return masked_m, up_gate_output, test_new_output, test_new_output_scale
        else:
            test_new_output = torch.zeros(
                (num_local_experts, max_m, moe_intermediate_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            return masked_m, up_gate_output, test_new_output

    def _clean_test_data_cache(self, index: int):
        if index % 1 == 0:
            torch.cuda.empty_cache()
        if index % 2 == 0:
            gc.collect()

    def _compare_output_diff(
        self,
        up_gate_output: torch.Tensor,
        fn: Callable,
        test_new_output: torch.Tensor,
        test_new_output_scale: Optional[torch.Tensor] = None,
    ) -> float:
        # Calculate ref output
        ref_output = self._generate_ref_output(up_gate_output)
        # Calculate test output
        fn()
        # Compare outputs
        if test_new_output_scale is not None:
            test_new_output = per_token_cast_back(
                test_new_output, test_new_output_scale
            )
        diff = calc_diff(test_new_output, ref_output)
        return diff

    def _calc_latency(
        self,
        fn: Callable,
    ) -> float:
        # Capture graph
        graph = capture_graph(lambda: fn(), num_warmups=2)
        # Profile
        latency_us = bench_compute_op(
            lambda: graph.replay(),
            num_warmups=2,
            num_tests=5,
            suppress_kineto_output=False,
            trace_path=None,
            position_shift=(3, 1),
        )
        return latency_us

    def _plot_latency_comparison(
        self,
        x_list: list[int],
        old_latency_list: list[float],
        new_latency_list: list[float],
        x_label: str,
        y_label: str,
        title: str,
        output_path: str,
    ):
        plt.figure(figsize=(16, 10))
        plt.plot(
            x_list,
            old_latency_list,
            marker="o",
            markersize=6,
            linewidth=2,
            label="Old Implementation",
            color="blue",
        )
        plt.plot(
            x_list,
            new_latency_list,
            marker="s",
            markersize=6,
            linewidth=2,
            label="New Implementation",
            color="red",
        )
        plt.xlabel(x_label, fontsize=14, fontweight="bold")
        plt.ylabel(y_label, fontsize=14, fontweight="bold")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(fontsize=12, loc="best")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def test_silu_mul_masked_fp8_iterative_num_local_experts_correctness(self):
        for i, num_local_experts in enumerate(
            range(
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
                self.MAX_NUM_LOCAL_EXPERTS + 1,
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
            )
        ):
            with self.subTest(
                num_local_experts=num_local_experts,
                expected_m=self.EXPECTED_M,
                moe_intermediate_size=self.MOE_INTERMEDIATE_SIZE,
            ):
                masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                    self._generate_test_data(
                        num_local_experts,
                        self.EXPECTED_M,
                        self.MOE_INTERMEDIATE_SIZE,
                        is_fp8=True,
                    )
                )

                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        scale_ue8m0=False,
                    ),
                    test_new_output=test_new_output,
                    test_new_output_scale=test_new_output_scale,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_fp8_iterative_expected_m_correctness(self):
        for i, expected_m in enumerate(
            range(
                self.STEP_SIZE_EXPECTED_M,
                self.MAX_EXPECTED_M + 1,
                self.STEP_SIZE_EXPECTED_M,
            )
        ):
            with self.subTest(
                num_local_experts=self.NUM_LOCAL_EXPERTS,
                expected_m=expected_m,
                moe_intermediate_size=self.MOE_INTERMEDIATE_SIZE,
            ):
                masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                    self._generate_test_data(
                        self.NUM_LOCAL_EXPERTS,
                        expected_m,
                        self.MOE_INTERMEDIATE_SIZE,
                        is_fp8=True,
                    )
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        scale_ue8m0=False,
                    ),
                    test_new_output=test_new_output,
                    test_new_output_scale=test_new_output_scale,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_fp8_iterative_moe_intermediate_size_correctness(self):
        for i, moe_intermediate_size in enumerate(
            range(
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
                self.MAX_MOE_INTERMEDIATE_SIZE + 1,
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
            )
        ):
            with self.subTest(
                num_local_experts=self.NUM_LOCAL_EXPERTS,
                expected_m=self.EXPECTED_M,
                moe_intermediate_size=moe_intermediate_size,
            ):
                masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                    self._generate_test_data(
                        self.NUM_LOCAL_EXPERTS,
                        self.EXPECTED_M,
                        moe_intermediate_size,
                        is_fp8=True,
                    )
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        scale_ue8m0=False,
                    ),
                    test_new_output=test_new_output,
                    test_new_output_scale=test_new_output_scale,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_fp8_combined_correctness(self):
        for i, (num_local_experts, expected_m, moe_intermediate_size) in enumerate(
            itertools.product(
                self.SEARCH_NUM_LOCAL_EXPERTS_LIST,
                self.SEARCH_EXPECTED_M_LIST,
                self.SEARCH_MOE_INTERMEDIATE_SIZE_LIST,
            )
        ):
            with self.subTest(
                num_local_experts=num_local_experts,
                expected_m=expected_m,
                moe_intermediate_size=moe_intermediate_size,
            ):
                masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                    self._generate_test_data(
                        num_local_experts,
                        expected_m,
                        moe_intermediate_size,
                        is_fp8=True,
                    )
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        scale_ue8m0=False,
                    ),
                    test_new_output=test_new_output,
                    test_new_output_scale=test_new_output_scale,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_bf16_iterative_num_local_experts_correctness(self):
        for i, num_local_experts in enumerate(
            range(
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
                self.MAX_NUM_LOCAL_EXPERTS + 1,
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
            )
        ):
            with self.subTest(
                num_local_experts=num_local_experts,
                expected_m=self.EXPECTED_M,
                moe_intermediate_size=self.MOE_INTERMEDIATE_SIZE,
            ):
                masked_m, up_gate_output, test_new_output = self._generate_test_data(
                    num_local_experts,
                    self.EXPECTED_M,
                    self.MOE_INTERMEDIATE_SIZE,
                    is_fp8=False,
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        group_size=128,
                    ),
                    test_new_output=test_new_output,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_bf16_iterative_expected_m_correctness(self):
        for i, expected_m in enumerate(
            range(
                self.STEP_SIZE_EXPECTED_M,
                self.MAX_EXPECTED_M + 1,
                self.STEP_SIZE_EXPECTED_M,
            )
        ):
            with self.subTest(
                num_local_experts=self.NUM_LOCAL_EXPERTS,
                expected_m=expected_m,
                moe_intermediate_size=self.MOE_INTERMEDIATE_SIZE,
            ):
                masked_m, up_gate_output, test_new_output = self._generate_test_data(
                    self.NUM_LOCAL_EXPERTS,
                    expected_m,
                    self.MOE_INTERMEDIATE_SIZE,
                    is_fp8=False,
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        group_size=128,
                    ),
                    test_new_output=test_new_output,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_bf16_iterative_moe_intermediate_size_correctness(self):
        for i, moe_intermediate_size in enumerate(
            range(
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
                self.MAX_MOE_INTERMEDIATE_SIZE + 1,
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
            )
        ):
            with self.subTest(
                num_local_experts=self.NUM_LOCAL_EXPERTS,
                expected_m=self.EXPECTED_M,
                moe_intermediate_size=moe_intermediate_size,
            ):
                masked_m, up_gate_output, test_new_output = self._generate_test_data(
                    self.NUM_LOCAL_EXPERTS,
                    self.EXPECTED_M,
                    moe_intermediate_size,
                    is_fp8=False,
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        group_size=128,
                    ),
                    test_new_output=test_new_output,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    def test_silu_mul_masked_bf16_combined_correctness(self):
        for i, (num_local_experts, expected_m, moe_intermediate_size) in enumerate(
            itertools.product(
                self.SEARCH_NUM_LOCAL_EXPERTS_LIST,
                self.SEARCH_EXPECTED_M_LIST,
                self.SEARCH_MOE_INTERMEDIATE_SIZE_LIST,
            )
        ):
            with self.subTest(
                num_local_experts=num_local_experts,
                expected_m=expected_m,
                moe_intermediate_size=moe_intermediate_size,
            ):
                masked_m, up_gate_output, test_new_output = self._generate_test_data(
                    num_local_experts,
                    expected_m,
                    moe_intermediate_size,
                    is_fp8=False,
                )
                diff = self._compare_output_diff(
                    up_gate_output=up_gate_output,
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        group_size=128,
                    ),
                    test_new_output=test_new_output,
                )
                self.assertLess(diff, 0.001)
                self._clean_test_data_cache(i)

    @unittest.skip("Skip profile fp8 silu mul masked test")
    def test_profile_fp8_silu_mul_masked(self):
        # Generate test data
        masked_m, up_gate_output, test_new_output, test_new_output_scale = (
            self._generate_test_data(
                self.NUM_LOCAL_EXPERTS,
                self.EXPECTED_M,
                self.MOE_INTERMEDIATE_SIZE,
            )
        )
        # Capture silu_mul_masked_fp8_post_quant_fwd graph
        new_graph = capture_graph(
            lambda: silu_mul_masked_fp8_post_quant_fwd(
                input=up_gate_output,
                output=test_new_output,
                output_scale=test_new_output_scale,
                quant_group_size=128,
                masked_m=masked_m,
                expected_m=self.EXPECTED_M,
                scale_ue8m0=False,
            ),
            num_warmups=2,
        )
        # Profile silu_mul_masked_fp8_post_quant_fwd
        _ = bench_compute_op(
            lambda: new_graph.replay(),
            num_warmups=2,
            num_tests=5,
            suppress_kineto_output=False,
            trace_path=os.path.join(
                self.output_dir, "silu_mul_masked_fp8_post_quant_fwd_profile.json"
            ),
            position_shift=(3, 1),
        )
        self._clean_test_data_cache(0)
        # Capture silu_mul_fp8_quant_deep_gemm_masked graph
        old_graph = capture_graph(
            lambda: silu_mul_fp8_quant_deep_gemm_masked(
                y=up_gate_output,
                tokens_per_expert=masked_m,
                group_size=128,
                use_ue8m0=False,
                eps=1e-10,
            ),
            num_warmups=2,
        )
        # Profile silu_mul_fp8_quant_deep_gemm_masked
        _ = bench_compute_op(
            lambda: old_graph.replay(),
            num_warmups=2,
            num_tests=5,
            suppress_kineto_output=False,
            trace_path=os.path.join(
                self.output_dir, "silu_mul_fp8_quant_deep_gemm_masked_profile.json"
            ),
            position_shift=(3, 1),
        )
        self._clean_test_data_cache(0)

    @unittest.skip("Skip profile bf16 silu mul masked test")
    def test_profile_bf16_silu_mul_masked(self):
        # Generate test data
        masked_m, up_gate_output, test_new_output = self._generate_test_data(
            self.NUM_LOCAL_EXPERTS,
            self.EXPECTED_M,
            self.MOE_INTERMEDIATE_SIZE,
            is_fp8=False,
        )
        # Capture silu_mul_masked_bf16_no_post_quant_fwd graph
        new_graph = capture_graph(
            lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                input=up_gate_output,
                output=test_new_output,
                masked_m=masked_m,
                expected_m=self.EXPECTED_M,
                group_size=128,
            ),
            num_warmups=2,
        )
        # Profile silu_mul_masked_bf16_no_post_quant_fwd
        _ = bench_compute_op(
            lambda: new_graph.replay(),
            num_warmups=2,
            num_tests=5,
            suppress_kineto_output=False,
            trace_path=os.path.join(
                self.output_dir, "silu_mul_masked_bf16_no_post_quant_fwd_profile.json"
            ),
            position_shift=(3, 1),
        )
        self._clean_test_data_cache(0)
        # Capture silu_mul_bf16_deep_gemm_masked graph
        old_graph = capture_graph(
            lambda: silu_mul_bf16_deep_gemm_masked(
                y=up_gate_output,
                tokens_per_expert=masked_m,
                group_size=128,
            ),
            num_warmups=2,
        )
        # Profile silu_mul_bf16_deep_gemm_masked
        _ = bench_compute_op(
            lambda: old_graph.replay(),
            num_warmups=2,
            num_tests=5,
            suppress_kineto_output=False,
            trace_path=os.path.join(
                self.output_dir, "silu_mul_bf16_deep_gemm_masked_profile.json"
            ),
            position_shift=(3, 1),
        )
        self._clean_test_data_cache(0)

    @unittest.skip("Skip plot fp8 silu mul masked latency vs num local experts test")
    def test_plot_silu_mul_masked_fp8_latency_vs_num_local_experts(self):
        # Iterate over all possible values of NUM_LOCAL_EXPERTS
        old_latency_list = []
        new_latency_list = []
        num_local_experts_list = list(
            range(
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
                self.MAX_NUM_LOCAL_EXPERTS + 1,
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
            )
        )
        for i, num_local_experts in enumerate(num_local_experts_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                self._generate_test_data(
                    num_local_experts,
                    self.EXPECTED_M,
                    self.MOE_INTERMEDIATE_SIZE,
                    is_fp8=True,
                )
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_fp8_quant_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                        use_ue8m0=False,
                        eps=1e-10,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        scale_ue8m0=False,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=num_local_experts_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="Number of Local Experts",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked FP8 Post Quant Fwd vs Number of Local Experts",
            output_path=os.path.join(
                self.output_dir, "fp8_vs_num_local_experts_latency.png"
            ),
        )

    @unittest.skip("Skip plot fp8 silu mul masked latency vs expected m test")
    def test_plot_silu_mul_masked_fp8_latency_vs_expected_m(self):
        # Iterate over all possible values of EXPECTED_M
        old_latency_list = []
        new_latency_list = []
        expected_m_list = list(
            range(
                self.STEP_SIZE_EXPECTED_M,
                self.MAX_EXPECTED_M + 1,
                self.STEP_SIZE_EXPECTED_M,
            )
        )
        for i, expected_m in enumerate(expected_m_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                self._generate_test_data(
                    self.NUM_LOCAL_EXPERTS,
                    expected_m,
                    self.MOE_INTERMEDIATE_SIZE,
                    is_fp8=True,
                )
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_fp8_quant_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                        use_ue8m0=False,
                        eps=1e-10,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        scale_ue8m0=False,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=expected_m_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="Expected M",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked FP8 Post Quant Fwd vs Expected M",
            output_path=os.path.join(self.output_dir, "fp8_vs_expected_m_latency.png"),
        )

    @unittest.skip(
        "Skip plot fp8 silu mul masked latency vs moe intermediate size test"
    )
    def test_plot_silu_mul_masked_fp8_latency_vs_moe_intermediate_size(self):
        # Iterate over all possible values of MOE_INTERMEDIATE_SIZE
        old_latency_list = []
        new_latency_list = []
        moe_intermediate_size_list = list(
            range(
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
                self.MAX_MOE_INTERMEDIATE_SIZE + 1,
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
            )
        )
        for i, moe_intermediate_size in enumerate(moe_intermediate_size_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output, test_new_output_scale = (
                self._generate_test_data(
                    self.NUM_LOCAL_EXPERTS,
                    self.EXPECTED_M,
                    moe_intermediate_size,
                    is_fp8=True,
                )
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_fp8_quant_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                        use_ue8m0=False,
                        eps=1e-10,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_fp8_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        output_scale=test_new_output_scale,
                        quant_group_size=128,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        scale_ue8m0=False,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=moe_intermediate_size_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="MOE Intermediate Size",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked FP8 Post Quant Fwd vs MOE Intermediate Size",
            output_path=os.path.join(
                self.output_dir, "fp8_vs_moe_intermediate_size_latency.png"
            ),
        )

    @unittest.skip("Skip plot bf16 silu mul masked latency vs num local experts test")
    def test_plot_silu_mul_masked_bf16_latency_vs_num_local_experts(self):
        # Iterate over all possible values of NUM_LOCAL_EXPERTS
        old_latency_list = []
        new_latency_list = []
        num_local_experts_list = list(
            range(
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
                self.MAX_NUM_LOCAL_EXPERTS + 1,
                self.STEP_SIZE_NUM_LOCAL_EXPERTS,
            )
        )
        for i, num_local_experts in enumerate(num_local_experts_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output = self._generate_test_data(
                num_local_experts,
                self.EXPECTED_M,
                self.MOE_INTERMEDIATE_SIZE,
                is_fp8=False,
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_bf16_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        group_size=128,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=num_local_experts_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="Number of Local Experts",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked Bf16 No Post Quant Fwd vs Number of Local Experts",
            output_path=os.path.join(
                self.output_dir, "bf16_vs_num_local_experts_latency.png"
            ),
        )

    @unittest.skip("Skip plot bf16 silu mul masked latency vs expected m test")
    def test_plot_silu_mul_masked_bf16_latency_vs_expected_m(self):
        # Iterate over all possible values of EXPECTED_M
        old_latency_list = []
        new_latency_list = []
        expected_m_list = list(
            range(
                self.STEP_SIZE_EXPECTED_M,
                self.MAX_EXPECTED_M + 1,
                self.STEP_SIZE_EXPECTED_M,
            )
        )
        for i, expected_m in enumerate(expected_m_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output = self._generate_test_data(
                self.NUM_LOCAL_EXPERTS,
                expected_m,
                self.MOE_INTERMEDIATE_SIZE,
                is_fp8=False,
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_bf16_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=expected_m,
                        group_size=128,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=expected_m_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="Expected M",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked Bf16 No Post Quant Fwd vs Expected M",
            output_path=os.path.join(self.output_dir, "bf16_vs_expected_m_latency.png"),
        )

    @unittest.skip(
        "Skip plot bf16 silu mul masked latency vs moe intermediate size test"
    )
    def test_plot_silu_mul_masked_bf16_latency_vs_moe_intermediate_size(self):
        # Iterate over all possible values of MOE_INTERMEDIATE_SIZE
        old_latency_list = []
        new_latency_list = []
        moe_intermediate_size_list = list(
            range(
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
                self.MAX_MOE_INTERMEDIATE_SIZE + 1,
                self.STEP_SIZE_MOE_INTERMEDIATE_SIZE,
            )
        )
        for i, moe_intermediate_size in enumerate(moe_intermediate_size_list):
            # Generate test data
            masked_m, up_gate_output, test_new_output = self._generate_test_data(
                self.NUM_LOCAL_EXPERTS,
                self.EXPECTED_M,
                moe_intermediate_size,
                is_fp8=False,
            )
            # Calculate latency of old implementation
            old_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_bf16_deep_gemm_masked(
                        y=up_gate_output,
                        tokens_per_expert=masked_m,
                        group_size=128,
                    )
                )
            )
            # Calculate latency of new implementation
            new_latency_list.append(
                self._calc_latency(
                    fn=lambda: silu_mul_masked_bf16_no_post_quant_fwd(
                        input=up_gate_output,
                        output=test_new_output,
                        masked_m=masked_m,
                        expected_m=self.EXPECTED_M,
                        group_size=128,
                    )
                )
            )
            self._clean_test_data_cache(i)
        # Plot latency comparison
        self._plot_latency_comparison(
            x_list=moe_intermediate_size_list,
            old_latency_list=old_latency_list,
            new_latency_list=new_latency_list,
            x_label="MOE Intermediate Size",
            y_label="Latency (us)",
            title="Latency Comparison of Silu Mul Masked Bf16 No Post Quant Fwd vs MOE Intermediate Size",
            output_path=os.path.join(
                self.output_dir, "bf16_vs_moe_intermediate_size_latency.png"
            ),
        )


if __name__ == "__main__":
    unittest.main()
