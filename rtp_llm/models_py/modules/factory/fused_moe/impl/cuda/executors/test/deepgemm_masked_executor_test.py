import itertools
import random
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
    DeepGemmMaskedExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.fused_moe_executor_test_util import (
    compare_with_ref_per_expert,
    generate_payload_and_weights,
    generate_ref_output,
    make_deepgemm_masked_test_config,
    quantize_payload_fp8_per_token_group,
    quantize_weights_fp8_per_block,
)


class DeepGemmMaskedExecutorTestBase:
    # - hidden_size must be divisible by 128
    # - UE8M0 packed scale requires hidden_size divisible by 512 (K/128 divisible by 4)
    HIDDEN_SIZES = [6144]
    MOE_INTERMEDIATE_SIZES = [2560]
    NUM_EXPERTS = [160]
    EP_SIZES = [2]
    MAX_GENERATE_BATCH_SIZES = [63]

    def _test_deepgemm_masked_executor(
        self,
        use_fp8: bool,
        use_ue8m0_scale: bool,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        ep_size: int,
        max_generate_batch_size: int,
    ) -> None:
        if num_experts % ep_size != 0:
            self.skipTest(
                f"skip invalid ep config: num_experts={num_experts} not divisible by ep_size={ep_size}"
            )
        config = make_deepgemm_masked_test_config(
            use_fp8=use_fp8,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            ep_size=ep_size,
            max_generate_batch_size=max_generate_batch_size,
        )
        expert_payload, weights = generate_payload_and_weights(config)
        expert_num_tokens = expert_payload.expert_tokens_meta.expert_num_tokens
        ref_output = generate_ref_output(config, expert_payload, weights)
        test_ue8m0 = False

        if use_fp8:
            # UE8M0 scale format is only used on SM100/SM120, and requires K/128 divisible by 4.
            if use_ue8m0_scale and (not is_deep_gemm_e8m0_used()):
                self.skipTest("UE8M0 scale format only used on SM100/SM120")
            if is_deep_gemm_e8m0_used() and (hidden_size % 512 != 0):
                self.skipTest(
                    f"skip UE8M0: hidden_size={hidden_size} not divisible by 512"
                )
            test_ue8m0 = use_ue8m0_scale and is_deep_gemm_e8m0_used()
            quantize_payload_fp8_per_token_group(
                expert_payload, use_e8m0_scale=test_ue8m0
            )
            # Keep weights casting consistent with UE8M0 mode when enabled.
            quantize_weights_fp8_per_block(weights, use_ue8m0=False)
            assert expert_payload.expert_x_scale is not None
            if test_ue8m0:
                assert is_deep_gemm_e8m0_used()
                assert expert_payload.expert_x_scale.dtype in (torch.int32, torch.int)
                assert expert_payload.expert_x_scale.size(2) == hidden_size // 128 // 4

        executor = DeepGemmMaskedExecutor(config, config.quant_config, weights)
        torch.cuda.synchronize()

        combine_payload: CombineForwardPayload = executor.execute(
            payload=expert_payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        assert combine_payload.expert_executions is None
        torch.cuda.synchronize()

        if not use_fp8:
            diff_th = 1e-5
        elif test_ue8m0:
            diff_th = 3e-3
        else:
            diff_th = 2.2e-3
        compare_with_ref_per_expert(
            combine_payload.fused_expert_output, ref_output, expert_num_tokens, diff_th
        )

    def _test_bf16(self) -> None:
        ran_any = False
        for (
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
        ) in itertools.product(
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
        ):
            with self.subTest(
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
            ):
                self._test_deepgemm_masked_executor(
                    use_fp8=False,
                    use_ue8m0_scale=False,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                )
            ran_any = True
        if not ran_any:
            self.skipTest("No valid bf16 normal cases to run on this machine/config")

    def _test_fp8_per_block(self) -> None:
        ran_any = False
        for (
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
        ) in itertools.product(
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
        ):
            with self.subTest(
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
            ):
                self._test_deepgemm_masked_executor(
                    use_fp8=True,
                    use_ue8m0_scale=False,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block normal cases to run on this machine/config"
            )

    def _test_fp8_per_block_ue8m0(self) -> None:
        ran_any = False
        for (
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
        ) in itertools.product(
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
        ):
            with self.subTest(
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
            ):
                if hidden_size % 512 != 0:
                    self.skipTest(
                        f"skip UE8M0: hidden_size={hidden_size} not divisible by 512"
                    )
                self._test_deepgemm_masked_executor(
                    use_fp8=True,
                    use_ue8m0_scale=True,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block UE8M0 normal cases to run on this machine/config"
            )


class DeepGemmMaskedExecutorTest(DeepGemmMaskedExecutorTestBase, unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if not has_deep_gemm():
            self.skipTest("deep_gemm is not available")
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.set_device(0)
        torch.set_default_device("cuda")

    def test_bf16(self) -> None:
        self._test_bf16()

    def test_fp8_per_block(self) -> None:
        self._test_fp8_per_block()


if __name__ == "__main__":
    unittest.main()
