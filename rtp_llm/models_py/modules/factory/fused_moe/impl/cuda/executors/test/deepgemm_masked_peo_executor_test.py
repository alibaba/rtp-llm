import itertools
import random
import unittest
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_peo_executor import (
    DeepGemmMaskedPeoExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.fused_moe_executor_test_util import (
    compare_with_ref_per_expert,
    generate_payload_and_weights,
    generate_ref_output,
    make_deepgemm_masked_test_config,
    make_dispatch_events,
    quantize_payload_fp8_per_token_group,
    quantize_weights_fp8_per_block,
)


class DeepGemmMaskedPeoExecutorTestBase:
    # - hidden_size must be divisible by 128
    # - UE8M0 packed scale requires hidden_size divisible by 512 (K/128 divisible by 4)
    HIDDEN_SIZES = [6144]
    MOE_INTERMEDIATE_SIZES = [2560]
    NUM_EXPERTS = [160]
    EP_SIZES = [2]
    MAX_GENERATE_BATCH_SIZES = [63]
    NUM_PEO_ROUNDS = [2]
    DEEP_EP_NUM_SM = [24]
    PEO_LEVELS = [3]

    def _test_deepgemm_masked_peo_executor(
        self,
        use_fp8: bool,
        use_ue8m0_scale: bool,
        enable_peo_level: int,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        ep_size: int,
        max_generate_batch_size: int,
        num_peo_rounds: int,
        deep_ep_num_sm: int,
    ) -> None:
        if num_experts % ep_size != 0:
            self.skipTest(
                f"skip invalid ep config: num_experts={num_experts} not divisible by ep_size={ep_size}"
            )
        num_local_experts = num_experts // ep_size
        if enable_peo_level > 0 and (num_local_experts % num_peo_rounds != 0):
            self.skipTest(
                f"skip invalid peo config: num_local_experts={num_local_experts} not divisible by num_peo_rounds={num_peo_rounds}"
            )
        config = make_deepgemm_masked_test_config(
            use_fp8=use_fp8,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            ep_size=ep_size,
            max_generate_batch_size=max_generate_batch_size,
            enable_peo_level=enable_peo_level,
            num_peo_rounds=num_peo_rounds,
            deep_ep_num_sm=deep_ep_num_sm,
        )
        expert_payload, weights = generate_payload_and_weights(config)
        expert_num_tokens = expert_payload.expert_tokens_meta.expert_num_tokens
        ref_output = generate_ref_output(config, expert_payload, weights)

        if use_fp8:
            expert_payload_use_ue8m0 = bool(use_ue8m0_scale or is_deep_gemm_e8m0_used())
            quantize_payload_fp8_per_token_group(
                expert_payload, use_e8m0_scale=expert_payload_use_ue8m0
            )
            quantize_weights_fp8_per_block(weights, use_ue8m0=False)
            assert expert_payload.expert_x_scale is not None
            if use_ue8m0_scale:
                assert is_deep_gemm_e8m0_used()
                assert expert_payload.expert_x_scale.dtype in (torch.int32, torch.int)
                assert expert_payload.expert_x_scale.size(2) == hidden_size // 128 // 4

        executor = DeepGemmMaskedPeoExecutor(config, config.quant_config, weights)
        torch.cuda.synchronize()

        if enable_peo_level == 4:
            expert_payload.dispatch_recv_events = []
        else:
            expert_payload.dispatch_recv_events = make_dispatch_events(
                enable_peo_level, num_peo_rounds
            )

        extra_expert_args: Optional[dict] = None
        if enable_peo_level == 3:
            extra_expert_args = {"comm_stream": torch.cuda.Stream()}

        combine_payload: CombineForwardPayload = executor.execute(
            payload=expert_payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=extra_expert_args,
        )

        assert combine_payload.expert_executions is not None
        assert len(combine_payload.expert_executions) == num_peo_rounds
        for fn in combine_payload.expert_executions:
            fn()
        torch.cuda.synchronize()

        if not use_fp8:
            diff_th = 1e-5
        elif use_ue8m0_scale:
            diff_th = 3e-3
        else:
            diff_th = 2.2e-3
        compare_with_ref_per_expert(
            combine_payload.fused_expert_output, ref_output, expert_num_tokens, diff_th
        )

    def _test_bf16_peo_overlap(self) -> None:
        ran_any = False
        for (
            enable_peo_level,
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
            num_peo_rounds,
            deep_ep_num_sm,
        ) in itertools.product(
            self.PEO_LEVELS,
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
            self.NUM_PEO_ROUNDS,
            self.DEEP_EP_NUM_SM,
        ):
            with self.subTest(
                enable_peo_level=enable_peo_level,
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
                num_peo_rounds=num_peo_rounds,
                deep_ep_num_sm=deep_ep_num_sm,
            ):
                self._test_deepgemm_masked_peo_executor(
                    use_fp8=False,
                    use_ue8m0_scale=False,
                    enable_peo_level=enable_peo_level,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid bf16 peo_overlap cases to run on this machine/config"
            )

    def _test_fp8_per_block_peo_overlap(self) -> None:
        ran_any = False
        for (
            enable_peo_level,
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
            num_peo_rounds,
            deep_ep_num_sm,
        ) in itertools.product(
            self.PEO_LEVELS,
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
            self.NUM_PEO_ROUNDS,
            self.DEEP_EP_NUM_SM,
        ):
            with self.subTest(
                enable_peo_level=enable_peo_level,
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
                num_peo_rounds=num_peo_rounds,
                deep_ep_num_sm=deep_ep_num_sm,
            ):
                self._test_deepgemm_masked_peo_executor(
                    use_fp8=True,
                    use_ue8m0_scale=False,
                    enable_peo_level=enable_peo_level,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block peo_overlap cases to run on this machine/config"
            )

    def _test_fp8_per_block_ue8m0_peo_overlap(self) -> None:
        if not is_deep_gemm_e8m0_used():
            self.skipTest("UE8M0 scale format only used on SM100/SM120")
        ran_any = False
        for (
            enable_peo_level,
            hidden_size,
            moe_inter_size,
            num_experts,
            ep_size,
            max_generate_batch_size,
            num_peo_rounds,
            deep_ep_num_sm,
        ) in itertools.product(
            self.PEO_LEVELS,
            self.HIDDEN_SIZES,
            self.MOE_INTERMEDIATE_SIZES,
            self.NUM_EXPERTS,
            self.EP_SIZES,
            self.MAX_GENERATE_BATCH_SIZES,
            self.NUM_PEO_ROUNDS,
            self.DEEP_EP_NUM_SM,
        ):
            with self.subTest(
                enable_peo_level=enable_peo_level,
                hidden_size=hidden_size,
                moe_inter_size=moe_inter_size,
                num_experts=num_experts,
                ep_size=ep_size,
                max_generate_batch_size=max_generate_batch_size,
                num_peo_rounds=num_peo_rounds,
                deep_ep_num_sm=deep_ep_num_sm,
            ):
                if hidden_size % 512 != 0:
                    self.skipTest(
                        f"skip UE8M0: hidden_size={hidden_size} not divisible by 512"
                    )
                self._test_deepgemm_masked_peo_executor(
                    use_fp8=True,
                    use_ue8m0_scale=True,
                    enable_peo_level=enable_peo_level,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block UE8M0 peo_overlap cases to run on this machine/config"
            )


class DeepGemmMaskedPeoExecutorTest(
    DeepGemmMaskedPeoExecutorTestBase, unittest.TestCase
):

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

    def test_bf16_peo_overlap(self) -> None:
        self._test_bf16_peo_overlap()

    def test_fp8_per_block_peo_overlap(self) -> None:
        self._test_fp8_per_block_peo_overlap()

    def test_fp8_per_block_ue8m0_peo_overlap(self) -> None:
        self._test_fp8_per_block_ue8m0_peo_overlap()


if __name__ == "__main__":
    unittest.main()
