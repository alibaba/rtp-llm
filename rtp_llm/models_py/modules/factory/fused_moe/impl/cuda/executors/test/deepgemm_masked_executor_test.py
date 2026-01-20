# type: ignore
import itertools
import random
from typing import Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
    DeepGemmMaskedExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.fused_moe_executor_test_util import (
    generate_payload_and_weights,
    generate_ref_output,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


def _make_config(
    use_fp8: bool,
    hidden_size: int,
    moe_intermediate_size: int,
    num_experts: int,
    ep_size: int,
    max_generate_batch_size: int,
    enable_peo_level: int,
    num_peo_rounds: int,
    deep_ep_num_sm: int,
) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.hidden_size = hidden_size
    model_config.moe_inter_size = moe_intermediate_size
    model_config.expert_num = num_experts
    model_config.moe_k = 8

    parallelism_config = ParallelismConfig()
    parallelism_config.world_size = ep_size
    parallelism_config.world_rank = 0
    parallelism_config.local_world_size = ep_size
    parallelism_config.local_rank = 0
    parallelism_config.dp_size = ep_size
    parallelism_config.dp_rank = 0
    parallelism_config.tp_size = 1
    parallelism_config.tp_rank = 0
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = 0

    moe_config = MoeConfig()
    moe_config.use_deepep_moe = False
    moe_config.use_deepep_internode = False
    moe_config.use_deepep_low_latency = True
    moe_config.enable_peo_level = enable_peo_level
    moe_config.num_peo_rounds = num_peo_rounds
    moe_config.deep_ep_num_sm = deep_ep_num_sm

    quant_config = (
        FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
        if use_fp8
        else FusedMoEQuantConfig(quant_dtype=None)
    )

    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        max_generate_batch_size=max_generate_batch_size,
        quant_config=quant_config,
        enable_comm_overlap=False,
    )


def _make_dispatch_events(
    enable_peo_level: int, num_peo_rounds: int
) -> List[torch.cuda.Event]:
    # For overlap_4, executor ignores dispatch events; others need a list of events.
    if enable_peo_level == 4:
        return []
    events: List[torch.cuda.Event] = []
    for _ in range(num_peo_rounds):
        ev = torch.cuda.Event()
        torch.cuda.current_stream().record_event(ev)
        events.append(ev)
    return events


def _quantize_payload_fp8(
    expert_payload: ExpertForwardPayload,
    use_e8m0_scale: bool,
) -> None:
    # NOTE:
    # `sgl_per_token_group_quant_fp8` expects a 2D tensor; passing a 3D [E, M, K]
    # tensor will hit `create_per_token_group_quant_fp8_output_scale(...).permute(-1, -2)`
    # which only works for 2D tensors.
    x_bf16 = expert_payload.expert_x.contiguous()
    assert (
        x_bf16.dim() == 3
    ), f"expected expert_x to be [E, M, K], got {tuple(x_bf16.shape)}"
    E, M, K = x_bf16.shape
    assert K % 128 == 0

    x_fp8 = torch.empty((E, M, K), device=x_bf16.device, dtype=torch.float8_e4m3fn)
    if use_e8m0_scale:
        assert (K // 128) % 4 == 0, f"UE8M0 requires K/128 divisible by 4, got K={K}"
        x_scale = torch.empty(
            (E, M, K // 128 // 4), device=x_bf16.device, dtype=torch.int32
        )
    else:
        x_scale = torch.empty(
            (E, M, K // 128), device=x_bf16.device, dtype=torch.float32
        )

    for i in range(E):
        q, s = sgl_per_token_group_quant_fp8(
            x_bf16[i],
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=use_e8m0_scale,
        )
        x_fp8[i] = q
        x_scale[i].copy_(s)

    expert_payload.expert_x = x_fp8
    expert_payload.expert_x_scale = x_scale


def _quantize_weights_fp8(weights: Dict[str, torch.Tensor], use_ue8m0: bool) -> None:
    w1 = weights[W.moe_w1]
    w2 = weights[W.moe_w2]
    E, N, K = w1.shape
    assert w2.shape[0] == E and w2.shape[1] == K and w2.shape[2] == N // 2
    assert N % 128 == 0 and K % 128 == 0 and (N // 2) % 128 == 0

    w1_fp8 = torch.empty((E, N, K), device="cuda", dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty((E, K, N // 2), device="cuda", dtype=torch.float8_e4m3fn)
    s1 = torch.empty((E, N // 128, K // 128), device="cuda", dtype=torch.float32)
    s2 = torch.empty((E, K // 128, (N // 2) // 128), device="cuda", dtype=torch.float32)

    for i in range(E):
        w1_fp8[i], s1[i] = per_block_cast_to_fp8(w1[i], use_ue8m0=use_ue8m0)
        w2_fp8[i], s2[i] = per_block_cast_to_fp8(w2[i], use_ue8m0=use_ue8m0)

    weights[W.moe_w1] = w1_fp8
    weights[W.moe_w2] = w2_fp8
    weights[W.moe_s1] = s1
    weights[W.moe_s2] = s2


def _compare_with_ref(
    output: torch.Tensor,
    ref_output: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    diff_th: float,
) -> None:
    for i, num_token in enumerate(expert_num_tokens.tolist()):
        num_token = int(num_token)
        if num_token <= 0:
            continue
        diff = calc_diff(output[i, :num_token], ref_output[i, :num_token])
        assert (
            diff < diff_th
        ), f"diff too large at expert {i}: {diff} >= {diff_th}, output={output[i, :num_token]}, ref_output={ref_output[i, :num_token]}"


class DeepGemmMaskedExecutorTest(TestCase):
    # - hidden_size must be divisible by 128
    # - UE8M0 packed scale requires hidden_size divisible by 512 (K/128 divisible by 4)
    HIDDEN_SIZES = [6144]
    MOE_INTERMEDIATE_SIZES = [2560]
    NUM_EXPERTS = [160]
    EP_SIZES = [2]
    MAX_GENERATE_BATCH_SIZES = [1, 64, 128]
    NUM_PEO_ROUNDS = [2, 4]
    DEEP_EP_NUM_SM = [24]
    PEO_LEVELS = [2, 3, 4]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        if not has_deep_gemm():
            raise SkipTest("deep_gemm is not available")
        random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.set_device(0)
        torch.set_default_device("cuda")

    def _run_one_case(
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
        config = _make_config(
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
            _quantize_payload_fp8(
                expert_payload, use_e8m0_scale=expert_payload_use_ue8m0
            )
            _quantize_weights_fp8(weights, use_ue8m0=False)
            assert expert_payload.expert_x_scale is not None
            if use_ue8m0_scale:
                assert is_deep_gemm_e8m0_used()
                assert expert_payload.expert_x_scale.dtype in (torch.int32, torch.int)
                assert expert_payload.expert_x_scale.size(2) == hidden_size // 128 // 4

        executor = DeepGemmMaskedExecutor(config, config.quant_config, weights)
        torch.cuda.synchronize()

        if enable_peo_level > 0:
            if enable_peo_level == 4:
                expert_payload.dispatch_recv_events = []
            else:
                expert_payload.dispatch_recv_events = _make_dispatch_events(
                    enable_peo_level, num_peo_rounds
                )
        else:
            expert_payload.dispatch_recv_events = None

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

        if enable_peo_level > 0:
            assert combine_payload.expert_executions is not None
            assert len(combine_payload.expert_executions) == num_peo_rounds
            for fn in combine_payload.expert_executions:
                fn()
        else:
            assert combine_payload.expert_executions is None
        torch.cuda.synchronize()

        if not use_fp8:
            diff_th = 1e-5
        elif use_ue8m0_scale:
            diff_th = 3e-3
        else:
            diff_th = 2.2e-3
        _compare_with_ref(
            combine_payload.fused_expert_output, ref_output, expert_num_tokens, diff_th
        )

    def test_bf16_normal(self) -> None:
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
                self._run_one_case(
                    use_fp8=False,
                    use_ue8m0_scale=False,
                    enable_peo_level=0,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=self.NUM_PEO_ROUNDS[0],
                    deep_ep_num_sm=self.DEEP_EP_NUM_SM[0],
                )
            ran_any = True
        if not ran_any:
            self.skipTest("No valid bf16 normal cases to run on this machine/config")

    def test_fp8_per_block_normal(self) -> None:
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
                self._run_one_case(
                    use_fp8=True,
                    use_ue8m0_scale=False,
                    enable_peo_level=0,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=self.NUM_PEO_ROUNDS[0],
                    deep_ep_num_sm=self.DEEP_EP_NUM_SM[0],
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block normal cases to run on this machine/config"
            )

    def test_fp8_per_block_ue8m0_normal(self) -> None:
        if not is_deep_gemm_e8m0_used():
            self.skipTest("UE8M0 scale format only used on SM100/SM120")
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
                self._run_one_case(
                    use_fp8=True,
                    use_ue8m0_scale=True,
                    enable_peo_level=0,
                    hidden_size=hidden_size,
                    moe_intermediate_size=moe_inter_size,
                    num_experts=num_experts,
                    ep_size=ep_size,
                    max_generate_batch_size=max_generate_batch_size,
                    num_peo_rounds=self.NUM_PEO_ROUNDS[0],
                    deep_ep_num_sm=self.DEEP_EP_NUM_SM[0],
                )
            ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block UE8M0 normal cases to run on this machine/config"
            )

    def test_bf16_peo_overlap(self) -> None:
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
                self._run_one_case(
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

    def test_fp8_per_block_peo_overlap(self) -> None:
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
                self._run_one_case(
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

    def test_fp8_per_block_ue8m0_peo_overlap(self) -> None:
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
                self._run_one_case(
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


if __name__ == "__main__":
    main()
