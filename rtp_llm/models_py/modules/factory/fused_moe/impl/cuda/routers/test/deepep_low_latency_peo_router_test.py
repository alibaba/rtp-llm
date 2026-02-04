import itertools
import os
import unittest
from typing import Tuple

import torch
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_peo_router import (
    DeepEpLowLatencyPeoRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.test.fused_moe_router_test_util import (
    build_parallelism_config,
    build_quant_config,
    build_reference_recv,
    create_test_data,
    dequant_per_expert,
    set_start_method,
    setup_low_latency_env,
    wait_dispatch_events,
)
from rtp_llm.ops import MoeConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.test.utils.port_util import PortsContext


class DeepEPLowLatencyPeoRouterTestBase:
    NUM_PROCESSES = [2]
    HIDDEN_SIZES = [6144]
    NUM_TOPK = [8]
    NUM_EXPERTS = [160]
    MAX_GENERATE_BATCH_SIZE = [127]
    TP_SIZE = [2]
    ENABLE_COMM_OVERLAP = [False]

    PEO_LEVELS = [3]
    NUM_PEO_ROUNDS = [4]
    DEEP_EP_NUM_SM = [24]

    USE_NVLINK_FOR_LOW_LATENCY_MODE = [True]

    @staticmethod
    def _init_router(
        rank: int,
        world_size: int,
        tp_size: int,
        nccl_port: int,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        max_generate_batch_size: int,
        enable_peo_level: int,
        num_peo_rounds: int,
        deep_ep_num_sm: int,
        use_nvlink_for_low_latency_mode: bool,
        quant_mode: str,
        enable_comm_overlap: bool,
    ) -> Tuple[MoEConfigAdapter, DeepEpLowLatencyPeoRouter, FusedMoEQuantConfig, int]:
        parallelism_config, dp_size = build_parallelism_config(
            rank=rank, world_size=world_size, tp_size=tp_size, nccl_port=nccl_port
        )

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
        setup_low_latency_env(use_nvlink_for_low_latency_mode)

        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")

        init_distributed_environment(
            parallelism_config=parallelism_config, backend="nccl", timeout=60
        )

        model_config = ModelConfig()
        model_config.hidden_size = hidden_size
        model_config.expert_num = num_experts
        model_config.moe_k = num_topk

        moe_config = MoeConfig()
        moe_config.use_deepep_moe = True
        moe_config.use_deepep_low_latency = True
        moe_config.use_deepep_internode = False
        moe_config.enable_peo_level = enable_peo_level
        moe_config.num_peo_rounds = num_peo_rounds
        moe_config.deep_ep_num_sm = deep_ep_num_sm

        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=max_generate_batch_size,
            enable_comm_overlap=enable_comm_overlap,
        )

        quant_config = build_quant_config(quant_mode)
        router = DeepEpLowLatencyPeoRouter(config, quant_config)
        return config, router, quant_config, dp_size

    @staticmethod
    def _test_deepep_low_latency_peo_router(
        rank: int,
        world_size: int,
        tp_size: int,
        nccl_port: int,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        max_generate_batch_size: int,
        enable_peo_level: int,
        num_peo_rounds: int,
        deep_ep_num_sm: int,
        use_nvlink_for_low_latency_mode: bool,
        quant_mode: str,
        enable_comm_overlap: bool,
    ) -> None:
        config, router, quant_config, dp_size = (
            DeepEPLowLatencyPeoRouterTestBase._init_router(
                rank=rank,
                world_size=world_size,
                tp_size=tp_size,
                nccl_port=nccl_port,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_topk=num_topk,
                max_generate_batch_size=max_generate_batch_size,
                enable_peo_level=enable_peo_level,
                num_peo_rounds=num_peo_rounds,
                deep_ep_num_sm=deep_ep_num_sm,
                use_nvlink_for_low_latency_mode=use_nvlink_for_low_latency_mode,
                quant_mode=quant_mode,
                enable_comm_overlap=enable_comm_overlap,
            )
        )

        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            ep_size = config.ep_size
            ep_rank = config.ep_rank
            dp_rank = config.dp_rank

            hidden_states, topk_ids, topk_weights = create_test_data(
                dp_size=dp_size,
                max_generate_batch_size=max_generate_batch_size,
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_topk=num_topk,
                quant_mode=quant_mode,
            )

            num_local_experts = num_experts // ep_size
            num_max_tokens = router.deepep_wrapper.ll_num_max_token_per_rank * ep_size
            ref_recv_x, ref_recv_count = build_reference_recv(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                ep_rank=ep_rank,
                ep_size=ep_size,
                num_experts=num_experts,
                num_max_tokens=num_max_tokens,
            )

            payload = router.prepare(
                hidden_states[dp_rank],
                None,
                None,
                topk_weights[dp_rank],
                topk_ids[dp_rank],
            )

            if enable_peo_level in (1, 2, 3):
                assert payload.dispatch_recv_events is not None
                assert len(payload.dispatch_recv_events) == num_peo_rounds
            elif enable_peo_level == 4:
                assert payload.dispatch_recv_events is None

            wait_dispatch_events(payload)

            if quant_config.is_quantized:
                assert payload.expert_x_scale is not None
                expert_x_scale = (
                    payload.expert_x_scale
                    if payload.expert_x_scale.is_contiguous()
                    else payload.expert_x_scale.contiguous()
                )
                recv_x = dequant_per_expert(
                    payload.expert_x,
                    expert_x_scale,
                    per_token_quant=quant_config.is_per_act_token,
                )
            else:
                recv_x = payload.expert_x

            int_mask = (2**32) - 1
            permuted_recv_x = torch.zeros_like(recv_x)
            for local_expert_id in range(num_local_experts):
                current_start_idx = 0
                recv_src_info = router.handle[0][local_expert_id]
                recv_layout_range = router.handle[1][local_expert_id]
                for j in range(ep_size):
                    begin_idx = (recv_layout_range[j] >> 32).item()
                    count = (recv_layout_range[j] & int_mask).item()
                    if count == 0:
                        continue
                    sorted_indices_per_rank = torch.argsort(
                        recv_src_info[begin_idx : begin_idx + count]
                    )
                    permuted_recv_x[
                        local_expert_id, current_start_idx : current_start_idx + count
                    ] = recv_x[local_expert_id, begin_idx + sorted_indices_per_rank]
                    current_start_idx += count

            if quant_mode == "fp8_per_token":
                ref_slice = ref_recv_x
                got_slice = permuted_recv_x
                diff_th = 1e-3
            else:
                ref_slice = ref_recv_x[:, :, -128:]
                got_slice = permuted_recv_x[:, :, -128:]
                diff_th = 1e-3 if quant_mode.startswith("fp8") else 1e-6
            diff = calc_diff(ref_slice, got_slice)
            assert diff < diff_th, f"dispatch diff too large: {diff} >= {diff_th}"

            got_recv_count = payload.expert_tokens_meta.expert_num_tokens
            assert got_recv_count is not None
            assert torch.equal(ref_recv_count, got_recv_count)

            extra_finalize_args = {"original_num_tokens": max_generate_batch_size}
            expert_executions = [lambda: None for _ in range(num_peo_rounds)]
            combine_payload = CombineForwardPayload(
                fused_expert_output=recv_x, expert_executions=expert_executions
            )
            combined_x = router.finalize(
                combine_payload,
                payload.expert_topk_weights,
                payload.expert_topk_ids,
                False,
                extra_finalize_args,
            )

            if quant_mode == "fp8_per_token":
                ref_out = hidden_states[dp_rank]
                got_out = combined_x
                out_th = 1e-3
            else:
                ref_out = hidden_states[dp_rank, :, -128:]
                got_out = combined_x[:, -128:]
                out_th = 1e-3 if quant_mode.startswith("fp8") else 1e-6
            out_diff = calc_diff(ref_out, got_out)
            assert out_diff < out_th, f"combine diff too large: {out_diff} >= {out_th}"
        finally:
            try:
                del router
            except Exception:
                pass
            DeepEPWrapper.reset()
            destroy_distributed_environment()

    def _test_bf16_peo_overlap(self):
        set_start_method()
        ran_any = False
        with PortsContext(None, 1) as ports:
            nccl_port = int(ports[0])
            for (
                world_size,
                hidden_size,
                num_topk,
                num_experts,
                tp_size,
                max_generate_batch_size,
                enable_comm_overlap,
                enable_peo_level,
                num_peo_rounds,
                deep_ep_num_sm,
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
                self.PEO_LEVELS,
                self.NUM_PEO_ROUNDS,
                self.DEEP_EP_NUM_SM,
                self.USE_NVLINK_FOR_LOW_LATENCY_MODE,
            ):
                with self.subTest(
                    world_size=world_size,
                    hidden_size=hidden_size,
                    num_topk=num_topk,
                    num_experts=num_experts,
                    tp_size=tp_size,
                    max_generate_batch_size=max_generate_batch_size,
                    enable_comm_overlap=enable_comm_overlap,
                    enable_peo_level=enable_peo_level,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                    use_nvlink_for_low_latency_mode=use_nvlink_for_low_latency_mode,
                ):
                    if world_size % tp_size != 0:
                        self.skipTest(
                            f"skip invalid combo: world_size={world_size} not divisible by tp_size={tp_size}"
                        )
                    if num_topk > num_experts:
                        self.skipTest(
                            f"skip invalid combo: num_topk={num_topk} > num_experts={num_experts}"
                        )
                    if num_experts % world_size != 0:
                        self.skipTest(
                            f"skip invalid combo: num_experts={num_experts} not divisible by world_size={world_size}"
                        )
                    mp.spawn(
                        DeepEPLowLatencyPeoRouterTestBase._test_deepep_low_latency_peo_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            enable_peo_level,
                            num_peo_rounds,
                            deep_ep_num_sm,
                            use_nvlink_for_low_latency_mode,
                            "bf16",
                            True,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid bf16 peo_overlap cases to run on this machine/config"
            )

    def _test_fp8_per_block_peo_overlap(self):
        set_start_method()
        ran_any = False
        with PortsContext(None, 1) as ports:
            nccl_port = int(ports[0])
            for (
                world_size,
                hidden_size,
                num_topk,
                num_experts,
                tp_size,
                max_generate_batch_size,
                enable_comm_overlap,
                enable_peo_level,
                num_peo_rounds,
                deep_ep_num_sm,
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
                self.PEO_LEVELS,
                self.NUM_PEO_ROUNDS,
                self.DEEP_EP_NUM_SM,
                self.USE_NVLINK_FOR_LOW_LATENCY_MODE,
            ):
                with self.subTest(
                    world_size=world_size,
                    hidden_size=hidden_size,
                    num_topk=num_topk,
                    num_experts=num_experts,
                    tp_size=tp_size,
                    max_generate_batch_size=max_generate_batch_size,
                    enable_comm_overlap=enable_comm_overlap,
                    enable_peo_level=enable_peo_level,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                    use_nvlink_for_low_latency_mode=use_nvlink_for_low_latency_mode,
                ):
                    if world_size % tp_size != 0:
                        self.skipTest(
                            f"skip invalid combo: world_size={world_size} not divisible by tp_size={tp_size}"
                        )
                    if num_topk > num_experts:
                        self.skipTest(
                            f"skip invalid combo: num_topk={num_topk} > num_experts={num_experts}"
                        )
                    if num_experts % world_size != 0:
                        self.skipTest(
                            f"skip invalid combo: num_experts={num_experts} not divisible by world_size={world_size}"
                        )
                    mp.spawn(
                        DeepEPLowLatencyPeoRouterTestBase._test_deepep_low_latency_peo_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            enable_peo_level,
                            num_peo_rounds,
                            deep_ep_num_sm,
                            use_nvlink_for_low_latency_mode,
                            "fp8_per_block",
                            True,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block peo_overlap cases to run on this machine/config"
            )

    def _test_fp8_per_token_peo_overlap(self):
        set_start_method()
        ran_any = False
        if not DeepEPWrapper.supported():
            self.skipTest("DeepEP not supported")
        with PortsContext(None, 1) as ports:
            nccl_port = int(ports[0])
            for (
                world_size,
                hidden_size,
                num_topk,
                num_experts,
                tp_size,
                max_generate_batch_size,
                enable_comm_overlap,
                enable_peo_level,
                num_peo_rounds,
                deep_ep_num_sm,
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
                self.PEO_LEVELS,
                self.NUM_PEO_ROUNDS,
                self.DEEP_EP_NUM_SM,
                self.USE_NVLINK_FOR_LOW_LATENCY_MODE,
            ):
                with self.subTest(
                    world_size=world_size,
                    hidden_size=hidden_size,
                    num_topk=num_topk,
                    num_experts=num_experts,
                    tp_size=tp_size,
                    max_generate_batch_size=max_generate_batch_size,
                    enable_comm_overlap=enable_comm_overlap,
                    enable_peo_level=enable_peo_level,
                    num_peo_rounds=num_peo_rounds,
                    deep_ep_num_sm=deep_ep_num_sm,
                    use_nvlink_for_low_latency_mode=use_nvlink_for_low_latency_mode,
                ):
                    if world_size % tp_size != 0:
                        self.skipTest(
                            f"skip invalid combo: world_size={world_size} not divisible by tp_size={tp_size}"
                        )
                    if num_topk > num_experts:
                        self.skipTest(
                            f"skip invalid combo: num_topk={num_topk} > num_experts={num_experts}"
                        )
                    if num_experts % world_size != 0:
                        self.skipTest(
                            f"skip invalid combo: num_experts={num_experts} not divisible by world_size={world_size}"
                        )
                    mp.spawn(
                        DeepEPLowLatencyPeoRouterTestBase._test_deepep_low_latency_peo_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            enable_peo_level,
                            num_peo_rounds,
                            deep_ep_num_sm,
                            use_nvlink_for_low_latency_mode,
                            "fp8_per_token",
                            True,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_token peo_overlap cases to run on this machine/config"
            )


class DeepEPLowLatencyPeoRouterTest(
    DeepEPLowLatencyPeoRouterTestBase, unittest.TestCase
):

    def test_bf16_peo_overlap(self):
        self._test_bf16_peo_overlap()

    def test_fp8_per_block_peo_overlap(self):
        self._test_fp8_per_block_peo_overlap()

    def test_fp8_per_token_peo_overlap(self):
        self._test_fp8_per_token_peo_overlap()


if __name__ == "__main__":
    unittest.main()
