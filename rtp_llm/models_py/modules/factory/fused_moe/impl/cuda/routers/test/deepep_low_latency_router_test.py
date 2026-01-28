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
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
    DeepEpLowLatencyRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.test.fused_moe_router_test_util import (
    build_parallelism_config,
    build_quant_config,
    build_reference_recv,
    create_test_data,
    dequant_per_expert,
    set_start_method,
    setup_low_latency_env,
)
from rtp_llm.ops import MoeConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.test.utils.port_util import PortsContext


class DeepEPLowLatencyRouterTestBase:
    NUM_PROCESSES = [2]
    HIDDEN_SIZES = [6144]
    NUM_TOPK = [8]
    NUM_EXPERTS = [160]
    MAX_GENERATE_BATCH_SIZE = [127]
    TP_SIZE = [2]
    ENABLE_COMM_OVERLAP = [False]

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
        use_nvlink_for_low_latency_mode: bool,
        quant_mode: str,
        enable_comm_overlap: bool,
    ) -> Tuple[MoEConfigAdapter, DeepEpLowLatencyRouter, FusedMoEQuantConfig, int]:
        parallelism_config, dp_size = build_parallelism_config(
            rank=rank, world_size=world_size, tp_size=tp_size, nccl_port=nccl_port
        )

        # Env
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
        setup_low_latency_env(use_nvlink_for_low_latency_mode)

        # Device
        torch.cuda.set_device(parallelism_config.local_rank)
        torch.set_default_device(f"cuda:{parallelism_config.local_rank}")

        # Init distributed
        init_distributed_environment(
            parallelism_config=parallelism_config, backend="nccl", timeout=60
        )

        # Model / MoE config
        model_config = ModelConfig()
        model_config.hidden_size = hidden_size
        model_config.expert_num = num_experts
        model_config.moe_k = num_topk

        moe_config = MoeConfig()
        moe_config.use_deepep_moe = True
        moe_config.use_deepep_low_latency = True
        moe_config.use_deepep_internode = False
        moe_config.enable_peo_level = 0
        # Keep default values for completeness (unused when PEO disabled)
        moe_config.num_peo_rounds = 2
        moe_config.deep_ep_num_sm = 24

        quant_config = build_quant_config(quant_mode)
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            enable_comm_overlap=enable_comm_overlap,
        )

        router = DeepEpLowLatencyRouter(config, quant_config)
        return config, router, quant_config, dp_size

    @staticmethod
    def _test_deepep_low_latency_router(
        rank: int,
        world_size: int,
        tp_size: int,
        nccl_port: int,
        hidden_size: int,
        num_experts: int,
        num_topk: int,
        max_generate_batch_size: int,
        use_nvlink_for_low_latency_mode: bool,
        quant_mode: str,
        enable_comm_overlap: bool,
    ) -> None:
        config, router, quant_config, dp_size = DeepEPLowLatencyRouterTest._init_router(
            rank=rank,
            world_size=world_size,
            tp_size=tp_size,
            nccl_port=nccl_port,
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_topk=num_topk,
            max_generate_batch_size=max_generate_batch_size,
            use_nvlink_for_low_latency_mode=use_nvlink_for_low_latency_mode,
            quant_mode=quant_mode,
            enable_comm_overlap=enable_comm_overlap,
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

            # Reference: what this ep_rank should receive for its local experts.
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

            # Router prepare (dispatch)
            payload = router.prepare(
                hidden_states[dp_rank],
                None,
                None,
                topk_weights[dp_rank],
                topk_ids[dp_rank],
            )

            # Dequant if needed.
            if quant_config.is_quantized:
                assert payload.expert_x_scale is not None
                # Align scale tensor memory layout before view/cast in per_token_cast_back,
                # similar to DeepEP tests (see deepep_test.py packed_recv_x handling).
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

            # Permute recv_x by recv_src_info to match ref ordering (same approach as old tests).
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

            # Validate payload data:
            # - fp8_per_block: only validate the last 128 dims (one FP8 block)
            # - fp8_per_token: validate the whole hidden vector (per-token scale depends on all dims)
            # - bf16: validate the last 128 dims (stable + enough for correctness)
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
            if not torch.equal(ref_recv_count, got_recv_count):
                diff_cnt = (
                    ref_recv_count.to(torch.int64) - got_recv_count.to(torch.int64)
                ).abs()
                num_mismatch = int((diff_cnt != 0).sum().item())
                max_abs = int(diff_cnt.max().item()) if diff_cnt.numel() else 0
                # Show a small sample for debugging.
                mismatch_idx = (
                    (diff_cnt != 0).nonzero(as_tuple=False).flatten()[:8].tolist()
                )
                raise AssertionError(
                    "expert_num_tokens mismatch: "
                    f"num_mismatch={num_mismatch}, max_abs={max_abs}, "
                    f"sample_idx={mismatch_idx}, "
                    f"ref={ref_recv_count[mismatch_idx].tolist() if mismatch_idx else []}, "
                    f"got={got_recv_count[mismatch_idx].tolist() if mismatch_idx else []}"
                )

            # Router finalize (combine)
            extra_finalize_args = {"original_num_tokens": max_generate_batch_size}
            combine_payload = CombineForwardPayload(fused_expert_output=recv_x)

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

    def _test_bf16(self):
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
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
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
                        DeepEPLowLatencyRouterTestBase._test_deepep_low_latency_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            use_nvlink_for_low_latency_mode,
                            "bf16",
                            enable_comm_overlap,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest("No valid bf16 normal cases to run on this machine/config")

    def _test_fp8_per_block(self):
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
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
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
                        DeepEPLowLatencyRouterTestBase._test_deepep_low_latency_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            use_nvlink_for_low_latency_mode,
                            "fp8_per_block",
                            enable_comm_overlap,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_block normal cases to run on this machine/config"
            )

    def _test_fp8_per_token(self):
        set_start_method()
        ran_any = False
        # Per-token quantization requires ACCL-EP; if not supported, skip fast.
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
                use_nvlink_for_low_latency_mode,
            ) in itertools.product(
                self.NUM_PROCESSES,
                self.HIDDEN_SIZES,
                self.NUM_TOPK,
                self.NUM_EXPERTS,
                self.TP_SIZE,
                self.MAX_GENERATE_BATCH_SIZE,
                self.ENABLE_COMM_OVERLAP,
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
                        DeepEPLowLatencyRouterTestBase._test_deepep_low_latency_router,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            use_nvlink_for_low_latency_mode,
                            "fp8_per_token",
                            enable_comm_overlap,
                        ),
                        nprocs=world_size,
                        join=True,
                    )
                ran_any = True
        if not ran_any:
            self.skipTest(
                "No valid fp8_per_token normal cases to run on this machine/config"
            )


class DeepEPLowLatencyRouterTest(DeepEPLowLatencyRouterTestBase, unittest.TestCase):

    def test_bf16(self):
        self._test_bf16()

    def test_fp8_per_block(self):
        self._test_fp8_per_block()

    def test_fp8_per_token(self):
        self._test_fp8_per_token()


if __name__ == "__main__":
    unittest.main()
