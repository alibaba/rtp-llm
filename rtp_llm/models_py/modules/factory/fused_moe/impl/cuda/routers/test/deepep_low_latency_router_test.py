# type: ignore
import itertools
import os
from typing import Tuple
from unittest import TestCase, main

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
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.device_resource import has_rdma_device
from rtp_llm.test.utils.numeric_util import calc_diff, per_token_cast_back
from rtp_llm.test.utils.port_util import PortsContext


def _set_start_method() -> None:
    # Unit tests may be run under different runners; avoid crashing if already set.
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def _setup_low_latency_env(use_nvlink_for_low_latency_mode: bool) -> None:
    """
    Follow DeepEP tests' convention: if user didn't set ACCL_LOW_LATENCY_OPTIMIZE,
    set it based on whether we want NVLink-only mode or allow RDMA mode.
    """
    if use_nvlink_for_low_latency_mode:
        os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "1")
    else:
        os.environ["ACCL_TOPO_FIX"] = "1"
        os.environ["ACCL_LOAD_BALANCE"] = "1"
        os.environ["NVSHMEM_IB_GID_INDEX"] = "3"
        os.environ.setdefault("ACCL_LOW_LATENCY_OPTIMIZE", "0")
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"


def _build_quant_config(mode: str) -> FusedMoEQuantConfig:
    if mode == "bf16":
        return FusedMoEQuantConfig(quant_dtype=None)
    if mode == "fp8_per_block":
        return FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128],
        )
    if mode == "fp8_per_token":
        # Match DeepEP per-token-quant test setup.
        os.environ["ACCL_FP8_CAST_LEVEL"] = "2"
        return FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
        )
    raise ValueError(f"unknown quant mode: {mode}")


def _dequant_per_expert(
    expert_x: torch.Tensor,
    expert_x_scale: torch.Tensor,
    per_token_quant: bool,
) -> torch.Tensor:
    # expert_x: [E_local, M, K]
    # expert_x_scale:
    # - fp8_per_block: [E_local, M, K/128]
    # - fp8_per_token: [E_local, M, 1]
    out = torch.empty(expert_x.size(), dtype=torch.bfloat16, device=expert_x.device)
    for i in range(expert_x.size(0)):
        out[i] = per_token_cast_back(
            expert_x[i],
            expert_x_scale[i],
            pertoken_quant=per_token_quant,
        )
    return out


def _wait_dispatch_events(payload) -> None:
    # PEO overlap mode returns events that guard when recv buffers become visible.
    evs = getattr(payload, "dispatch_recv_events", None)
    if not evs:
        return
    for ev in evs:
        torch.cuda.current_stream().wait_event(ev)
    # Make sure we can safely read expert_x/expert_x_scale.
    torch.cuda.synchronize()


def _build_parallelism_config(
    rank: int,
    world_size: int,
    tp_size: int,
    nccl_port: int,
) -> Tuple[ParallelismConfig, int]:
    dp_size = world_size // tp_size
    ep_size = world_size

    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = tp_size
    parallelism_config.tp_rank = rank % tp_size
    parallelism_config.dp_size = dp_size
    parallelism_config.dp_rank = rank // tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = rank % ep_size
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size
    parallelism_config.nccl_ip = "127.0.0.1"
    parallelism_config.th_nccl_port = nccl_port
    return parallelism_config, dp_size


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
) -> Tuple[MoEConfigAdapter, DeepEpLowLatencyRouter, FusedMoEQuantConfig, int]:
    parallelism_config, dp_size = _build_parallelism_config(
        rank=rank, world_size=world_size, tp_size=tp_size, nccl_port=nccl_port
    )

    # Env
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    _setup_low_latency_env(use_nvlink_for_low_latency_mode)

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

    quant_config = _build_quant_config(quant_mode)
    router = DeepEpLowLatencyRouter(config, quant_config)
    return config, router, quant_config, dp_size


def _create_test_data(
    dp_size: int,
    max_generate_batch_size: int,
    hidden_size: int,
    num_experts: int,
    num_topk: int,
    quant_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create deterministic test data across all ranks.

    Quant-error control:
    - fp8_per_block: only require the LAST 128 elements to be identical (one fp8 block),
      other dimensions can vary and are not validated.
    - fp8_per_token: the whole hidden_size has identical values (stronger constraint),
      because per-token scales depend on all hidden dims.
    """
    # Keep values in a small range to reduce FP8 quantization error. The scale factor
    # does not need to match block size (128); it's purely for numeric stability.
    token_vals = (
        torch.arange(
            dp_size * max_generate_batch_size, device="cuda", dtype=torch.float32
        ).view(dp_size, max_generate_batch_size, 1)
        * 0.01
    )

    if quant_mode == "fp8_per_token":
        # Entire hidden vector shares one value -> smaller dynamic range for per-token quant.
        hidden_states = token_vals.repeat(1, 1, hidden_size).to(torch.bfloat16)
    else:
        # bf16 / fp8_per_block: allow other dims to vary; only enforce LAST 128 dims identical.
        assert (
            hidden_size >= 128
        ), f"hidden_size must be >= 128 to validate tail block, got {hidden_size}"
        hidden_states = (
            torch.randn(
                (dp_size, max_generate_batch_size, hidden_size),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.01
        )
        hidden_states[:, :, -128:] = token_vals.repeat(1, 1, 128).to(torch.bfloat16)

    topk_ids = torch.rand(
        (dp_size, max_generate_batch_size, num_experts), device="cuda"
    ).topk(num_topk, dim=-1, largest=True)[1]
    topk_weights = (
        torch.ones((dp_size, max_generate_batch_size, num_topk), device="cuda").to(
            torch.float32
        )
        / num_topk
    )
    return hidden_states, topk_ids, topk_weights


def _build_reference_recv(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    num_experts: int,
    num_max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build reference expert recv buffers for this ep_rank."""
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    hidden_size = hidden_states.size(-1)
    ref_recv_x = torch.zeros(
        (num_local_experts, num_max_tokens, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ref_recv_count = torch.zeros((num_local_experts,), dtype=torch.int32, device="cuda")
    for local_expert_id in range(num_local_experts):
        expert_id = ep_rank * num_local_experts + local_expert_id
        expert_mask = (topk_ids == expert_id).any(dim=-1)  # [dp_size, T]
        num_selected_tokens = expert_mask.sum()
        ref_recv_x[local_expert_id, :num_selected_tokens] = hidden_states[expert_mask]
        ref_recv_count[local_expert_id] = num_selected_tokens
    return ref_recv_x, ref_recv_count


def _run_single_case(
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
    config, router, quant_config, dp_size = _init_router(
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

    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        ep_size = config.ep_size
        ep_rank = config.ep_rank
        dp_rank = config.dp_rank

        hidden_states, topk_ids, topk_weights = _create_test_data(
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
        ref_recv_x, ref_recv_count = _build_reference_recv(
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

        # PEO overlap contracts:
        # - level 1/2/3 should return dispatch_recv_events (one per round)
        # - level 4 uses return_recv_hook=False and does not provide events
        if enable_peo_level in (1, 2, 3):
            assert payload.dispatch_recv_events is not None
            assert len(payload.dispatch_recv_events) == num_peo_rounds
        elif enable_peo_level == 4:
            assert payload.dispatch_recv_events is None

        # PEO overlap: wait all recv events before touching recv buffers.
        _wait_dispatch_events(payload)

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
            recv_x = _dequant_per_expert(
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
        if enable_peo_level > 0:
            # PEO overlap path requires expert_executions list.
            expert_executions = [lambda: None for _ in range(num_peo_rounds)]
            combine_payload = CombineForwardPayload(
                fused_expert_output=recv_x, expert_executions=expert_executions
            )
        else:
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


class DeepEPLowLatencyRouterTest(TestCase):
    NUM_PROCESSES = [2]
    HIDDEN_SIZES = [6144]
    NUM_TOPK = [8]
    NUM_EXPERTS = [160]
    MAX_GENERATE_BATCH_SIZE = [128]
    TP_SIZE = [1, 2]
    ENABLE_COMM_OVERLAP = [True, False]

    PEO_LEVELS = [3]
    NUM_PEO_ROUNDS = [4]
    DEEP_EP_NUM_SM = [24]

    USE_NVLINK_FOR_LOW_LATENCY_MODE = [True, False] if has_rdma_device() else [True]

    def test_bf16_normal(self):
        _set_start_method()
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
                        _run_single_case,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            0,
                            self.NUM_PEO_ROUNDS[0],
                            self.DEEP_EP_NUM_SM[0],
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

    def test_fp8_per_block_normal(self):
        _set_start_method()
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
                        _run_single_case,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            0,
                            self.NUM_PEO_ROUNDS[0],
                            self.DEEP_EP_NUM_SM[0],
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

    def test_fp8_per_token_normal(self):
        _set_start_method()
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
                        _run_single_case,
                        args=(
                            world_size,
                            tp_size,
                            nccl_port,
                            hidden_size,
                            num_experts,
                            num_topk,
                            max_generate_batch_size,
                            0,  # enable_peo_level
                            self.NUM_PEO_ROUNDS[0],
                            self.DEEP_EP_NUM_SM[0],
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

    def test_bf16_peo_overlap(self):
        _set_start_method()
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
                        _run_single_case,
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

    def test_fp8_per_block_peo_overlap(self):
        _set_start_method()
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
                        _run_single_case,
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

    def test_fp8_per_token_peo_overlap(self):
        _set_start_method()
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
                        _run_single_case,
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


if __name__ == "__main__":
    main()
