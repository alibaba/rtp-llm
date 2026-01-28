# type: ignore
import logging
import os
import pytest

import torch
import torch.multiprocessing as mp

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
DeepEPWrapper = pytest.importorskip("rtp_llm.models_py.distributed.deepep_wrapper").DeepEPWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
DeepEpLowLatencyRouter = pytest.importorskip("rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router").DeepEpLowLatencyRouter
from rtp_llm.ops import MoeConfig, ParallelismConfig, RuntimeConfig
from rtp_llm.test.utils.numeric_util import per_token_cast_back
from rtp_llm.test.utils.port_util import PortManager
from pytest import mark


NUM_TOKEN_PER_RANK = 64
HIDDEN_SIZE = 7168
TOPK = 8
NUM_EXPERTS = 128


def _init_router(
    rank: int, use_fp8: bool, parallelism_config: ParallelismConfig, nccl_port: int
):
    # set env
    world_size = parallelism_config.world_size
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(world_size))
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
    os.environ["ACCL_TOPO_FIX"] = "1"
    os.environ["ACCL_LOAD_BALANCE"] = "1"
    # init params
    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 128
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 500000
    model_config.moe_k = TOPK
    model_config.expert_num = NUM_EXPERTS
    model_config.hidden_size = HIDDEN_SIZE

    # Use the provided parallelism_config directly
    parallelism_config.nccl_ip = "127.0.0.1"
    parallelism_config.th_nccl_port = nccl_port

    moe_config = MoeConfig()
    moe_config.use_deepep_low_latency = True
    moe_config.use_deepep_internode = False

    runtime_config = RuntimeConfig()
    runtime_config.max_generate_batch_size = NUM_TOKEN_PER_RANK

    config = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        max_generate_batch_size=NUM_TOKEN_PER_RANK,
    )

    torch.cuda.set_device(parallelism_config.local_rank)
    torch.set_default_device(f"cuda:{parallelism_config.local_rank}")
    init_distributed_environment(
        parallelism_config=parallelism_config, backend="nccl", timeout=60
    )
    # DeepEPWrapper will be initialized by router with correct ll_num_max_token_per_rank

    router = DeepEpLowLatencyRouter(
        config,
        FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn if use_fp8 else None,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=[128, 128] if use_fp8 else None,
        ),
    )

    return config, router


def _destroy_router(router: DeepEpLowLatencyRouter):
    del router
    DeepEPWrapper.reset()
    destroy_distributed_environment()


def _run_deepep_low_latency_router_test(
    rank: int, use_fp8: bool, parallelism_config: ParallelismConfig, nccl_port: int
):
    config, router = _init_router(rank, use_fp8, parallelism_config, nccl_port)
    # construct data
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    ep_size = config.ep_size
    ep_rank = config.ep_rank
    dp_size = config.dp_size
    dp_rank = config.dp_rank
    num_experts = config.expert_num
    hidden_size = config.hidden_size
    num_topk = config.moe_k
    num_max_tokens = router._ll_num_max_token_per_rank * ep_size

    # for per dp rank
    num_token_per_rank = config.max_generate_batch_size
    int_mask = (2**32) - 1
    hidden_states = torch.randn((dp_size, num_token_per_rank, hidden_size)).to(
        torch.bfloat16
    )
    # hidden_states = torch.ones((ep_size, num_token_per_rank, hidden_size)).to(torch.bfloat16) * torch.arange(
    #     2, ep_size + 2, dtype=torch.bfloat16
    # ).view(ep_size, 1, 1)
    hidden_states[:, :, :128] = (
        torch.arange(dp_size * num_token_per_rank, dtype=torch.bfloat16)
        .view(dp_size, num_token_per_rank, 1)
        .repeat(1, 1, 128)
        .cuda()
    )
    topk_ids = torch.rand(dp_size, num_token_per_rank, num_experts).topk(
        num_topk, dim=-1, largest=True
    )[1]
    topk_weights = (
        torch.ones((dp_size, num_token_per_rank, num_topk)).to(torch.float32) / num_topk
    )
    # print(f"[rank: {rank}] hidden_states: {hidden_states[:5, :5, 127:132]}")
    # reference data
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    ref_recv_x = (
        torch.zeros((num_local_experts, num_max_tokens, hidden_size))
        .to(torch.bfloat16)
        .cuda()
    )
    ref_recv_count = torch.zeros((num_local_experts), dtype=torch.int32).cuda()
    for local_expert_id in range(num_local_experts):
        expert_id = ep_rank * num_local_experts + local_expert_id
        expert_mask = (topk_ids == expert_id).any(
            dim=-1
        )  # shape: (ep_size, num_token_per_rank)
        num_selected_tokens = expert_mask.sum()
        ref_recv_x[local_expert_id, :num_selected_tokens] = hidden_states[expert_mask]
        ref_recv_count[local_expert_id] = num_selected_tokens
    # router prepare
    payload = router.prepare(
        hidden_states[dp_rank],
        None,
        None,
        topk_weights[dp_rank],
        topk_ids[dp_rank],
    )
    # quant back to bfloat16 if use_fp8
    num_expert_recv_tokens = payload.expert_tokens_meta.expert_num_tokens
    if use_fp8 and payload.expert_x_scale is not None:
        recv_x = torch.zeros(payload.expert_x.size(), dtype=torch.bfloat16).cuda()
        for local_expert_id in range(num_local_experts):
            recv_x[local_expert_id] = per_token_cast_back(
                payload.expert_x[local_expert_id],
                payload.expert_x_scale[local_expert_id],
            )
    else:
        recv_x = payload.expert_x
    # print handle
    # for local_expert_id in range(min(5, num_local_experts)):
    #     recv_count, recv_src_info, recv_layout_range = (
    #         num_expert_recv_tokens[local_expert_id],
    #         router.handle[0][local_expert_id],
    #         router.handle[1][local_expert_id],
    #     )
    #     print(f"[rank: {rank}] recv_src_info: {recv_src_info[:recv_count]}")
    #     for j in range(ep_size):
    #         begin_idx, count = (recv_layout_range[j] >> 32).item(), (recv_layout_range[j] & int_mask).item()
    #         print(f"[rank: {rank}] recv_rank: {j}, begin_idx: {begin_idx}, count: {count}")
    # permute recv_x
    permuted_recv_x = torch.zeros_like(recv_x)
    for local_expert_id in range(num_local_experts):
        current_start_idx = 0
        recv_src_info = router.handle[0][local_expert_id]
        recv_layout_range = router.handle[1][local_expert_id]
        for j in range(ep_size):
            begin_idx, count = (recv_layout_range[j] >> 32).item(), (
                recv_layout_range[j] & int_mask
            ).item()
            sorted_indices_per_rank = torch.argsort(
                recv_src_info[begin_idx : begin_idx + count]
            )
            permuted_recv_x[
                local_expert_id, current_start_idx : current_start_idx + count
            ] = recv_x[local_expert_id, begin_idx + sorted_indices_per_rank]
            current_start_idx += count
    # check recv_x
    # print(f"[rank: {rank}] recv_x: {recv_x[:5, :5, 127:132]}, ref_recv_x: {ref_recv_x[:5, :5, 127:132]}")
    # torch.testing.assert_close(ref_recv_x, recv_x)
    # check permuted_recv_x
    # print(
    #     f"[rank: {rank}] permuted_recv_x: {permuted_recv_x[:5, :5, 127:132]}, ref_recv_x: {ref_recv_x[:5, :5, 127:132]}"
    # )
    torch.testing.assert_close(
        ref_recv_x[:, :, :128], permuted_recv_x[:, :, :128], atol=1e-2, rtol=1e-1
    )
    # check num_expert_recv_tokens
    # print(
    #     f"[rank: {rank}] ref_recv_count: {ref_recv_count}, payload.expert_tokens_meta.expert_num_tokens: {payload.expert_tokens_meta.expert_num_tokens}"
    # )
    torch.testing.assert_close(
        ref_recv_count, num_expert_recv_tokens, atol=1e-8, rtol=1e-8
    )
    # pass num_token_per_rank to finalize for gather
    extra_finalize_args = {"original_num_tokens": num_token_per_rank}
    # router finalize
    combined_x = router.finalize(
        CombineForwardPayload(fused_expert_output=recv_x),
        payload.expert_topk_weights,
        payload.expert_topk_ids,
        False,
        extra_finalize_args,
    )
    # print(f"[rank: {rank}] combined_x: {combined_x[:5, 127:132]}, hidden_states: {hidden_states[ep_rank][:5, 127:132]}")
    torch.testing.assert_close(
        hidden_states[dp_rank, :, :128], combined_x[:, :128], atol=1e-2, rtol=1e-1
    )
    _destroy_router(router)


def _spawn_wrapper(
    rank: int, use_fp8: bool, world_size: int, test_tp_size: int, nccl_port: int
):
    """Wrapper function for mp.spawn that calculates parallelism config."""
    dp_size = world_size // test_tp_size
    ep_size = world_size  # EP size equals world_size for low latency router

    # Calculate parallelism config for this rank
    parallelism_config = ParallelismConfig()
    parallelism_config.tp_size = test_tp_size
    parallelism_config.tp_rank = rank % test_tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.ep_rank = rank % ep_size
    parallelism_config.dp_size = dp_size
    parallelism_config.dp_rank = rank // test_tp_size
    parallelism_config.local_rank = rank
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = rank
    parallelism_config.local_world_size = world_size
    _run_deepep_low_latency_router_test(rank, use_fp8, parallelism_config, nccl_port)


@mark.H20
@mark.cuda
@mark.gpu(count=2)
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "use_fp8",
    [True, False],
    ids=lambda v: f"use_fp8={v}",
)
@pytest.mark.parametrize(
    "test_tp_size",
    [1, 2],
    ids=lambda v: f"tp_size={v}",
)
def test_deepep_low_latency_router(use_fp8: bool, test_tp_size: int):
    port_manager = PortManager()
    ports, locks = port_manager.get_consecutive_ports(1)
    nccl_port = ports[0]

    world_size = 2
    try:
        logging.info(
            f"test_deepep_low_latency_router: use_fp8: {use_fp8}, test_tp_size: {test_tp_size}, world_size: {world_size}"
        )
        mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
            _spawn_wrapper,
            args=(use_fp8, world_size, test_tp_size, nccl_port),
            nprocs=world_size,
            join=True,
        )
    finally:
        for lock in locks:
            lock.__exit__(None, None, None)


if __name__ == "__main__":
    test_deepep_low_latency_router()
