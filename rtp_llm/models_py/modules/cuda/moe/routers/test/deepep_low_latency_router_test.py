# type: ignore
import os

import torch
import torch.multiprocessing as mp
from librtp_compute_ops import init_device

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.worker_info import (
    g_master_info,
    g_parallel_info,
    update_master_info,
)
from rtp_llm.models_py.distributed.test.process_group_state import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.cuda.moe.routers.deepep_low_latency_router import (
    DeepEpLowLatencyRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.test.utils.port_util import PortsContext

NUM_TOKEN_PER_RANK = 64
HIDDEN_SIZE = 7168
TOPK = 8
NUM_EXPERTS = 128


def per_token_cast_back(x_fp8: torch.Tensor, x_scales: torch.Tensor):
    if x_scales.dtype == torch.int:
        if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
            x_scales = x_scales << 23
        else:
            x_scales = x_scales.view(dtype=torch.int8).to(torch.int) << 23

        x_scales = x_scales.view(dtype=torch.float)

    if os.getenv("ACCL_FP8_CAST_LEVEL", "1") == "2":
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, x_fp8.size(1))
    else:
        x_fp32 = x_fp8.to(torch.float32).view(x_fp8.size(0), -1, 128)

    x_scales = x_scales.view(x_fp8.size(0), -1, 1)
    return (x_fp32 * x_scales).view(x_fp8.shape).to(torch.bfloat16)


def _init_router(rank: int, use_fp8: bool):
    # set env
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(int(os.environ["WORLD_SIZE"]))
    )
    os.environ["ACCL_DISPATCH_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_COMBINE_NUM_WARP_GROUPS"] = "4"
    os.environ["ACCL_LOW_LATENCY_OPTIMIZE"] = "1"
    os.environ["ACCL_TOPO_FIX"] = "1"
    os.environ["ACCL_LOAD_BALANCE"] = "1"
    os.environ["WORLD_RANK"] = str(rank)
    g_parallel_info.reload()
    update_master_info(f"0.0.0.0", int(os.environ["MASTER_PORT"]))
    # init params
    config = GptInitModelParameters(
        head_num=2,
        size_per_head=128,
        layer_num=2,
        max_seq_len=2048,
        vocab_size=500000,
    )
    config.nccl_ip = "127.0.0.1"
    config.dp_rank = g_parallel_info.dp_rank
    config.dp_size = g_parallel_info.dp_size
    config.tp_rank = g_parallel_info.tp_rank
    config.tp_size = g_parallel_info.tp_size
    config.ep_rank = g_parallel_info.ep_rank
    config.ep_size = g_parallel_info.ep_size
    config.local_rank = rank
    config.world_size = g_parallel_info.world_size
    config.moe_config.use_deepep_low_latency = True
    config.moe_config.use_deepep_internode = False
    config.gpt_init_params.ffn_disaggregate_config.enable_ffn_disaggregate = False
    config.moe_k = TOPK
    config.expert_num = NUM_EXPERTS
    config.hidden_size = HIDDEN_SIZE
    config.max_generate_batch_size = NUM_TOKEN_PER_RANK
    torch.cuda.set_device(config.local_rank)
    torch.set_default_device(f"cuda:{config.local_rank}")
    init_distributed_environment(config, backend="nccl", timeout=60)
    router = DeepEpLowLatencyRouter(
        config,
        use_fp8_dispatch=use_fp8,
        zero_copy=False,
        async_finish=False,
        return_recv_hook=False,
    )
    config.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
    config.th_nccl_port = g_master_info.th_nccl_port
    config.tp_nccl_port = g_master_info.tp_nccl_port
    config.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
    init_device(config)
    return config, router


def _destroy_router(router: DeepEpLowLatencyRouter):
    del router
    destroy_distributed_environment()


def _run_deepep_low_latency_router_test(rank: int, use_fp8: bool):
    config, router = _init_router(rank, use_fp8)
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
    ll_num_token_per_rank = (
        config.max_generate_batch_size + config.tp_size - 1
    ) // config.tp_size
    num_max_tokens = ll_num_token_per_rank * ep_size

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
        (
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128],
            )
            if use_fp8
            else FusedMoEQuantConfig(quant_dtype=None)
        ),
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
    torch.testing.assert_close(ref_recv_x[:, :, :128], permuted_recv_x[:, :, :128])
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
        recv_x,
        payload.expert_topk_weights,
        payload.expert_topk_ids,
        False,
        extra_finalize_args,
    )
    # print(f"[rank: {rank}] combined_x: {combined_x[:5, 127:132]}, hidden_states: {hidden_states[ep_rank][:5, 127:132]}")
    torch.testing.assert_close(hidden_states[dp_rank, :, :128], combined_x[:, :128])
    _destroy_router(router)


def test_deepep_low_latency_router():
    with PortsContext(None, 1) as ports:
        os.environ["MASTER_PORT"] = str(ports[0])

    world_size = 2
    test_tp_sizes = [1, 2]

    for use_fp8 in [True, False]:
        for test_tp_size in test_tp_sizes:
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["TP_SIZE"] = str(test_tp_size)
            os.environ["DP_SIZE"] = str(world_size // test_tp_size)
            mp.spawn(  # pyright: ignore[reportPrivateImportUsage]
                _run_deepep_low_latency_router_test,
                args=(use_fp8,),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":
    test_deepep_low_latency_router()
