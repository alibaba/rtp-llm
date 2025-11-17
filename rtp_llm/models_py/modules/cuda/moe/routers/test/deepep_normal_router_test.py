import multiprocessing as mp
import os
import random
from typing import List

import torch
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
from rtp_llm.models_py.modules.cuda.moe.routers.deepep_normal_router import (
    DeepepNormalRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.test.utils.port_util import PortsContext

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128  # isort:skip


def init_router(rank: int, use_fp8: bool):
    g_parallel_info.reload()
    update_master_info(f"0.0.0.0", int(os.environ["START_PORT"]))
    print(f"rank {rank}, {g_parallel_info}")
    config = GptInitModelParameters(0, 0, 0, 0, 0)
    config.moe_config.use_deepep_low_latency = False
    config.expert_num = 16
    config.hidden_size = 1024
    config.tp_size = g_parallel_info.tp_size
    config.tp_rank = g_parallel_info.tp_rank
    config.ep_size = g_parallel_info.ep_size
    config.ep_rank = g_parallel_info.ep_rank
    config.dp_size = g_parallel_info.dp_size
    config.dp_rank = g_parallel_info.dp_rank
    config.ffn_tp_rank = g_parallel_info.ffn_tp_rank
    config.ffn_tp_size = g_parallel_info.ffn_tp_size
    config.local_rank = rank
    init_distributed_environment(config, backend="nccl", timeout=60)
    router = DeepepNormalRouter(config, use_fp8, expert_alignment=1)
    config.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
    config.th_nccl_port = g_master_info.th_nccl_port
    config.tp_nccl_port = g_master_info.tp_nccl_port
    config.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
    init_device(config)
    return config, router


# payload.expert_x: [token, dim], type: fp8
# payload.expert_x_scale: [token, dim / 128], type: fp32
def dequant_to_bf16(expert_x: torch.Tensor, expert_x_scale: torch.Tensor):
    # 需要将scale转置后扩展成[token, dim]，然后和expert_x相乘
    # 转置scale: [dim / 128, token] -> [token, dim / 128]
    # 扩展scale: [token, dim / 128] -> [token, dim]
    # 每个128维的块重复128次
    scale_expanded = expert_x_scale.repeat_interleave(128, dim=1)

    # 将expert_x转换为fp32进行乘法运算
    expert_x_fp32 = expert_x.float()
    # 相乘得到最终的combine_x
    combine_x = expert_x_fp32 * scale_expanded
    return combine_x.bfloat16()


def worker_function(rank: int, use_fp8: bool, token_num_per_rank: List[int]):
    random.seed(rank)
    config, router = init_router(rank, use_fp8)
    try:
        dp_rank = config.dp_rank
        dp_size = config.dp_size
        top_k = config.expert_num
        # test dispatch
        current_device = torch.device(f"cuda:{rank}")
        for i in range(5):
            token_num = token_num_per_rank[dp_rank]
            # 相同dp_rank的a1相同
            torch.manual_seed(rank * dp_size + dp_rank)
            a1 = (
                torch.randn([token_num, config.hidden_size])
                .to(current_device)
                .to(torch.bfloat16)
            )

            a1[:, :128] = (
                torch.arange(token_num, dtype=torch.bfloat16)
                .view(token_num, 1)
                .repeat(1, 128)
                .cuda()
            )

            topk_weights = torch.ones([token_num, top_k]).to(current_device)
            topk_ids = torch.arange(config.expert_num, device=current_device).repeat(
                token_num, 1
            )
            quant_config = FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128],
            )
            payload = router.prepare(
                a1,
                None,
                None,
                topk_weights,
                topk_ids,
                quant_config,
            )
            assert payload.expert_tokens_meta.expert_num_tokens_cpu == [
                sum(token_num_per_rank)
            ] * (config.expert_num // config.world_size)
            if router.use_fp8:
                combine_x = dequant_to_bf16(payload.expert_x, payload.expert_x_scale)
            else:
                combine_x = payload.expert_x
            # pass token_num to finalize for gather
            extra_finalize_args = {"original_num_tokens": token_num}
            a2 = router.finalize(
                combine_x, topk_weights, topk_ids, False, extra_finalize_args
            )
            if router.use_fp8:
                x, scale = trt_fp8_quantize_128(a1, False)
                ref_a2 = dequant_to_bf16(x, scale) * config.world_size
            else:
                ref_a2 = a1 * config.world_size
            torch.testing.assert_close(ref_a2[:, :128], a2[:, :128])
    finally:
        destroy_distributed_environment()


def test_single(world_size: int, test_tp_size: int, use_fp8: bool):
    with PortsContext(None, 1) as ports:
        start_port = ports[0]
        os.environ["START_PORT"] = str(start_port)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["TP_SIZE"] = str(test_tp_size)
        os.environ["DP_SIZE"] = str(world_size // test_tp_size)

        # 启动world_size个进程
        processes = []
        token_num_per_rank = [
            random.randint(4, 12) // 4 * 4 for _ in range(world_size // test_tp_size)
        ]
        for rank in range(world_size):
            # 为每个进程设置环境变量
            os.environ["WORLD_RANK"] = str(rank)
            # 创建进程，调用函数留空
            p = mp.Process(
                target=worker_function,
                args=(rank, use_fp8, token_num_per_rank),
                kwargs={},
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=300)
            # KILL Process
            if p.is_alive():
                p.terminate()
                p.join(timeout=60)
                if p.is_alive():
                    p.kill()
                    p.join()
                raise Exception("Process timeout")
            else:
                if p.exitcode != 0:
                    raise RuntimeError(f"子进程异常退出，退出码: {p.exitcode}")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    world_size = 2
    test_tp_sizes = [1, 2]

    # 为每个world_size运行test_single函数
    for use_fp8 in [True, False]:
        for test_tp_size in test_tp_sizes:
            test_single(world_size, test_tp_size, use_fp8)
