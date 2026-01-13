import multiprocessing as mp
import random
from typing import List

import torch
import pytest
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.distributed.collective_torch import (
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
DeepepNormalRouter = pytest.importorskip("rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.deepep_normal_router").DeepepNormalRouter
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.port_util import PortManager

import rtp_llm.ops.compute_ops as compute_ops  # isort:skip
from pytest import mark

def init_router(
    rank: int, use_fp8: bool, parallelism_config: ParallelismConfig, nccl_port: int
):
    # Create configuration objects
    model_config = ModelConfig()
    model_config.expert_num = 16
    model_config.hidden_size = 1024

    # Use the provided parallelism_config directly
    parallelism_config.world_rank = rank
    parallelism_config.local_rank = rank
    parallelism_config.nccl_ip = "127.0.0.1"
    parallelism_config.th_nccl_port = nccl_port

    moe_config = MoeConfig()
    moe_config.use_deepep_low_latency = False

    # Create MoEConfigAdapter
    config_adapter = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        max_generate_batch_size=0,
    )

    init_distributed_environment(
        parallelism_config,
        nccl_ip="127.0.0.1",
        th_nccl_port=nccl_port,
        backend="nccl",
        timeout=60,
    )
    router = DeepepNormalRouter(config_adapter, use_fp8, expert_alignment=1)
    return config_adapter, router


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


def worker_function(
    rank: int,
    use_fp8: bool,
    token_num_per_rank: List[int],
    parallelism_config: ParallelismConfig,
    nccl_port: int,
):
    random.seed(rank)
    config, router = init_router(rank, use_fp8, parallelism_config, nccl_port)
    try:
        top_k = config.expert_num
        # test dispatch
        current_device = torch.device(f"cuda:{rank}")
        for i in range(5):
            token_num = token_num_per_rank[rank]
            a1 = (
                torch.randn([token_num, config.hidden_size])
                .to(current_device)
                .to(torch.bfloat16)
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
            )
            assert payload.expert_tokens_meta.expert_num_tokens_cpu == [
                sum(token_num_per_rank)
            ] * (config.expert_num // config.world_size)
            if router.use_fp8:
                combine_x = dequant_to_bf16(payload.expert_x, payload.expert_x_scale)
            else:
                combine_x = payload.expert_x
            a2 = router.finalize(combine_x, topk_weights, topk_ids, False, None)
            if router.use_fp8:
                x, scale = compute_ops.trt_fp8_quantize_128(a1, False)
                ref_a2 = dequant_to_bf16(x, scale) * config.world_size
            else:
                ref_a2 = a1 * config.world_size
            torch.testing.assert_close(ref_a2, a2)
            print("pass test")
    finally:
        destroy_distributed_environment()


@mark.MI308X
@mark.rocm
@mark.gpu
@mark.skip(reason="TODO: fix this test")
def test_single(world_size: int, use_fp8: bool):
    port_manager = PortManager()
    ports, locks = port_manager.get_consecutive_ports(1)
    nccl_port = ports[0]

    tp_size = world_size  # TP size equals world_size for ROCm normal router
    dp_size = 1
    ep_size = world_size  # EP size equals world_size for normal router

    # 启动world_size个进程
    processes = []
    token_num_per_rank = [random.randint(4, 12) // 4 * 4 for _ in range(world_size)]
    for rank in range(world_size):
        # Calculate parallelism config for this rank
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = tp_size
        parallelism_config.tp_rank = rank % tp_size
        parallelism_config.ep_size = ep_size
        parallelism_config.ep_rank = rank % ep_size
        parallelism_config.dp_size = dp_size
        parallelism_config.dp_rank = rank // tp_size
        parallelism_config.world_size = world_size
        parallelism_config.world_rank = rank
        parallelism_config.local_world_size = world_size

        # 创建进程
        p = mp.Process(
            target=worker_function,
            args=(rank, use_fp8, token_num_per_rank, parallelism_config, nccl_port),
            kwargs={},
        )
        processes.append(p)
        p.start()

    # Release locks after all processes start
    for lock in locks:
        lock.__exit__(None, None, None)

    for p in processes:
        p.join(timeout=30)
        # KILL Process
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
            raise Exception("Process timeout")
        else:
            if p.exitcode != 0:
                raise RuntimeError(f"子进程异常退出，退出码: {p.exitcode}")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    # 获取当前可用的GPU数量
    max_gpu_count = torch.cuda.device_count()
    print(f"当前可用GPU数量: {max_gpu_count}")

    # 根据最大GPU数量裁剪world_size列表
    available_world_sizes = [ws for ws in [2, 4] if ws <= max_gpu_count]
    print(f"可用的world_size: {available_world_sizes}")

    # 为每个world_size运行test_single函数
    for use_fp8 in [False]:
        for world_size in available_world_sizes:
            test_single(world_size, use_fp8)
