#!/usr/bin/env python3
"""Distributed GPU smoke test for the rtp-llm MegaMoE backend (FlyDSL 2-stage fused).

Exercises the full rtp-llm path — MegaMoeWrapper -> MegaMoePassthroughRouter ->
MegaMoeFusedExecutor -> FlyDSL FusedMoEZeroCopyFp8 — through the FusedMoe module,
and asserts the output shape matches the input and values are finite.

Requires 8 AMD GPUs + FlyDSL + mori. Launch with::

  MORI_SHMEM_HEAP_SIZE=8G \\
  PYTHONPATH=/home/admin/qinhanwen/codes/rtp-llm:/home/admin/qinhanwen/codes/FlyDSL \\
  torchrun --standalone --nproc_per_node=8 \\
      rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/megamoe_gpu_smoke.py \\
      --model-dim 4096 --inter-dim 2048 --experts 32 --topk 4 --tokens 128
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

DTYPE_FP8 = torch.float8_e4m3fnuz


def _info(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def _setup_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=dev
        )
    return rank, world, local_rank, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=4096)
    ap.add_argument("--inter-dim", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--max-tok-per-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rank, world, local_rank, dev = _setup_dist()
    assert args.experts % world == 0, "experts must divide world_size"
    epr = args.experts // world
    model_dim, inter_dim = args.model_dim, args.inter_dim
    topk = args.topk

    from rtp_llm.models_py.distributed.megamoe_wrapper import (
        MegaMoeWrapper,
        MegaMoeWrapperConfig,
        init_megamoe_wrapper_from_config,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
    from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
        FusedMoEQuantConfig,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm._utils import (
        get_rocm_fp8_dtype,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.megamoe_executor import (
        MegaMoeFusedExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.routers.megamoe_router import (
        MegaMoePassthroughRouter,
    )
    from rtp_llm.utils.model_weight import W

    if not MegaMoeWrapper.supported():
        _info(rank, "[SKIP] FlyDSL MegaMoE not importable in this env")
        return

    # power-of-two max_tok_per_rank
    mtpr = 1
    while mtpr < max(args.max_tok_per_rank, args.tokens):
        mtpr <<= 1

    wrapper_config = MegaMoeWrapperConfig(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=args.experts,
        topk=topk,
        max_tok_per_rank=mtpr,
    )
    init_megamoe_wrapper_from_config(wrapper_config)
    _info(rank, f"[init] wrapper ready epr={epr} mtpr={mtpr}")

    # ---- build FlyDSL-layout weights ----
    from tests.utils import pertoken_quant, shuffle_weight

    init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    w1_fp32 = torch.randn((args.experts, 2 * inter_dim, model_dim), device=dev) * init
    w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=DTYPE_FP8)
    w1_shuffled = shuffle_weight(w1_q)
    w1_local = w1_shuffled[rank * epr : (rank + 1) * epr].contiguous()
    scale_w1_local = scale_w1[rank * epr : (rank + 1) * epr].contiguous()
    w1_flat = w1_local.view(epr * (2 * inter_dim), model_dim).contiguous()
    scale_w1_1d = scale_w1_local.view(-1).contiguous()

    # w2 (bf16, per-rank local experts) for injection.
    w2_local = (
        torch.randn((epr, model_dim, inter_dim), device=dev, dtype=torch.bfloat16)
        * init
    )

    weights = {
        W.moe_w1: w1_flat,
        W.moe_s1: scale_w1_1d,
        W.moe_w2: w2_local,
    }

    # ---- build config adapter ----
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )
    from rtp_llm.ops import MoeConfig, ParallelismConfig

    model_config = ModelConfig()
    model_config.attn_config.head_num = 8
    model_config.expert_num = args.experts
    model_config.moe_k = topk
    model_config.hidden_size = model_dim
    model_config.inter_size = inter_dim
    model_config.moe_inter_size = inter_dim

    pconfig = ParallelismConfig()
    pconfig.ep_size = world
    pconfig.ep_rank = rank
    pconfig.world_size = world
    pconfig.world_rank = rank
    pconfig.local_rank = local_rank
    pconfig.local_world_size = world

    mconfig = MoeConfig()
    mconfig.ll_num_max_token = mtpr

    os.environ["USE_MEGAMOE"] = "1"
    adapter = MoEConfigAdapter(
        model_config=model_config, parallelism_config=pconfig, moe_config=mconfig
    )

    quant_config = FusedMoEQuantConfig(
        quant_dtype=get_rocm_fp8_dtype(), per_act_token_quant=True
    )
    router = MegaMoePassthroughRouter(adapter, quant_config)
    executor = MegaMoeFusedExecutor(adapter, quant_config, weights)
    moe = FusedMoe(router=router, fused_experts=executor, expert_num=args.experts)

    # ---- inputs ----
    torch.manual_seed(args.seed + rank)
    tokens = min(args.tokens, mtpr)
    x = torch.randn((tokens, model_dim), device=dev, dtype=torch.bfloat16) * init
    score = torch.randn((tokens, args.experts), device=dev, dtype=torch.float32)
    tv, ti = torch.topk(score, k=topk, dim=1)
    topk_ids = ti.to(torch.int32).contiguous()
    topk_weights = torch.softmax(tv, dim=1).contiguous()

    torch.cuda.synchronize()
    out = moe.forward(x, topk_weights, topk_ids)
    torch.cuda.synchronize()

    ok_shape = tuple(out.shape) == (tokens, model_dim)
    ok_finite = bool(torch.isfinite(out).all().item())
    nonzero = float(out.abs().sum().item())
    _info(
        rank,
        f"[result] out.shape={tuple(out.shape)} shape_ok={ok_shape} "
        f"finite={ok_finite} abs_sum={nonzero:.3f}",
    )
    assert ok_shape, f"shape mismatch: {out.shape} != ({tokens},{model_dim})"
    assert ok_finite, "output contains non-finite values"
    _info(rank, "[PASS] MegaMoE GPU smoke test succeeded")

    try:
        import mori.shmem as ms

        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
