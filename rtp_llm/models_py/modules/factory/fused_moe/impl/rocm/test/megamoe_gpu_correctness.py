#!/usr/bin/env python3
"""Distributed GPU numerical correctness test for the rtp-llm MegaMoE backend.

Unlike megamoe_gpu_smoke.py (which only checks shape + finiteness), this test
compares the FlyDSL 2-stage fused MegaMoE output against an fp8-quant-aware
torch reference and asserts a relative-L2 error below a threshold.

It feeds the executor UNSHUFFLED bf16 weights (w1 in [gate, up] order,
[epr, 2*inter, model_dim]; w2 in [epr, model_dim, inter_dim]) — exactly what the
loader delivers for MegaMoE once the aiter (16,16) pre-shuffle is skipped — so
that the executor's own pertoken_quant + FlyDSL shuffle_weight produce the layout
the kernels expect.

Requires 8 AMD GPUs + FlyDSL + mori. Launch with::

  MORI_SHMEM_HEAP_SIZE=8G \\
  PYTHONPATH=/home/admin/qinhanwen/codes/rtp-llm:/home/admin/qinhanwen/codes/FlyDSL \\
  torchrun --standalone --nproc_per_node=8 \\
      rtp_llm/models_py/modules/factory/fused_moe/impl/rocm/test/megamoe_gpu_correctness.py \\
      --model-dim 2048 --inter-dim 768 --experts 32 --topk 8 --tokens 64
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

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


def _deq(pertoken_quant, w_fp32):
    """pertoken_quant then dequant, mirroring what the fp8 kernels see."""
    q, s = pertoken_quant(w_fp32, quant_dtype=DTYPE_FP8)
    return q.to(torch.float32) * s.to(torch.float32)


def _reference(
    pertoken_quant, x_bf16, topk_ids, topk_weights, w1_fp32, w2_fp32, inter_dim
):
    """fp8-quant-aware torch reference for one rank's tokens.

    w1_fp32: [E, 2*inter, model_dim] in [gate, up] order.
    w2_fp32: [E, model_dim, inter_dim].
    Returns [tokens, model_dim] fp32.
    """
    dev = x_bf16.device
    tokens, model_dim = x_bf16.shape
    topk = topk_ids.shape[1]

    # x per-token quant/dequant (matches kernel's dynamic_per_token_scaled_quant).
    x_q, x_s = pertoken_quant(x_bf16.to(torch.float32), quant_dtype=DTYPE_FP8)
    x_deq = x_q.to(torch.float32) * x_s.to(torch.float32)  # [tokens, model_dim]

    w1_deq = _deq(pertoken_quant, w1_fp32)  # [E, 2*inter, model_dim]
    w2_deq = _deq(pertoken_quant, w2_fp32)  # [E, model_dim, inter_dim]

    out = torch.zeros((tokens, model_dim), device=dev, dtype=torch.float32)
    ids = topk_ids.to(torch.long)
    wts = topk_weights.to(torch.float32)
    for i in range(tokens):
        acc = torch.zeros((model_dim,), device=dev, dtype=torch.float32)
        for k in range(topk):
            e = int(ids[i, k].item())
            y = x_deq[i] @ w1_deq[e].t()  # [2*inter]
            gate = y[:inter_dim]
            up = y[inter_dim:]
            a1 = F.silu(gate) * up  # [inter]
            # requant intermediate to fp8 (stage2 input), like the kernel.
            a1_q, a1_s = pertoken_quant(a1.unsqueeze(0), quant_dtype=DTYPE_FP8)
            a1_deq = (a1_q.to(torch.float32) * a1_s.to(torch.float32)).squeeze(0)
            o = a1_deq @ w2_deq[e].t()  # [model_dim]
            acc += wts[i, k] * o
        out[i] = acc
    return out


def _rel_l2(got, ref):
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    return (torch.norm(got - ref) / (torch.norm(ref) + 1e-8)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dim", type=int, default=2048)
    ap.add_argument("--inter-dim", type=int, default=768)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max-tok-per-rank", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--same-tokens",
        action="store_true",
        help="all ranks use IDENTICAL tokens/routing (reproduces the "
        "real model DP=1/TP=8 replicated-token EP case).",
    )
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument(
        "--force-expert",
        type=int,
        default=-1,
        help="if >=0, force ALL tokens' topk ids to this single "
        "global expert (isolates cross-rank scatter: only the "
        "rank hosting it computes, then scatters to all srcs).",
    )
    args = ap.parse_args()

    rank, world, local_rank, dev = _setup_dist()
    assert args.experts % world == 0, "experts must divide world_size"
    epr = args.experts // world
    model_dim, inter_dim, topk = args.model_dim, args.inter_dim, args.topk

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

    from tests.utils import pertoken_quant  # noqa: F401  (used by reference)

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

    # ---- GLOBAL unshuffled bf16 weights (same on every rank via fixed seed) ----
    init = float(model_dim) ** -0.25
    torch.manual_seed(args.seed)
    w1_fp32 = torch.randn((args.experts, 2 * inter_dim, model_dim), device=dev) * init
    torch.manual_seed(args.seed + 777)
    w2_fp32 = torch.randn((args.experts, model_dim, inter_dim), device=dev) * init

    # Local slice fed to the executor as UNSHUFFLED bf16 (loader-equivalent).
    w1_local_bf16 = (
        w1_fp32[rank * epr : (rank + 1) * epr].to(torch.bfloat16).contiguous()
    )
    w2_local_bf16 = (
        w2_fp32[rank * epr : (rank + 1) * epr].to(torch.bfloat16).contiguous()
    )

    weights = {W.moe_w1: w1_local_bf16, W.moe_w2: w2_local_bf16}

    # ---- config adapter ----
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

    # ---- inputs (per-rank tokens; global expert ids) ----
    # --same-tokens reproduces the real model's DP=1/TP=8 case where every rank
    # sees the IDENTICAL tokens (rank-independent seed); default gives each rank
    # its own distinct tokens (clean per-rank ownership).
    _tok_seed = (
        (args.seed + 1)
        if getattr(args, "same_tokens", False)
        else (args.seed + rank + 1)
    )
    torch.manual_seed(_tok_seed)
    tokens = min(args.tokens, mtpr)
    x = torch.randn((tokens, model_dim), device=dev, dtype=torch.bfloat16) * init
    score = torch.randn((tokens, args.experts), device=dev, dtype=torch.float32)
    tv, ti = torch.topk(score, k=topk, dim=1)
    topk_ids = ti.to(torch.int32).contiguous()
    topk_weights = torch.softmax(tv, dim=1).contiguous()

    if args.force_expert >= 0:
        # Isolate cross-rank scatter: every (token, slot) targets one global
        # expert, so only the rank hosting it computes GEMM, then scatters the
        # result back to every source rank. Distinct experts per slot keep the
        # sorted-table s-field valid (Plan B requires topk distinct dest slots).
        base = int(args.force_expert)
        forced = (
            base + torch.arange(topk, device=dev, dtype=torch.int32)
        ) % args.experts
        topk_ids = forced.unsqueeze(0).repeat(tokens, 1).contiguous()
        topk_weights = torch.full(
            (tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32
        ).contiguous()
        _info(rank, f"[force] all tokens -> experts {forced.tolist()}")

    torch.cuda.synchronize()
    try:
        import mori.shmem as _ms
    except Exception:
        _ms = None

    # Warmup: the FlyDSL chained op JIT-compiles on the first call, which can
    # desync the cross-rank P2P scatter. Warm up (with barriers) before measuring.
    for _w in range(3):
        if _ms is not None:
            _ms.shmem_barrier_all()
        _ = moe.forward(x, topk_weights, topk_ids)
        torch.cuda.synchronize()
    if _ms is not None:
        _ms.shmem_barrier_all()

    out = moe.forward(x, topk_weights, topk_ids).to(torch.float32)
    torch.cuda.synchronize()
    if _ms is not None:
        _ms.shmem_barrier_all()

    def _ref_variant(weighted: bool, swap_gate_up: bool):
        dev_ = x.device
        tokens_, model_dim_ = x.shape
        x_q, x_s = pertoken_quant(x.to(torch.float32), quant_dtype=DTYPE_FP8)
        x_deq = x_q.to(torch.float32) * x_s.to(torch.float32)
        w1_deq = _deq(pertoken_quant, w1_fp32)
        w2_deq = _deq(pertoken_quant, w2_fp32)
        out_ = torch.zeros((tokens_, model_dim_), device=dev_, dtype=torch.float32)
        ids_ = topk_ids.to(torch.long)
        wts_ = topk_weights.to(torch.float32)
        for i in range(tokens_):
            acc = torch.zeros((model_dim_,), device=dev_, dtype=torch.float32)
            for k in range(topk):
                e = int(ids_[i, k].item())
                y = x_deq[i] @ w1_deq[e].t()
                if swap_gate_up:
                    up = y[:inter_dim]
                    gate = y[inter_dim:]
                else:
                    gate = y[:inter_dim]
                    up = y[inter_dim:]
                a1 = F.silu(gate) * up
                a1_q, a1_s = pertoken_quant(a1.unsqueeze(0), quant_dtype=DTYPE_FP8)
                a1_deq = (a1_q.to(torch.float32) * a1_s.to(torch.float32)).squeeze(0)
                o = a1_deq @ w2_deq[e].t()
                acc += (wts_[i, k] * o) if weighted else o
            out_[i] = acc
        return out_

    ref = _ref_variant(weighted=True, swap_gate_up=False)
    ref_unw = _ref_variant(weighted=False, swap_gate_up=False)
    ref_swap = _ref_variant(weighted=True, swap_gate_up=True)

    # best-fit global scale on the primary weighted variant.
    denom = float((ref * ref).sum().item()) + 1e-8
    alpha = float((out * ref).sum().item()) / denom
    rel_scaled = _rel_l2(out, alpha * ref)

    rel = _rel_l2(out, ref)
    rel_unw = _rel_l2(out, ref_unw)
    rel_swap = _rel_l2(out, ref_swap)
    finite = bool(torch.isfinite(out).all().item())

    # gather per-rank [rel, out_norm, ref_norm] for a distribution view.
    stat = torch.tensor(
        [rel, float(torch.norm(out).item()), float(torch.norm(ref).item())],
        device=dev,
    )
    gathered = [torch.zeros_like(stat) for _ in range(world)]
    dist.all_gather(gathered, stat)
    if rank == 0:
        for r_, g in enumerate(gathered):
            print(
                f"[per-rank] rank={r_} relL2={g[0].item():.4f} "
                f"out_norm={g[1].item():.3f} ref_norm={g[2].item():.3f}",
                flush=True,
            )

    rel_t = torch.tensor([rel], device=dev)
    dist.all_reduce(rel_t, op=dist.ReduceOp.MAX)
    rel_max = rel_t.item()

    _info(
        rank,
        f"[diag] relL2 weighted={rel:.4f} unweighted={rel_unw:.4f} "
        f"gateup_swap={rel_swap:.4f} bestfit_scale(alpha={alpha:.3f})={rel_scaled:.4f}",
    )
    _info(
        rank,
        f"[result] relL2(weighted)={rel:.4f} rel_max_over_ranks={rel_max:.4f} "
        f"finite={finite} threshold={args.threshold}",
    )
    _info(rank, f"[sample] out[0,:4]={out[0,:4].tolist()}")
    _info(rank, f"[sample] ref[0,:4]={ref[0,:4].tolist()}")
    _info(rank, f"[sample] ref_unw[0,:4]={ref_unw[0,:4].tolist()}")

    if rel_max < args.threshold:
        _info(
            rank, f"[PASS] MegaMoE correctness relL2={rel_max:.4f} < {args.threshold}"
        )
    else:
        _info(
            rank, f"[FAIL] MegaMoE correctness relL2={rel_max:.4f} >= {args.threshold}"
        )

    try:
        import mori.shmem as ms

        ms.shmem_finalize()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
