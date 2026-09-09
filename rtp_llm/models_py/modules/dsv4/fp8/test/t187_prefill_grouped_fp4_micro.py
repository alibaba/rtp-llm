#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep import (
    PpuLegacyDeepEPStrategy as DeepEPStrategy,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_legacy_deepep import (
    _select_ppu_grouped_fp4_capacity,
)
from rtp_llm.utils.model_weight import W


def make_weights(experts: int, dim: int, inter: int) -> dict:
    def packed(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.randint(
            -32,
            32,
            (experts, out_dim, in_dim // 2),
            dtype=torch.int8,
            device="cuda",
        )

    def scales(out_dim: int, in_dim: int) -> torch.Tensor:
        return torch.full(
            (experts, out_dim, in_dim // 32),
            120,
            dtype=torch.uint8,
            device="cuda",
        ).view(torch.float8_e8m0fnu)

    return {
        W.v4_routed_w1_w: packed(inter, dim),
        W.v4_routed_w1_s: scales(inter, dim),
        W.v4_routed_w2_w: packed(dim, inter),
        W.v4_routed_w2_s: scales(dim, inter),
        W.v4_routed_w3_w: packed(inter, dim),
        W.v4_routed_w3_s: scales(inter, dim),
    }


def clone_weights(weights: dict) -> dict:
    return {name: tensor.clone() for name, tensor in weights.items()}


def bench(fn, warmup: int = 2, iters: int = 5) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--inter", type=int, default=256)
    parser.add_argument("--route-slots", type=int, default=6)
    parser.add_argument("--local-routes-per-token", type=int, default=6)
    parser.add_argument("--max-tokens-per-rank", type=int)
    parser.add_argument("--selected-capacity", type=int)
    parser.add_argument("--ll-capacity", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(20260826)
    experts, dim, inter, topk = 32, args.dim, args.inter, 6
    if not 0 < args.local_routes_per_token <= args.route_slots:
        raise ValueError("local routes must be in [1, route slots]")
    cfg = MoeCfg(
        layer_id=0,
        dim=dim,
        moe_inter_dim=inter,
        n_routed_experts=256,
        n_activated_experts=topk,
        swiglu_limit=10.0,
        ep_size=8,
        ep_rank=0,
        n_local_experts=experts,
        local_expert_start=0,
        local_expert_end=experts,
        max_tokens_per_rank=args.max_tokens_per_rank or args.tokens,
    )
    weights = make_weights(experts, dim, inter)

    os.environ["DSV4_PPU_GROUPED_FP4"] = "1"
    grouped = DeepEPStrategy(cfg).cuda()
    grouped.setup_weights(clone_weights(weights))

    x = torch.randn(args.tokens, dim, dtype=torch.bfloat16, device="cuda") * 0.2
    indices = torch.full(
        (args.tokens, args.route_slots), -1, dtype=torch.int64, device="cuda"
    )
    indices[:, : args.local_routes_per_token] = (
        torch.arange(
            args.tokens * args.local_routes_per_token,
            dtype=torch.int64,
            device="cuda",
        )
        .view(args.tokens, args.local_routes_per_token)
        .remainder_(experts)
    )
    route_weights = torch.rand(
        args.tokens, args.route_slots, dtype=torch.float32, device="cuda"
    )
    route_weights[:, args.local_routes_per_token :] = 0
    route_weights /= route_weights.sum(dim=-1, keepdim=True)
    valid_indices = indices[indices >= 0]
    counts = torch.bincount(valid_indices, minlength=experts).cpu().tolist()
    capacity = args.selected_capacity or _select_ppu_grouped_fp4_capacity(
        128, counts, experts, fixed_shape=False
    )

    padded_capacity = capacity + 128

    def run_selected() -> torch.Tensor:
        return grouped._forward_ppu_grouped_fp4(x, route_weights, indices, capacity)

    def run_padded() -> torch.Tensor:
        return grouped._forward_ppu_grouped_fp4(
            x, route_weights, indices, padded_capacity
        )

    with torch.inference_mode():
        selected_y = run_selected().clone()
        padded_y = run_padded().clone()
        torch.cuda.synchronize()
        diff = (selected_y.float() - padded_y.float()).abs()
        rel = diff.mean().item() / (padded_y.float().abs().mean().item() + 1e-6)
        selected_ms = bench(run_selected)
        padded_ms = bench(run_padded)

        ll_result = None
        if args.ll_capacity:
            if args.ll_capacity < capacity:
                raise ValueError("ll-capacity must be >= selected capacity")
            os.environ["DSV4_PPU_GROUPED_FP4_CAPACITY"] = str(capacity)
            compact_expert_x = torch.randn(
                experts, capacity, dim, dtype=torch.bfloat16, device="cuda"
            )
            ll_expert_x = torch.zeros(
                experts, args.ll_capacity, dim, dtype=torch.bfloat16, device="cuda"
            )
            ll_expert_x[:, :capacity].copy_(compact_expert_x)
            expert_counts = torch.arange(
                1, experts + 1, dtype=torch.int32, device="cuda"
            ).clamp_(max=capacity)

            def run_compact_packed() -> torch.Tensor:
                return grouped._compute_ppu_grouped_fp4_packed(
                    compact_expert_x, expert_counts
                )

            def run_ll_packed() -> torch.Tensor:
                return grouped._compute_ppu_grouped_fp4_packed(
                    ll_expert_x, expert_counts
                )

            compact_packed_y = run_compact_packed().clone()
            ll_packed_y = run_ll_packed().clone()
            torch.cuda.synchronize()
            valid_rows = torch.arange(capacity, device="cuda").view(1, -1) < (
                expert_counts.view(-1, 1)
            )
            ll_diff = (
                (compact_packed_y[valid_rows] - ll_packed_y[:, :capacity][valid_rows])
                .float()
                .abs()
            )
            ll_rel = ll_diff.mean().item() / (
                compact_packed_y[valid_rows].float().abs().mean().item() + 1e-6
            )
            ll_result = {
                "ll_capacity": args.ll_capacity,
                "compact_prefix_relative_mean_error": ll_rel,
                "compact_ms": bench(run_compact_packed, warmup=1, iters=2),
                "ll_ms": bench(run_ll_packed, warmup=1, iters=2),
            }

    result = {
        "result": "pass" if rel < 0.001 and capacity >= max(counts) else "fail",
        "tokens": args.tokens,
        "experts": experts,
        "dim": dim,
        "inter": inter,
        "topk": topk,
        "route_slots": args.route_slots,
        "local_routes_per_token": args.local_routes_per_token,
        "max_tokens_per_rank": cfg.max_tokens_per_rank,
        "max_expert_tokens": max(counts),
        "selected_capacity": capacity,
        "padded_capacity": padded_capacity,
        "lossless_capacity": capacity >= max(counts),
        "capacity_invariance_relative_mean_error": rel,
        "selected_capacity_ms": selected_ms,
        "padded_capacity_ms": padded_ms,
        "finite": bool(torch.isfinite(selected_y).all().item()),
        "low_latency_compaction": ll_result,
    }
    if (
        ll_result is not None
        and ll_result["compact_prefix_relative_mean_error"] >= 0.001
    ):
        result["result"] = "fail"
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["result"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
