#!/usr/bin/env python3
"""Routed-MoE kernel time per layer: TP-sharded experts vs a wide-EP rank.

Runs FlashInfer ``cutlass_fused_moe`` (MXFP8 activations x MXFP4 weights, the
SM120 decode path of DeepSeek-V4) on one GPU with random weights, autotuned,
inside a CUDA graph over ``--layers`` distinct layers so weights are re-read
from DRAM every layer (no L2 reuse across layers). Routing is uniform random
top-k over the global experts.

  tp: every rank holds all experts with 1/``--tp`` of the intermediate dim and
      processes all tokens of the TP group.
  ep: one rank of a ``--ep``-way expert-parallel group holds
      experts/``--ep`` full experts; tokens are the global batch, ids outside
      the rank's range are masked (-1), as after an all-to-all dispatch.

Output: one JSON line per point with us/layer, active local experts, routed
(token, expert) pairs and the effective weight bandwidth.
"""
import argparse
import json
import random

import torch
from flashinfer import block_scale_interleave, mxfp8_quantize
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
from flashinfer.fused_moe.core import ActivationType

FP4_BLOCK = 32


def make_layer(E, inter, D, dev):
    w13 = torch.randint(-128, 127, (E, 2 * inter, D // 2), dtype=torch.int8, device=dev)
    w2 = torch.randint(-128, 127, (E, D, inter // 2), dtype=torch.int8, device=dev)
    s13 = torch.randint(
        118, 126, (E, 2 * inter, D // FP4_BLOCK), dtype=torch.uint8, device=dev
    )
    s2 = torch.randint(
        118, 126, (E, D, inter // FP4_BLOCK), dtype=torch.uint8, device=dev
    )
    s13 = block_scale_interleave(s13).reshape(E, 2 * inter, D // FP4_BLOCK)
    s2 = block_scale_interleave(s2).reshape(E, D, inter // FP4_BLOCK)
    return w13, w2, s13, s2


def routing(n_tok, args, E_local, mode, dev):
    ids = torch.empty((n_tok, args.topk), dtype=torch.int32)
    w = torch.rand((n_tok, args.topk)) + 0.1
    for t in range(n_tok):
        ids[t] = torch.tensor(random.sample(range(args.experts), args.topk))
    if mode == "ep":
        loc = ids
        valid = (loc >= 0) & (loc < E_local)
        ids = torch.where(valid, loc, torch.full_like(loc, -1))
    return ids.to(dev), w.to(dev)


def bench(args, mode, n_tok, dev):
    D = args.hidden
    if mode == "tp":
        E, inter = args.experts, args.inter // args.tp
    else:
        E, inter = args.experts // args.ep, args.inter
    layers = [make_layer(E, inter, D, dev) for _ in range(args.layers)]
    ones = torch.ones(E, dtype=torch.float32, device=dev)
    lim = torch.full_like(ones, 10.0)
    x = torch.randn(n_tok, D, dtype=torch.bfloat16, device=dev)
    xq, xsf = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=True)
    out = torch.empty_like(x)
    cap = 1 if n_tok <= 1 else 1 << (n_tok - 1).bit_length()
    ws_bytes = cutlass_fused_moe_workspace_size(
        cap, D, inter, E, args.topk,
        x_dtype=torch.float8_e4m3fn, weight_dtype=torch.long,
        output_dtype=torch.bfloat16, activation_type=ActivationType.Swiglu,
        use_mxfp8_act_scaling=True, use_fused_finalize=False, device=dev,
    )  # fmt: skip
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    routs = [routing(n_tok, args, E, mode, dev) for _ in range(args.layers)]
    if mode == "ep":
        pairs = [int((r[0] >= 0).sum()) for r in routs]
        act = [int(torch.unique(r[0][r[0] >= 0]).numel()) for r in routs]
    else:
        pairs = [n_tok * args.topk] * args.layers
        act = [int(torch.unique(r[0]).numel()) for r in routs]

    def run(li, tune=False):
        w13, w2, s13, s2 = layers[li]
        ids, wt = routs[li]
        kw = dict(
            input=xq, token_selected_experts=ids,
            token_final_scales=wt.float().contiguous(),
            fc1_expert_weights=w13.view(torch.uint8).view(torch.long),
            fc2_expert_weights=w2.view(torch.uint8).view(torch.long),
            output_dtype=torch.bfloat16,
            quant_scales=[s13.view(torch.int32), ones, s2.view(torch.int32), ones],
            input_sf=xsf, swiglu_limit=lim, output=out,
            use_mxfp8_act_scaling=True, use_fused_finalize=False, enable_pdl=False,
            workspace_buffer=ws, tune_max_num_tokens=n_tok,
            activation_type=ActivationType.Swiglu,
        )  # fmt: skip
        if tune:
            with autotune(True):
                cutlass_fused_moe(**kw)
        else:
            cutlass_fused_moe(**kw)

    run(0, tune=True)
    for li in range(args.layers):
        run(li)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for li in range(args.layers):
            run(li)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(args.iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    per_layer = s.elapsed_time(e) / args.iters / args.layers * 1e3
    exp_bytes = 3 * D * inter * (0.5 + 1 / FP4_BLOCK)
    eff_bw = sum(act) / args.layers * exp_bytes / (per_layer * 1e-6) / 1e9
    del layers, g, ws
    torch.cuda.empty_cache()
    return dict(
        mode=mode, experts_local=E, inter=inter, n_tok=n_tok,
        us_per_layer=round(per_layer, 1),
        active_experts=round(sum(act) / args.layers, 1),
        routed_pairs=round(sum(pairs) / args.layers, 1),
        eff_weight_bw_GBs=round(eff_bw),
    )  # fmt: skip


def ints(s):
    return [int(v) for v in s.split(",") if v]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tp-tokens", default="8,16,24,32,48,64,96,128",
                    help="tokens of the TP group per step")  # fmt: skip
    ap.add_argument("--ep-tokens", default="32,64,96,128,192,256,384,512",
                    help="global tokens of the EP group per step")  # fmt: skip
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--ep", type=int, default=16)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=2048, help="moe_intermediate_size")
    ap.add_argument("--layers", type=int, default=43)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device("cuda")
    for mode, toks in (("tp", ints(args.tp_tokens)), ("ep", ints(args.ep_tokens))):
        for n in toks:
            print(json.dumps(bench(args, mode, n, dev)), flush=True)


if __name__ == "__main__":
    main()
