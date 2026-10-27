#!/usr/bin/env python3
"""Latency of small NCCL collectives at decode message sizes, inside CUDA graphs.

For each token count T (hidden ``--hidden``, bf16) measures all_reduce of
[T, H], reduce_scatter [T, H] -> [T/W, H], all_gather [T/W, H] -> [T, H] and
all_to_all_single of [T, H/2] bf16 (the byte size of an FP8 dispatch). Each op
is captured 20x in a CUDA graph and replayed, matching how decode runs them.
Run with the NCCL environment of the server under test, e.g.

  NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 \\
  torchrun --nproc_per_node 4 nccl_decode_collectives.py

Output: one JSON line per token count on rank 0 (us per call).
"""
import argparse
import json
import os

import torch
import torch.distributed as dist


def timeit(fn, iters):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters // 20):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / (iters // 20 * 20) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", default="8,16,24,32,48,64,96,128")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")
    H = args.hidden
    bf16 = torch.bfloat16
    for tok in [int(v) for v in args.tokens.split(",") if v]:
        x = torch.randn(tok, H, dtype=bf16, device=dev)
        r = {"world": ws, "tokens": tok, "bytes": tok * H * 2}
        r["allreduce_us"] = round(timeit(lambda: dist.all_reduce(x), args.iters), 1)
        if tok % ws == 0:
            o_rs = torch.empty(tok // ws, H, dtype=bf16, device=dev)
            r["reduce_scatter_us"] = round(
                timeit(lambda: dist.reduce_scatter_tensor(o_rs, x), args.iters), 1
            )
            part = torch.randn(tok // ws, H, dtype=bf16, device=dev)
            o_ag = torch.empty(tok, H, dtype=bf16, device=dev)
            r["all_gather_us"] = round(
                timeit(lambda: dist.all_gather_into_tensor(o_ag, part), args.iters), 1
            )
            a_in = torch.randn(tok, H // 2, dtype=bf16, device=dev)
            a_out = torch.empty_like(a_in)
            r["a2a_fp8_us"] = round(
                timeit(lambda: dist.all_to_all_single(a_out, a_in), args.iters), 1
            )
        if rank == 0:
            print(json.dumps(r), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
