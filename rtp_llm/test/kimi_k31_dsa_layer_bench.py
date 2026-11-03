"""K3.1 DSA decode layer latency on eight GPUs: one KimiK31DSA layer replayed as the decode runner does.

The layer, caches, workspace and graph come from kimi_k31_dsa_layer_fixture (serving config, Page-RR
DCP8 geometry, the runner's graph-width page tables, batches padded to the captured bucket). Each
case fills a batch of requests with random history and captures [L2 flush -> layer] * inner in one
graph; the flush-only burst is subtracted. Each rank takes the median over replays and the case
reports the maximum over ranks (the slowest rank gates the collectives).

  torchrun --no-python --nproc_per_node=8 bazel-bin/rtp_llm/test/kimi_k31_dsa_layer_bench
      --config-json <K3.1 checkpoint>/config.json --cases '[[4, 32, 8192]]'
A case is [Q, batch, context] or {"queries": Q, "contexts": [...]}; the default is the K3 cases.
"""

import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.distributed as dist

try:
    from rtp_llm.test import kimi_k31_dsa_layer_fixture as fx
except ImportError:  # run as a plain script next to the fixture (no Bazel runfiles)
    import kimi_k31_dsa_layer_fixture as fx

# K3.1 decode capture sizes, then steps of 16 for larger batches.
CAPTURE = (1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128)
K3_CASES = [[4, b, c] for b, c in (
    (32, 8 << 10), (128, 8 << 10), (12, 16 << 10), (13, 16 << 10), (32, 16 << 10),
    (68, 16 << 10), (128, 16 << 10), (32, 32 << 10), (34, 32 << 10), (128, 32 << 10),
    (17, 64 << 10), (32, 64 << 10), (128, 64 << 10), (8, 128 << 10), (32, 128 << 10),
    (68, 128 << 10), (4, 256 << 10), (8, 256 << 10), (34, 256 << 10), (2, 512 << 10),
    (17, 512 << 10), (1, 1 << 20), (8, 1 << 20),
)]


def parse_case(case):
    if isinstance(case, dict):
        return case["queries"], list(case["contexts"])
    queries, batch, context = case
    return queries, [context] * batch


def bucket_of(batch):
    for size in CAPTURE:
        if size >= batch:
            return size
    return math.ceil(batch / 16) * 16


def measure(body, flush, inner, replays=5):
    """Critical per-call latency (us): max over ranks of the median flush-subtracted burst."""

    def burst(with_body):
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            for _ in range(inner):
                flush.zero_()
                if with_body:
                    body()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()
        times = []
        for _ in range(replays):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(begin.elapsed_time(end) * 1000)
        graph.reset()
        return statistics.median(times)

    local = torch.tensor([(burst(True) - burst(False)) / inner], device="cuda")
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return round(local.item(), 1)


def kernel_table(body, flush, inner=4):
    """GPU time per kernel name (us per layer call) over one replay of [L2 flush -> layer] * inner."""
    from torch.profiler import ProfilerActivity, profile

    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            flush.zero_()
            body()
    graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        # Align the ranks inside the profiled region: collectives then measure transfer, not the
        # skew of the ranks' profiler start-up.
        dist.barrier()
        torch.cuda.synchronize()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    graph.reset()
    inner *= 3
    totals = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA and "AllReduce" not in event.name:
            totals[event.name] = totals.get(event.name, 0.0) + event.device_time / inner
    return sorted(((round(us, 1), name[:90]) for name, us in totals.items()), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--cases", default=json.dumps(K3_CASES))
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--inner", type=int, default=8, help="layer calls per timed graph")
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--kernels", action="store_true", help="also print rank 0's per-kernel GPU time")
    args = parser.parse_args()
    parallel = fx.init_distributed(int(os.environ["WORLD_SIZE"]))
    tp, rank = parallel.tp_size, parallel.tp_rank
    flush = torch.empty(2 * torch.cuda.get_device_properties(0).L2_cache_size, dtype=torch.uint8, device="cuda")
    by_queries = {}
    for case in json.loads(args.cases):
        queries, contexts = parse_case(case)
        by_queries.setdefault(queries, []).append(contexts)
    for queries, cases in by_queries.items():
        config = fx.load_config(args.config_json, max_seq_len=args.max_seq_len, queries=queries)
        layer_index = config.k3_runtime_config.dsa.layers[0]
        weights, _ = fx.loader_weights(config, parallel, seed=args.seed)
        layer = fx.build_layer(config, parallel, weights, layer_index)
        weights_by_layer = [None] * config.num_layers
        weights_by_layer[layer_index] = weights
        unit = fx.sp_unit(tp, queries)
        largest = max(math.ceil(bucket_of(len(c)) / unit) * unit for c in cases)
        owner = fx.build_owner(config, parallel, [(layer_index, weights)], max_requests=largest, queries=queries)
        recorder = fx.Recorder(layer)
        columns = sum(
            max(fx.owner_columns(n, tp, r) for r in range(tp)) for n in max(cases, key=lambda c: sum(c))
        )
        pools = {tag: fx.Pool(tag, columns + 2, "cuda") for tag in fx.TAGS}
        pool_handle = torch.cuda.graph_pool_handle()
        generator = torch.Generator().manual_seed(args.seed + rank)
        hidden_gen = torch.Generator(device="cuda").manual_seed(args.seed * 13 + rank)
        for contexts in cases:
            requests = []
            for index, length in enumerate(contexts):
                request = fx.Request(args.seed * 1000 + index, length - queries, length)
                if not request.allocate(pools, tp, rank, generator):
                    raise RuntimeError("bench page pool is too small")
                request.write_history(pools, tp, rank)
                requests.append(request)
            bucket = bucket_of(len(requests))
            slot = fx.GraphSlot(config, parallel, [layer], weights_by_layer, owner, recorder, pools,
                                queries=queries, bucket=bucket, max_seq_len=args.max_seq_len,
                                pool_handle=pool_handle)
            rows = slot.hidden.shape[0]
            hidden = torch.randn(rows, config.hidden_size, generator=hidden_gen, device="cuda").bfloat16()
            live = torch.arange(rank * rows, (rank + 1) * rows, device="cuda") < len(requests) * queries
            slot.load(requests, hidden * live[:, None])

            body = slot.forward

            latency = measure(body, flush, args.inner, args.replays)
            if args.kernels:
                table = kernel_table(body, flush)
                if rank == 0:
                    print("KERNELS " + json.dumps({"batch": len(contexts), "max_len": max(contexts),
                                                   "kernels": table[:25]}), flush=True)
            if rank == 0:
                print("BENCH " + json.dumps({
                    "q": queries, "batch": len(contexts), "bucket": bucket,
                    "min_len": min(contexts), "max_len": max(contexts), "layer_us": latency,
                }), flush=True)
            del slot
            for request in requests:
                request.release(pools)
            torch.cuda.empty_cache()
    dist.barrier()
    # NCCL teardown with captured graphs alive can hang in destroy_process_group; the run is over.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
