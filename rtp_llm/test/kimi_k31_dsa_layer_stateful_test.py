"""K3.1 DSA decode: long replay runs of many layers and buckets, for state carried between launches.

All DSA layers of the model (24 at full depth) run in one graph per bucket and share one workspace
owner and the step's attention impls, as in the served model; several bucket graphs are captured
from one memory pool. The run replays them in random order, interleaved with eager calls, for many
launches. Each graph keeps fixed inputs and is prepared before every replay, as the runner does, so
every replay must select the same positions as that graph's first replay and match its output within
--replay-tolerance (bit-exact replays are counted; selection order may vary between replays);
eager calls must match it within tolerance. Every --refresh launches one graph gets new requests and
new hidden rows, and its last layer passes the layer test's fp32 checks.

  torchrun --no-python --nproc_per_node=8 bazel-bin/rtp_llm/test/kimi_k31_dsa_layer_stateful_test
      --config-json <K3.1 checkpoint>/config.json --queries 4 --launches 10000
"""

import argparse
import faulthandler
import json
import math
import os
import random
import sys

import torch
import torch.distributed as dist

try:
    from rtp_llm.test import kimi_k31_dsa_layer_fixture as fx
    from rtp_llm.test import kimi_k31_dsa_layer_test as layer_test
except ImportError:  # run as a plain script next to the fixture (no Bazel runfiles)
    import kimi_k31_dsa_layer_fixture as fx
    import kimi_k31_dsa_layer_test as layer_test


class Bucket:
    def __init__(self, slot, server, hidden_gen):
        self.slot, self.server, self.hidden_gen = slot, server, hidden_gen
        self.first = None

    def refresh(self, batch, pools, tp, rank, hidden_size):
        requests = self.server.step(batch)
        for request in requests:
            if request.fresh:
                request.write_history(pools, tp, rank)
        rows = self.slot.hidden.shape[0]
        hidden = torch.randn(rows, hidden_size, generator=self.hidden_gen, device="cuda").bfloat16()
        live = torch.arange(rank * rows, (rank + 1) * rows, device="cuda") < len(requests) * self.slot.queries
        self.requests, self.hidden = requests, hidden * live[:, None]
        self.step = layer_test.Step(requests, self.slot.queries, self.slot.padded)
        self.first = [output.clone() for output in self.replay()]
        self.first_selected = self.selected()
        self.first_inner = self.inner()

    def inner(self):
        """The last layer's live rows at the Indexer and sparse-MLA boundaries, as recorded."""
        rows = self.step.rows
        return {name: self.slot.recorded[name][:rows].clone() for name in ("selected", "sfa_query", "sfa_output")}

    def selected(self):
        """The last layer's selection per live row as a set (sorted)."""
        rows = self.step.rows
        return self.slot.recorded["selected"][:rows].sort(dim=1).values.clone()

    def replay(self):
        # The runner prepares every replay (prepare_cuda_graph), also when the inputs repeat:
        # other buckets' preparations in between may rewrite shared metadata.
        self.slot.load(self.requests, self.hidden)
        return self.slot.replay()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--queries", type=int, default=4)
    parser.add_argument("--buckets", default="[1, 8, 32, 128]")
    parser.add_argument("--layers", type=int, default=0, help="DSA layers to stack (0: all of a 93-layer model)")
    parser.add_argument("--launches", type=int, default=10000)
    parser.add_argument("--eager-every", type=int, default=97)
    parser.add_argument("--refresh", type=int, default=500, help="launches between new inputs and fp32 checks")
    parser.add_argument("--max-seq-len", type=int, default=1 << 20)
    parser.add_argument("--max-context", type=int, default=128 << 10, help="longest request of the run")
    parser.add_argument("--pool-pages", type=int, default=48 * 1024)
    parser.add_argument("--eager-tolerance", type=float, default=2e-2)
    parser.add_argument("--replay-tolerance", type=float, default=1e-2,
                        help="replay vs first replay, max error / max value (selections must match exactly)")
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--step-timeout", type=int, default=1800)
    args = parser.parse_args()
    parallel = fx.init_distributed(int(os.environ["WORLD_SIZE"]))
    tp, rank = parallel.tp_size, parallel.tp_rank
    faulthandler.enable(file=sys.stderr, all_threads=True)
    queries = args.queries
    buckets = [b for b in json.loads(args.buckets) if b * queries <= fx.MAX_TOKENS]
    config = fx.load_config(args.config_json, max_seq_len=args.max_seq_len, queries=queries, layers=93)
    indices = list(config.k3_runtime_config.dsa.layers)
    indices = indices[: args.layers] if args.layers else indices
    weights_by_layer = [None] * config.num_layers
    layers, recorders, globals_by_layer = [], [], {}
    for index in indices:
        weights, global_weights = fx.layer_weights(config, parallel, seed=args.seed + index)
        weights_by_layer[index] = weights
        globals_by_layer[index] = global_weights
        layers.append(fx.build_layer(config, parallel, weights, index))
        recorders.append(fx.Recorder(layers[-1]))
    unit = fx.sp_unit(tp, queries)
    owner = fx.build_owner(
        config, parallel, [(i, weights_by_layer[i]) for i in indices],
        max_requests=max(math.ceil(b / unit) * unit for b in buckets), queries=queries,
    )
    pools = {tag: fx.Pool(tag, args.pool_pages, "cuda") for tag in fx.TAGS}
    pool_handle = torch.cuda.graph_pool_handle()
    decisions = random.Random(args.seed)
    graphs = []
    for number, bucket in enumerate(buckets):
        slot = fx.GraphSlot(config, parallel, layers, weights_by_layer, owner, recorders[-1], pools,
                            queries=queries, bucket=bucket, max_seq_len=args.max_seq_len, pool_handle=pool_handle)
        server = layer_test.Server(queries, pools, max_seq_len=args.max_context, tp=tp, rank=rank,
                                   seed=args.seed * 10 + number, budget=(args.pool_pages - 1) // len(buckets))
        graphs.append(Bucket(slot, server, torch.Generator(device="cuda").manual_seed(args.seed * 13 + rank + 8 * number)))
    for index, graph in enumerate(graphs):
        low = buckets[index - 1] + 1 if index else 1
        graph.refresh(decisions.randint(low, graph.slot.bucket), pools, tp, rank, config.hidden_size)
    counts = {"replays": 0, "replay_mismatch": 0, "bit_exact": 0, "max_replay_rel": 0.0,
              "bit_exact_selected": 0, "bit_exact_sfa_query": 0, "bit_exact_sfa_output": 0,
              "eager": 0, "eager_mismatch": 0, "checks": 0, "check_failures": 0}
    first_failure = None
    for launch in range(args.launches):
        faulthandler.dump_traceback_later(args.step_timeout, exit=True)
        graph = graphs[decisions.randrange(len(graphs))]
        if launch % args.eager_every == args.eager_every - 1:
            # The same inputs through freshly built eager impls, between the graphs' replays.
            inputs = fx.attention_inputs(graph.step.requests, queries, graph.slot.padded, graph_width=None,
                                         tp=tp, rank=rank, device="cuda")
            fmha = fx.build_fmha(config, parallel, weights_by_layer, inputs, owner, recorders[-1], graph=False)
            outputs = [layer(graph.slot.hidden, fmha, graph.slot.caches[layer.layer_idx], inputs["full"], None)
                       for layer in layers]
            worst = max(((o.float() - f.float()).abs().max() / f.float().abs().max().clamp(min=1e-6)).item()
                        for o, f in zip(outputs, graph.first))
            counts["eager"] += 1
            counts["eager_mismatch"] += int(worst > args.eager_tolerance)
            if worst > args.eager_tolerance and first_failure is None:
                first_failure = f"launch {launch}: eager differs from replay by {worst:.3g}"
            continue
        if launch % args.refresh == args.refresh - 1:
            index = graphs.index(graph)
            low = buckets[index - 1] + 1 if index else 1
            graph.refresh(decisions.randint(low, graph.slot.bucket), pools, tp, rank, config.hidden_size)
            recorded = graph.slot.recorded
            problems, _ = layer_test.check_selection(graph.step, recorded, pools, tp, rank)
            attention, stats = layer_test.check_attention(
                graph.step, recorded, pools, globals_by_layer[indices[-1]], tp, rank,
                graph.slot.fmha["full"].fmha_impl.scale,
            )
            problems += attention
            counts["checks"] += 1
            counts["check_failures"] += int(bool(problems))
            if problems and first_failure is None:
                first_failure = f"launch {launch}: {problems[:2]}"
            continue
        outputs = graph.replay()
        counts["replays"] += 1
        counts["bit_exact"] += int(all(torch.equal(o, f) for o, f in zip(outputs, graph.first)))
        # Where a replay first departs from the first one: selection order, MLA input or output.
        for name, value in graph.inner().items():
            counts["bit_exact_" + name] += int(torch.equal(value, graph.first_inner[name]))
        worst = max(((o.float() - f.float()).abs().max() / f.float().abs().max().clamp(min=1e-6)).item()
                    for o, f in zip(outputs, graph.first))
        counts["max_replay_rel"] = max(counts["max_replay_rel"], round(worst, 6))
        same_sets = torch.equal(graph.selected(), graph.first_selected)
        if not same_sets or worst > args.replay_tolerance:
            counts["replay_mismatch"] += 1
            if first_failure is None:
                first_failure = (f"launch {launch}: bucket {graph.slot.bucket} replay differs from its first "
                                 f"(selection equal {same_sets}, output rel {worst:.3g})")
        if rank == 0 and launch % 1000 == 0:
            print("PROGRESS " + json.dumps({"launch": launch, **counts}), flush=True)
    faulthandler.cancel_dump_traceback_later()
    totals = torch.tensor([counts["replay_mismatch"], counts["eager_mismatch"], counts["check_failures"]],
                          device="cuda")
    dist.all_reduce(totals)
    report = {"rank": rank, "layers": len(layers), "buckets": buckets, **counts, "first_failure": first_failure}
    if rank == 0 or first_failure:
        print("STATEFUL " + json.dumps(report), flush=True)
    if totals.sum().item() and rank == 0:
        print(f"stateful run failed: {totals.tolist()} (replay, eager, check)", flush=True)
    dist.barrier()
    # NCCL teardown with captured graphs alive can hang in destroy_process_group; the run is over.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if totals.sum().item() else 0)


if __name__ == "__main__":
    main()
