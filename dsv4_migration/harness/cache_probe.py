"""Synthetic cache-only latency bounds; not model TPOT or goodput."""

import argparse
import json
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.fp8.csa_cache import CsaLayerCache, CsaRequestSlots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--ctas", type=int, default=64)
    args = parser.parse_args()
    results = []
    for batch in (1, 16, 64):
        entries, topk, hot = 64, 1024, 2048
        blocks = batch * 512 + 1
        stride = ((entries * 584 + 575) // 576) * 576
        source_storage = torch.zeros(
            (blocks, stride), dtype=torch.uint8, pin_memory=True
        )
        source = source_storage.as_strided((blocks, entries, 584), (stride, 584, 1))
        requests = CsaRequestSlots(blocks, batch, "cuda")
        # Slightly more than the private-cache floor; selected IDs are beyond the resident prefix.
        budget = (
            batch * hot // entries * stride
            + blocks * entries * 4
            + batch * 100000
            + stride
        )
        cache = CsaLayerCache(
            source, requests, budget_bytes=budget, topk=topk, fetch_ctas=args.ctas
        )
        table = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda").reshape(
            batch, 1
        )
        requests.begin_prefill(table)
        requests.register(
            table, torch.full((batch,), 131072, dtype=torch.int32, device="cuda")
        )
        base = (
            cache.resident_tokens
            + torch.arange(batch, dtype=torch.int32, device="cuda")[:, None] * 30000
        )
        columns = torch.arange(topk, dtype=torch.int32, device="cuda")[None, :]
        step = torch.zeros(1, dtype=torch.int32, device="cuda")
        deferred = torch.full((batch,), -1, dtype=torch.int64, device="cuda")
        for changed in (0, 128, 1024):

            def run():
                step.add_(1)
                selected = (
                    base
                    + columns
                    + torch.where(columns >= topk - changed, (step % 16) * 1536, 0)
                )
                cache.prefetch(selected, deferred)
                cache.wait()

            warmup = torch.cuda.Stream()
            warmup.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(warmup)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(32):
                graph.replay()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            counts = cache.counts.sum(0).cpu().tolist()
            result = {
                "batch": batch,
                "topk": topk,
                "private_hot_entries": hot,
                "fetch_ctas": args.ctas,
                "changed_entries_per_request": changed,
                "counts_resident_hot_miss": counts,
                "cache_only_graph_ms": start.elapsed_time(end) / 100,
                "graph_includes_synthetic_index_update": True,
            }
            results.append(result)
            print(json.dumps(result), flush=True)
        del graph, cache, source, source_storage, requests
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
